"""Evaluate exported COCO detection predictions with pycocotools."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path


METRIC_NAMES = (
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar1",
    "ar10",
    "ar100",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--class-agnostic",
        action="store_true",
        help="Evaluate localization by remapping all GT and prediction categories to one foreground class.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_coco_predictions(args.annotations, args.predictions, class_agnostic=args.class_agnostic)
    print(",".join(METRIC_NAMES))
    print(",".join(f"{metrics[name]:.6f}" for name in METRIC_NAMES))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(METRIC_NAMES))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"saved_csv: {args.out}")


def evaluate_coco_predictions(
    annotation_json: Path,
    prediction_json: Path,
    class_agnostic: bool = False,
) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - depends on environment extras.
        raise RuntimeError("pycocotools is required for standard COCO evaluation") from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        normalized_annotations = normalize_coco_annotations(annotation_json, Path(tmp_dir) / "annotations.json")
        normalized_predictions = normalize_coco_predictions(
            prediction_json,
            Path(tmp_dir) / "predictions.json",
            class_agnostic=class_agnostic,
        )
        if class_agnostic:
            normalized_annotations = make_class_agnostic_annotations(
                normalized_annotations,
                Path(tmp_dir) / "annotations_class_agnostic.json",
            )
        with redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(normalized_annotations))
        predictions = json.loads(normalized_predictions.read_text(encoding="utf-8"))
        if not predictions:
            return {name: 0.0 for name in METRIC_NAMES}
        with redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(str(normalized_predictions))
            evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
    stats = [float(value) for value in evaluator.stats.tolist()]
    return dict(zip(METRIC_NAMES, stats[: len(METRIC_NAMES)], strict=True))


def normalize_coco_annotations(source: Path, destination: Path) -> Path:
    data = json.loads(source.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise KeyError(f"COCO annotation JSON missing required key: {key}")
    changed = False
    if "info" not in data:
        data["info"] = {"description": "normalized for pycocotools loadRes"}
        changed = True
    if "licenses" not in data:
        data["licenses"] = []
        changed = True
    if not changed:
        return source
    destination.write_text(json.dumps(data), encoding="utf-8")
    return destination


def normalize_coco_predictions(source: Path, destination: Path, class_agnostic: bool) -> Path:
    if not class_agnostic:
        return source
    predictions = json.loads(source.read_text(encoding="utf-8"))
    for prediction in predictions:
        prediction["category_id"] = 1
    destination.write_text(json.dumps(predictions), encoding="utf-8")
    return destination


def make_class_agnostic_annotations(source: Path, destination: Path) -> Path:
    data = json.loads(source.read_text(encoding="utf-8"))
    for annotation in data["annotations"]:
        annotation["category_id"] = 1
    data["categories"] = [{"id": 1, "name": "object", "supercategory": "object"}]
    destination.write_text(json.dumps(data), encoding="utf-8")
    return destination


if __name__ == "__main__":
    main()
