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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_coco_predictions(args.annotations, args.predictions)
    print(",".join(METRIC_NAMES))
    print(",".join(f"{metrics[name]:.6f}" for name in METRIC_NAMES))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(METRIC_NAMES))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"saved_csv: {args.out}")


def evaluate_coco_predictions(annotation_json: Path, prediction_json: Path) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - depends on environment extras.
        raise RuntimeError("pycocotools is required for standard COCO evaluation") from exc

    predictions = json.loads(prediction_json.read_text(encoding="utf-8"))
    if not predictions:
        return {name: 0.0 for name in METRIC_NAMES}

    with tempfile.TemporaryDirectory() as tmp_dir:
        normalized_annotations = normalize_coco_annotations(annotation_json, Path(tmp_dir) / "annotations.json")
        with redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(normalized_annotations))
            coco_dt = coco_gt.loadRes(str(prediction_json))
            evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
    stats = [float(value) for value in evaluator.stats.tolist()]
    return dict(zip(METRIC_NAMES, stats[: len(METRIC_NAMES)], strict=True))


def normalize_coco_annotations(source: Path, destination: Path) -> Path:
    data = json.loads(source.read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    main()
