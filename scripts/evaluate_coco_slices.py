"""Evaluate COCO predictions across common robustness slices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_coco_predictions import METRIC_NAMES, evaluate_coco_predictions
from scripts.filter_coco_annotations import filter_coco_annotations


DEFAULT_SLICES = ("all", "offcenter", "center", "small", "medium", "large")
FIELDNAMES = ("slice", "images", "annotations", *METRIC_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slices", nargs="+", choices=DEFAULT_SLICES, default=list(DEFAULT_SLICES))
    parser.add_argument("--center-radius", type=float, default=0.25)
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    parser.add_argument("--large-area-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_coco_slices(
        annotation_json=args.annotations,
        prediction_json=args.predictions,
        slices=tuple(args.slices),
        center_radius=args.center_radius,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_slice_cocoeval: {args.out}")
    for row in rows:
        print(
            f"{row['slice']}: images={row['images']} annotations={row['annotations']} "
            f"ap={row['ap']:.4f} ap50={row['ap50']:.4f} ap75={row['ap75']:.4f}"
        )


def evaluate_coco_slices(
    annotation_json: Path,
    prediction_json: Path,
    slices: tuple[str, ...] = DEFAULT_SLICES,
    center_radius: float = 0.25,
    small_area_ratio: float = 0.05,
    large_area_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    rows = []
    predictions = json.loads(prediction_json.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for slice_name in slices:
            filtered = filter_coco_annotations(
                annotation_json,
                slice_name=slice_name,
                center_radius=center_radius,
                small_area_ratio=small_area_ratio,
                large_area_ratio=large_area_ratio,
            )
            slice_path = tmp_root / f"{slice_name}.json"
            slice_path.write_text(json.dumps(filtered), encoding="utf-8")
            slice_predictions_path = tmp_root / f"{slice_name}_predictions.json"
            image_ids = {int(image["id"]) for image in filtered["images"]}
            slice_predictions = [row for row in predictions if int(row["image_id"]) in image_ids]
            slice_predictions_path.write_text(json.dumps(slice_predictions), encoding="utf-8")
            metrics = (
                evaluate_coco_predictions(slice_path, slice_predictions_path)
                if filtered["annotations"]
                else {name: 0.0 for name in METRIC_NAMES}
            )
            rows.append(
                {
                    "slice": slice_name,
                    "images": len(filtered["images"]),
                    "annotations": len(filtered["annotations"]),
                    **metrics,
                }
            )
    return rows


if __name__ == "__main__":
    main()
