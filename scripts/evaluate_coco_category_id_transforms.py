"""Evaluate simple category-id transforms for COCO prediction JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_coco_predictions import METRIC_NAMES, evaluate_coco_predictions


FIELDNAMES = (
    "variant",
    "offset",
    "predictions",
    "in_gt_category_predictions",
    "in_gt_category_fraction",
    *METRIC_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=[-3, -2, -1, 0, 1, 2, 3])
    parser.add_argument(
        "--drop-invalid-categories",
        action="store_true",
        help="Drop transformed predictions whose category is not present in the annotation JSON.",
    )
    parser.add_argument(
        "--include-class-agnostic",
        action="store_true",
        help="Add a localization-only class-agnostic row for reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_category_id_transforms(
        annotation_json=args.annotations,
        prediction_json=args.predictions,
        offsets=tuple(args.offsets),
        drop_invalid_categories=args.drop_invalid_categories,
        include_class_agnostic=args.include_class_agnostic,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_category_transform_eval: {args.out}")
    for row in rows:
        print(
            f"{row['variant']}: ap={row['ap']:.4f} ap50={row['ap50']:.4f} "
            f"in_gt={row['in_gt_category_fraction']:.4f}"
        )


def evaluate_category_id_transforms(
    *,
    annotation_json: Path,
    prediction_json: Path,
    offsets: tuple[int, ...],
    drop_invalid_categories: bool = False,
    include_class_agnostic: bool = False,
) -> list[dict[str, float | int | str]]:
    annotations = json.loads(annotation_json.read_text(encoding="utf-8"))
    predictions = filter_predictions_to_annotation_images(
        json.loads(prediction_json.read_text(encoding="utf-8")),
        annotation_image_ids(annotations),
    )
    gt_categories = annotation_category_ids(annotations)
    rows: list[dict[str, float | int | str]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        filtered_prediction_json = tmp_root / "predictions_filtered.json"
        filtered_prediction_json.write_text(json.dumps(predictions), encoding="utf-8")
        if include_class_agnostic:
            metrics = evaluate_coco_predictions(annotation_json, filtered_prediction_json, class_agnostic=True)
            rows.append(
                {
                    "variant": "class_agnostic",
                    "offset": "",
                    **prediction_category_coverage(predictions, gt_categories),
                    **metrics,
                }
            )
        for offset in offsets:
            variant = f"offset_{offset:+d}"
            shifted = shift_prediction_categories(
                predictions,
                offset=offset,
                valid_categories=gt_categories if drop_invalid_categories else None,
            )
            shifted_path = tmp_root / f"{variant}.json"
            shifted_path.write_text(json.dumps(shifted), encoding="utf-8")
            metrics = evaluate_coco_predictions(annotation_json, shifted_path)
            rows.append(
                {
                    "variant": variant,
                    "offset": offset,
                    **prediction_category_coverage(shifted, gt_categories),
                    **metrics,
                }
            )
    return rows


def annotation_category_ids(data: dict[str, Any]) -> set[int]:
    return {int(category["id"]) for category in data.get("categories", [])}


def annotation_image_ids(data: dict[str, Any]) -> set[int]:
    return {int(image["id"]) for image in data.get("images", [])}


def filter_predictions_to_annotation_images(
    predictions: list[dict[str, Any]],
    image_ids: set[int],
) -> list[dict[str, Any]]:
    return [prediction for prediction in predictions if int(prediction["image_id"]) in image_ids]


def shift_prediction_categories(
    predictions: list[dict[str, Any]],
    *,
    offset: int,
    valid_categories: set[int] | None = None,
) -> list[dict[str, Any]]:
    shifted = []
    for prediction in predictions:
        row = dict(prediction)
        row["category_id"] = int(row["category_id"]) + offset
        if valid_categories is not None and row["category_id"] not in valid_categories:
            continue
        shifted.append(row)
    return shifted


def prediction_category_coverage(
    predictions: list[dict[str, Any]],
    gt_categories: set[int],
) -> dict[str, float | int]:
    total = len(predictions)
    in_gt = sum(1 for prediction in predictions if int(prediction["category_id"]) in gt_categories)
    return {
        "predictions": total,
        "in_gt_category_predictions": in_gt,
        "in_gt_category_fraction": 0.0 if total == 0 else in_gt / total,
    }


if __name__ == "__main__":
    main()
