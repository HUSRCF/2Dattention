"""Measure category coverage gaps under global-topK vs per-category-topK recall."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_coco_score_iou_correlation import xyxy_iou
from scripts.evaluate_coco_prediction_recall import SLICES, annotation_rows, predictions_grouped_by_image


FIELDNAMES = (
    "slice",
    "top_k",
    "iou_threshold",
    "gt_count",
    "global_topk_loc_recall",
    "global_topk_class_recall",
    "per_category_topk_class_recall",
    "global_category_retention",
    "ranking_gap",
    "mean_global_loc_best_iou",
    "mean_global_class_best_iou",
    "mean_per_category_class_best_iou",
)

PER_CATEGORY_FIELDNAMES = (
    "category_id",
    "category_name",
    "top_k",
    "iou_threshold",
    "gt_count",
    "global_topk_loc_recall",
    "global_topk_class_recall",
    "per_category_topk_class_recall",
    "global_category_retention",
    "loc_to_class_gap",
    "ranking_gap",
    "mean_global_loc_best_iou",
    "mean_global_class_best_iou",
    "mean_per_category_class_best_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-category-out", type=Path)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--slices", nargs="+", choices=SLICES, default=list(SLICES))
    parser.add_argument("--center-radius", type=float, default=0.25)
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    parser.add_argument("--large-area-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_category_coverage_gap(
        annotation_json=args.annotations,
        prediction_json=args.predictions,
        top_ks=tuple(sorted(set(args.top_k))),
        iou_thresholds=tuple(args.iou_thresholds),
        slices=tuple(args.slices),
        center_radius=args.center_radius,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.per_category_out is not None:
        category_rows = analyze_per_category_coverage_gap(
            annotation_json=args.annotations,
            prediction_json=args.predictions,
            top_ks=tuple(sorted(set(args.top_k))),
            iou_thresholds=tuple(args.iou_thresholds),
            center_radius=args.center_radius,
            small_area_ratio=args.small_area_ratio,
            large_area_ratio=args.large_area_ratio,
        )
        args.per_category_out.parent.mkdir(parents=True, exist_ok=True)
        with args.per_category_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PER_CATEGORY_FIELDNAMES), lineterminator="\n")
            writer.writeheader()
            writer.writerows(category_rows)
        print(f"saved_per_category_coverage_gap: {args.per_category_out}")
    print(f"saved_category_coverage_gap: {args.out}")
    for row in rows:
        if row["slice"] == "all" and row["iou_threshold"] == 0.5:
            print(
                f"all@{row['top_k']}: loc={row['global_topk_loc_recall']:.4f} "
                f"class_global={row['global_topk_class_recall']:.4f} "
                f"class_percat={row['per_category_topk_class_recall']:.4f}"
            )


def analyze_category_coverage_gap(
    *,
    annotation_json: Path,
    prediction_json: Path,
    top_ks: tuple[int, ...] = (10, 50, 100),
    iou_thresholds: tuple[float, ...] = (0.5, 0.75),
    slices: tuple[str, ...] = SLICES,
    center_radius: float = 0.25,
    small_area_ratio: float = 0.05,
    large_area_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    annotations = json.loads(annotation_json.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_json.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in annotations.get("images", [])}
    gt_rows = annotation_rows(
        annotations,
        images,
        center_radius=center_radius,
        small_area_ratio=small_area_ratio,
        large_area_ratio=large_area_ratio,
    )
    predictions_by_image = predictions_grouped_by_image(predictions)
    accumulators = {
        (slice_name, top_k): {
            "global_loc": [],
            "global_class": [],
            "per_category_class": [],
        }
        for slice_name in slices
        for top_k in top_ks
    }

    for gt in gt_rows:
        image_predictions = predictions_by_image.get(gt["image_id"], [])
        for top_k in top_ks:
            global_topk = image_predictions[:top_k]
            per_category_topk = [row for row in image_predictions if row["category_id"] == gt["category_id"]][
                :top_k
            ]
            global_loc_best = best_iou(gt, global_topk, require_category=False)
            global_class_best = best_iou(gt, global_topk, require_category=True)
            per_category_class_best = best_iou(gt, per_category_topk, require_category=False)
            for slice_name in slices:
                if gt["slices"][slice_name]:
                    bucket = accumulators[(slice_name, top_k)]
                    bucket["global_loc"].append(global_loc_best)
                    bucket["global_class"].append(global_class_best)
                    bucket["per_category_class"].append(per_category_class_best)

    rows = []
    for slice_name in slices:
        for top_k in top_ks:
            bucket = accumulators[(slice_name, top_k)]
            gt_count = len(bucket["global_loc"])
            for threshold in iou_thresholds:
                global_loc_recall = recall_at(bucket["global_loc"], threshold)
                global_class_recall = recall_at(bucket["global_class"], threshold)
                per_category_class_recall = recall_at(bucket["per_category_class"], threshold)
                rows.append(
                    {
                        "slice": slice_name,
                        "top_k": top_k,
                        "iou_threshold": threshold,
                        "gt_count": gt_count,
                        "global_topk_loc_recall": global_loc_recall,
                        "global_topk_class_recall": global_class_recall,
                        "per_category_topk_class_recall": per_category_class_recall,
                        "global_category_retention": safe_ratio(global_class_recall, global_loc_recall),
                        "ranking_gap": per_category_class_recall - global_class_recall,
                        "mean_global_loc_best_iou": mean(bucket["global_loc"]),
                        "mean_global_class_best_iou": mean(bucket["global_class"]),
                        "mean_per_category_class_best_iou": mean(bucket["per_category_class"]),
                    }
                )
    return rows


def analyze_per_category_coverage_gap(
    *,
    annotation_json: Path,
    prediction_json: Path,
    top_ks: tuple[int, ...] = (10, 50, 100),
    iou_thresholds: tuple[float, ...] = (0.5, 0.75),
    center_radius: float = 0.25,
    small_area_ratio: float = 0.05,
    large_area_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    annotations = json.loads(annotation_json.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_json.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in annotations.get("images", [])}
    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in annotations.get("categories", [])
    }
    gt_rows = annotation_rows(
        annotations,
        images,
        center_radius=center_radius,
        small_area_ratio=small_area_ratio,
        large_area_ratio=large_area_ratio,
    )
    predictions_by_image = predictions_grouped_by_image(predictions)
    accumulators: dict[tuple[int, int], dict[str, list[float]]] = {}
    for gt in gt_rows:
        category_id = int(gt["category_id"])
        image_predictions = predictions_by_image.get(gt["image_id"], [])
        for top_k in top_ks:
            global_topk = image_predictions[:top_k]
            per_category_topk = [row for row in image_predictions if row["category_id"] == category_id][:top_k]
            bucket = accumulators.setdefault(
                (category_id, top_k),
                {"global_loc": [], "global_class": [], "per_category_class": []},
            )
            bucket["global_loc"].append(best_iou(gt, global_topk, require_category=False))
            bucket["global_class"].append(best_iou(gt, global_topk, require_category=True))
            bucket["per_category_class"].append(best_iou(gt, per_category_topk, require_category=False))

    rows = []
    for category_id, top_k in sorted(accumulators):
        bucket = accumulators[(category_id, top_k)]
        gt_count = len(bucket["global_loc"])
        for threshold in iou_thresholds:
            global_loc_recall = recall_at(bucket["global_loc"], threshold)
            global_class_recall = recall_at(bucket["global_class"], threshold)
            per_category_class_recall = recall_at(bucket["per_category_class"], threshold)
            rows.append(
                {
                    "category_id": category_id,
                    "category_name": categories.get(category_id, str(category_id)),
                    "top_k": top_k,
                    "iou_threshold": threshold,
                    "gt_count": gt_count,
                    "global_topk_loc_recall": global_loc_recall,
                    "global_topk_class_recall": global_class_recall,
                    "per_category_topk_class_recall": per_category_class_recall,
                    "global_category_retention": safe_ratio(global_class_recall, global_loc_recall),
                    "loc_to_class_gap": global_loc_recall - global_class_recall,
                    "ranking_gap": per_category_class_recall - global_class_recall,
                    "mean_global_loc_best_iou": mean(bucket["global_loc"]),
                    "mean_global_class_best_iou": mean(bucket["global_class"]),
                    "mean_per_category_class_best_iou": mean(bucket["per_category_class"]),
                }
            )
    rows.sort(
        key=lambda row: (
            row["iou_threshold"],
            row["top_k"],
            -row["loc_to_class_gap"],
            -row["gt_count"],
            row["category_id"],
        )
    )
    return rows


def best_iou(gt: dict[str, Any], predictions: list[dict[str, Any]], *, require_category: bool) -> float:
    values = (
        xyxy_iou(gt["box"], prediction["box"])
        for prediction in predictions
        if not require_category or prediction["category_id"] == gt["category_id"]
    )
    return max(values, default=0.0)


def recall_at(values: list[float], threshold: float) -> float:
    return 0.0 if not values else sum(value >= threshold for value in values) / len(values)


def mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else numerator / denominator


if __name__ == "__main__":
    main()
