"""Summarize high-IoU category confusions for COCO prediction JSON files."""

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
from scripts.evaluate_coco_prediction_recall import annotation_rows, predictions_grouped_by_image


FIELDNAMES = (
    "gt_category_id",
    "gt_category_name",
    "pred_category_id",
    "pred_category_name",
    "top_k",
    "iou_threshold",
    "gt_count",
    "loc_matched_count",
    "pair_count",
    "pair_fraction_of_gt",
    "pair_fraction_of_loc_matched",
    "mean_iou",
    "mean_score",
    "is_correct_category",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--center-radius", type=float, default=0.25)
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    parser.add_argument("--large-area-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_category_confusions(
        annotation_json=args.annotations,
        prediction_json=args.predictions,
        top_k=args.top_k,
        iou_threshold=args.iou_threshold,
        center_radius=args.center_radius,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_category_confusions: {args.out}")
    for row in rows[:12]:
        verdict = "correct" if row["is_correct_category"] else "wrong"
        print(
            f"{row['gt_category_name']} -> {row['pred_category_name']} "
            f"{row['pair_count']}/{row['gt_count']} {verdict} "
            f"mean_iou={row['mean_iou']:.3f}"
        )


def analyze_category_confusions(
    *,
    annotation_json: Path,
    prediction_json: Path,
    top_k: int = 100,
    iou_threshold: float = 0.5,
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
    gt_counts: dict[int, int] = {}
    loc_counts: dict[int, int] = {}
    pair_stats: dict[tuple[int, int], dict[str, Any]] = {}

    for gt in gt_rows:
        gt_category_id = int(gt["category_id"])
        gt_counts[gt_category_id] = gt_counts.get(gt_category_id, 0) + 1
        image_predictions = predictions_by_image.get(gt["image_id"], [])[:top_k]
        best_prediction = None
        best_iou = 0.0
        for prediction in image_predictions:
            iou = xyxy_iou(gt["box"], prediction["box"])
            if iou > best_iou:
                best_iou = iou
                best_prediction = prediction
        if best_prediction is None or best_iou < iou_threshold:
            continue
        pred_category_id = int(best_prediction["category_id"])
        loc_counts[gt_category_id] = loc_counts.get(gt_category_id, 0) + 1
        key = (gt_category_id, pred_category_id)
        stats = pair_stats.setdefault(key, {"count": 0, "iou_sum": 0.0, "score_sum": 0.0})
        stats["count"] += 1
        stats["iou_sum"] += best_iou
        stats["score_sum"] += float(best_prediction.get("score", 0.0))

    rows = []
    for (gt_category_id, pred_category_id), stats in pair_stats.items():
        gt_count = gt_counts.get(gt_category_id, 0)
        loc_count = loc_counts.get(gt_category_id, 0)
        pair_count = int(stats["count"])
        rows.append(
            {
                "gt_category_id": gt_category_id,
                "gt_category_name": categories.get(gt_category_id, str(gt_category_id)),
                "pred_category_id": pred_category_id,
                "pred_category_name": categories.get(pred_category_id, str(pred_category_id)),
                "top_k": top_k,
                "iou_threshold": iou_threshold,
                "gt_count": gt_count,
                "loc_matched_count": loc_count,
                "pair_count": pair_count,
                "pair_fraction_of_gt": safe_ratio(pair_count, gt_count),
                "pair_fraction_of_loc_matched": safe_ratio(pair_count, loc_count),
                "mean_iou": safe_ratio(float(stats["iou_sum"]), pair_count),
                "mean_score": safe_ratio(float(stats["score_sum"]), pair_count),
                "is_correct_category": gt_category_id == pred_category_id,
            }
        )
    rows.sort(
        key=lambda row: (
            row["is_correct_category"],
            -row["pair_count"],
            -row["pair_fraction_of_loc_matched"],
            row["gt_category_id"],
            row["pred_category_id"],
        )
    )
    return rows


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else numerator / denominator


if __name__ == "__main__":
    main()
