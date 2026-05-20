"""Analyze score-to-IoU alignment for COCO predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rescore_coco_predictions_by_oracle_iou import gt_boxes_by_image, xywh_to_xyxy


FIELDNAMES = (
    "predictions",
    "images",
    "mean_score",
    "mean_nearest_iou",
    "pearson_score_iou",
    "spearman_score_iou",
    "top10_mean_iou",
    "top50_mean_iou",
    "top100_mean_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--class-aware",
        action="store_true",
        help="Compute nearest IoU only against GT boxes with the same category id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = analyze_score_iou(args.annotations, args.predictions, class_aware=args.class_aware)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print(f"saved_score_iou_analysis: {args.out}")
    print(
        f"predictions={row['predictions']} pearson={row['pearson_score_iou']:.4f} "
        f"spearman={row['spearman_score_iou']:.4f} top100_iou={row['top100_mean_iou']:.4f}"
    )


def analyze_score_iou(
    annotation_json: Path,
    prediction_json: Path,
    *,
    class_aware: bool = False,
) -> dict[str, float | int]:
    annotations = json.loads(annotation_json.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_json.read_text(encoding="utf-8"))
    gt_by_image = gt_boxes_by_image(annotations)
    records = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        pred_category = int(prediction["category_id"])
        gt_rows = gt_by_image.get(image_id, [])
        if class_aware:
            gt_rows = [row for row in gt_rows if int(row["category_id"]) == pred_category]
        pred_box = xywh_to_xyxy([float(value) for value in prediction["bbox"]])
        nearest_iou = max((xyxy_iou(pred_box, row["box"]) for row in gt_rows), default=0.0)
        records.append(
            {
                "image_id": image_id,
                "score": float(prediction.get("score", 0.0)),
                "nearest_iou": nearest_iou,
            }
        )
    scores = [row["score"] for row in records]
    ious = [row["nearest_iou"] for row in records]
    sorted_records = sorted(records, key=lambda row: row["score"], reverse=True)
    return {
        "predictions": len(records),
        "images": len({row["image_id"] for row in records}),
        "mean_score": safe_mean(scores),
        "mean_nearest_iou": safe_mean(ious),
        "pearson_score_iou": pearson(scores, ious),
        "spearman_score_iou": spearman(scores, ious),
        "top10_mean_iou": topk_mean_iou(sorted_records, 10),
        "top50_mean_iou": topk_mean_iou(sorted_records, 50),
        "top100_mean_iou": topk_mean_iou(sorted_records, 100),
    }


def topk_mean_iou(records: list[dict[str, float | int]], k: int) -> float:
    return safe_mean([float(row["nearest_iou"]) for row in records[:k]])


def xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0.0 else inter_area / union


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = safe_mean(xs)
    mean_y = safe_mean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    denom_x = math.sqrt(sum(x * x for x in centered_x))
    denom_y = math.sqrt(sum(y * y for y in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return numerator / (denom_x * denom_y)


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return pearson(ranks(xs), ranks(ys))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks_out = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        avg_rank = (start + end - 1) / 2.0
        for index in range(start, end):
            ranks_out[indexed[index][0]] = avg_rank
        start = end
    return ranks_out


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
