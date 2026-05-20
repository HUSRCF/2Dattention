"""Evaluate top-k GT recall from an exported COCO prediction JSON."""

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
from scripts.rescore_coco_predictions_by_oracle_iou import xywh_to_xyxy


SLICES = ("all", "offcenter", "center", "small", "medium", "large")
FIELDNAMES = ("slice", "top_k", "iou_threshold", "gt_count", "recall", "mean_best_iou", "class_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 50, 100, 300])
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--slices", nargs="+", choices=SLICES, default=list(SLICES))
    parser.add_argument("--class-aware", action="store_true")
    parser.add_argument("--center-radius", type=float, default=0.25)
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    parser.add_argument("--large-area-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_prediction_recall(
        annotation_json=args.annotations,
        prediction_json=args.predictions,
        top_ks=tuple(sorted(set(args.top_k))),
        iou_thresholds=tuple(args.iou_thresholds),
        slices=tuple(args.slices),
        class_aware=args.class_aware,
        center_radius=args.center_radius,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_prediction_recall: {args.out}")
    for row in rows:
        if row["slice"] == "all" and row["iou_threshold"] == 0.5:
            print(f"all@{row['top_k']}: recall50={row['recall']:.4f} mean_best_iou={row['mean_best_iou']:.4f}")


def evaluate_prediction_recall(
    annotation_json: Path,
    prediction_json: Path,
    *,
    top_ks: tuple[int, ...] = (10, 50, 100, 300),
    iou_thresholds: tuple[float, ...] = (0.5, 0.75),
    slices: tuple[str, ...] = SLICES,
    class_aware: bool = False,
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
    accumulators = {(slice_name, top_k): [] for slice_name in slices for top_k in top_ks}
    for gt in gt_rows:
        image_predictions = predictions_by_image.get(gt["image_id"], [])
        if class_aware:
            image_predictions = [row for row in image_predictions if row["category_id"] == gt["category_id"]]
        for top_k in top_ks:
            best_iou = max(
                (xyxy_iou(gt["box"], prediction["box"]) for prediction in image_predictions[:top_k]),
                default=0.0,
            )
            for slice_name in slices:
                if gt["slices"][slice_name]:
                    accumulators[(slice_name, top_k)].append(best_iou)
    rows = []
    for slice_name in slices:
        for top_k in top_ks:
            values = accumulators[(slice_name, top_k)]
            gt_count = len(values)
            mean_best_iou = sum(values) / gt_count if gt_count else 0.0
            for threshold in iou_thresholds:
                recall = sum(value >= threshold for value in values) / gt_count if gt_count else 0.0
                rows.append(
                    {
                        "slice": slice_name,
                        "top_k": top_k,
                        "iou_threshold": threshold,
                        "gt_count": gt_count,
                        "recall": recall,
                        "mean_best_iou": mean_best_iou,
                        "class_aware": class_aware,
                    }
                )
    return rows


def annotation_rows(
    annotations: dict[str, Any],
    images: dict[int, dict[str, Any]],
    *,
    center_radius: float,
    small_area_ratio: float,
    large_area_ratio: float,
) -> list[dict[str, Any]]:
    rows = []
    for annotation in annotations.get("annotations", []):
        image = images[int(annotation["image_id"])]
        box = xywh_to_xyxy(annotation["bbox"])
        rows.append(
            {
                "image_id": int(annotation["image_id"]),
                "category_id": int(annotation["category_id"]),
                "box": box,
                "slices": slice_flags(
                    annotation["bbox"],
                    image,
                    center_radius=center_radius,
                    small_area_ratio=small_area_ratio,
                    large_area_ratio=large_area_ratio,
                ),
            }
        )
    return rows


def predictions_grouped_by_image(predictions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        rows.setdefault(image_id, []).append(
            {
                "category_id": int(prediction["category_id"]),
                "score": float(prediction.get("score", 0.0)),
                "box": xywh_to_xyxy(prediction["bbox"]),
            }
        )
    for image_rows in rows.values():
        image_rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def slice_flags(
    bbox: list[float],
    image: dict[str, Any],
    *,
    center_radius: float = 0.25,
    small_area_ratio: float = 0.05,
    large_area_ratio: float = 0.25,
) -> dict[str, bool]:
    image_width = max(1.0, float(image.get("width", 1.0) or 1.0))
    image_height = max(1.0, float(image.get("height", 1.0) or 1.0))
    x, y, width, height = [float(value) for value in bbox]
    cx = (x + 0.5 * width) / image_width
    cy = (y + 0.5 * height) / image_height
    center_distance = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    area_ratio = max(0.0, width) * max(0.0, height) / (image_width * image_height)
    return {
        "all": True,
        "offcenter": center_distance >= center_radius,
        "center": center_distance < center_radius,
        "small": area_ratio < small_area_ratio,
        "medium": small_area_ratio <= area_ratio < large_area_ratio,
        "large": area_ratio >= large_area_ratio,
    }


if __name__ == "__main__":
    main()
