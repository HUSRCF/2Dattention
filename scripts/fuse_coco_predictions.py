"""Fuse COCO detection predictions with score filtering and greedy NMS."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections-per-image", type=int, default=100)
    parser.add_argument("--class-agnostic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions: list[dict[str, Any]] = []
    for path in args.predictions:
        predictions.extend(json.loads(path.read_text(encoding="utf-8")))
    fused = fuse_coco_predictions(
        predictions,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_detections_per_image=args.max_detections_per_image,
        class_agnostic=args.class_agnostic,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fused, indent=2) + "\n", encoding="utf-8")
    print(f"saved_fused_predictions: {args.out}")
    print(f"input_predictions: {len(predictions)}")
    print(f"fused_predictions: {len(fused)}")


def fuse_coco_predictions(
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    max_detections_per_image: int = 100,
    class_agnostic: bool = False,
) -> list[dict[str, Any]]:
    valid_predictions = [normalize_prediction(prediction) for prediction in predictions]
    valid_predictions = [
        prediction
        for prediction in valid_predictions
        if prediction is not None and prediction["score"] >= score_threshold
    ]

    by_group: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in valid_predictions:
        group_category = 0 if class_agnostic else int(prediction["category_id"])
        by_group[(int(prediction["image_id"]), group_category)].append(prediction)

    kept_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (image_id, _), group_predictions in by_group.items():
        for prediction in greedy_nms(group_predictions, iou_threshold=iou_threshold):
            kept_by_image[image_id].append(prediction)

    fused: list[dict[str, Any]] = []
    for image_id in sorted(kept_by_image):
        image_predictions = sorted(kept_by_image[image_id], key=lambda item: item["score"], reverse=True)
        fused.extend(image_predictions[:max_detections_per_image])
    return fused


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any] | None:
    bbox = [float(value) for value in prediction.get("bbox", [])]
    if len(bbox) != 4:
        return None
    x, y, width, height = bbox
    score = float(prediction.get("score", 0.0))
    if width <= 0 or height <= 0 or score < 0:
        return None
    return {
        "image_id": int(prediction["image_id"]),
        "category_id": int(prediction["category_id"]),
        "bbox": [x, y, width, height],
        "score": score,
    }


def greedy_nms(predictions: list[dict[str, Any]], *, iou_threshold: float) -> list[dict[str, Any]]:
    remaining = sorted(predictions, key=lambda item: item["score"], reverse=True)
    kept: list[dict[str, Any]] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            prediction
            for prediction in remaining
            if xywh_iou(current["bbox"], prediction["bbox"]) <= iou_threshold
        ]
    return kept


def xywh_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = aw * ah + bw * bh - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


if __name__ == "__main__":
    main()
