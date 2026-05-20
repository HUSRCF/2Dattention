"""Diagnostic-only rescoring of COCO predictions by nearest-GT IoU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torchvision.ops import box_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--class-aware",
        action="store_true",
        help="Only compare predictions to GT boxes with the same category id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = rescore_predictions_by_oracle_iou(
        annotation_json=args.annotations,
        prediction_json=args.predictions,
        class_aware=args.class_aware,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records), encoding="utf-8")
    print(f"saved_oracle_rescored_predictions: {args.out}")
    print(f"prediction_count: {len(records)}")


def rescore_predictions_by_oracle_iou(
    annotation_json: Path,
    prediction_json: Path,
    class_aware: bool = False,
) -> list[dict[str, Any]]:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_json.read_text(encoding="utf-8"))
    gt_by_image = gt_boxes_by_image(data)
    rescored = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        gt_rows = gt_by_image.get(image_id, [])
        if class_aware:
            gt_rows = [row for row in gt_rows if row["category_id"] == category_id]
        gt_boxes = torch.tensor([row["box"] for row in gt_rows], dtype=torch.float32).reshape(-1, 4)
        pred_box = torch.tensor([xywh_to_xyxy(prediction["bbox"])], dtype=torch.float32)
        score = 0.0
        if gt_boxes.numel():
            score = float(box_iou(pred_box, gt_boxes).max().item())
        row = dict(prediction)
        row["score"] = score
        rescored.append(row)
    return rescored


def gt_boxes_by_image(data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for annotation in data.get("annotations", []):
        image_id = int(annotation["image_id"])
        rows.setdefault(image_id, []).append(
            {
                "category_id": int(annotation["category_id"]),
                "box": xywh_to_xyxy(annotation["bbox"]),
            }
        )
    return rows


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = [float(value) for value in box]
    return [x, y, x + max(0.0, width), y + max(0.0, height)]


if __name__ == "__main__":
    main()
