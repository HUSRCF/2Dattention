"""Evaluate deterministic anchor-grid oracle recall on a COCO detection split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_torchvision_proposal_recall import FIELDNAMES, SLICES, best_iou_per_gt, proposal_slice_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--strides", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--scales", nargs="+", type=float, default=[8, 12, 16, 24, 32, 48, 64, 96])
    parser.add_argument("--aspect-ratios", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--center-radius", type=float, default=0.25)
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    parser.add_argument("--large-area-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_by_image, image_sizes = load_scaled_boxes(args.eval_json, image_size=args.image_size)
    anchors = generate_anchor_grid(
        image_size=args.image_size,
        strides=tuple(args.strides),
        scales=tuple(args.scales),
        aspect_ratios=tuple(args.aspect_ratios),
    )
    rows = evaluate_anchor_grid_recall(
        gt_by_image=gt_by_image,
        image_sizes=image_sizes,
        anchors=anchors,
        iou_thresholds=tuple(args.iou_thresholds),
        center_radius=args.center_radius,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FIELDNAMES, "anchor_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_anchor_grid_recall: {args.out}")
    print(f"anchor_count: {len(anchors)}")
    for row in rows:
        if row["slice"] == "all" and row["iou_threshold"] in {0.5, 0.75}:
            print(
                f"all IoU>={row['iou_threshold']}: "
                f"recall={row['recall']:.4f} mean_best_iou={row['mean_best_iou']:.4f}"
            )


def load_scaled_boxes(annotation_json: Path, image_size: int) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    with annotation_json.open(encoding="utf-8") as handle:
        data = json.load(handle)
    image_sizes = {
        int(image["id"]): (float(image["height"]), float(image["width"]))
        for image in data.get("images", [])
    }
    boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in image_sizes}
    for annotation in data.get("annotations", []):
        image_id = int(annotation["image_id"])
        original_h, original_w = image_sizes[image_id]
        scale_x = image_size / max(1.0, original_w)
        scale_y = image_size / max(1.0, original_h)
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        boxes_by_image.setdefault(image_id, []).append(
            [
                x * scale_x,
                y * scale_y,
                (x + width) * scale_x,
                (y + height) * scale_y,
            ]
        )
    gt_by_image = {
        image_id: torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        for image_id, boxes in boxes_by_image.items()
    }
    resized_sizes = {
        image_id: torch.tensor([float(image_size), float(image_size)], dtype=torch.float32)
        for image_id in image_sizes
    }
    return gt_by_image, resized_sizes


def generate_anchor_grid(
    image_size: int,
    strides: tuple[int, ...],
    scales: tuple[float, ...],
    aspect_ratios: tuple[float, ...],
) -> Tensor:
    boxes: list[list[float]] = []
    for stride in strides:
        centers = torch.arange(stride / 2.0, image_size, stride, dtype=torch.float32)
        for cy in centers.tolist():
            for cx in centers.tolist():
                for scale in scales:
                    for ratio in aspect_ratios:
                        width = scale * math.sqrt(ratio)
                        height = scale / math.sqrt(ratio)
                        x1 = max(0.0, cx - width / 2.0)
                        y1 = max(0.0, cy - height / 2.0)
                        x2 = min(float(image_size), cx + width / 2.0)
                        y2 = min(float(image_size), cy + height / 2.0)
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def evaluate_anchor_grid_recall(
    gt_by_image: dict[int, Tensor],
    image_sizes: dict[int, Tensor],
    anchors: Tensor,
    iou_thresholds: tuple[float, ...],
    center_radius: float,
    small_area_ratio: float,
    large_area_ratio: float,
) -> list[dict[str, Any]]:
    accumulators = {slice_name: [] for slice_name in SLICES}
    for image_id, gt_boxes in gt_by_image.items():
        if gt_boxes.numel() == 0:
            continue
        best_ious = best_iou_per_gt(gt_boxes, anchors)
        slice_masks = proposal_slice_masks(
            gt_boxes,
            resized_size=image_sizes[image_id],
            center_radius=center_radius,
            small_area_ratio=small_area_ratio,
            large_area_ratio=large_area_ratio,
        )
        for slice_name, mask in slice_masks.items():
            accumulators[slice_name].extend(best_ious[mask].tolist())
    rows: list[dict[str, Any]] = []
    for slice_name in SLICES:
        values = torch.tensor(accumulators[slice_name], dtype=torch.float32)
        gt_count = int(values.numel())
        mean_best_iou = float(values.mean().item()) if gt_count else 0.0
        for threshold in iou_thresholds:
            rows.append(
                {
                    "slice": slice_name,
                    "top_k": "oracle_grid",
                    "iou_threshold": threshold,
                    "gt_count": gt_count,
                    "recall": float((values >= threshold).float().mean().item()) if gt_count else 0.0,
                    "mean_best_iou": mean_best_iou,
                    "anchor_count": int(len(anchors)),
                }
            )
    return rows


if __name__ == "__main__":
    main()
