"""Evaluate torchvision RPN proposal recall before ROI classification."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_torchvision_coco_detector import (  # noqa: E402
    CocoDetectionLite,
    build_model,
    collate_detection,
    load_checkpoint,
    max_category_id,
    select_device,
)


SLICES = ("all", "offcenter", "center", "small", "medium", "large")
FIELDNAMES = ("slice", "top_k", "iou_threshold", "gt_count", "recall", "mean_best_iou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-eval-images", type=int, default=0)
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 50, 100, 150])
    parser.add_argument("--iou-thresholds", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--max-proposals", type=int, default=300)
    parser.add_argument("--center-radius", type=float, default=0.25)
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    parser.add_argument("--large-area-ratio", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--weights",
        choices=("none", "coco"),
        default="none",
        help="Use COCO-pretrained detector weights if available; may download if not cached.",
    )
    parser.add_argument("--weights-file", type=Path, default=None)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional project detector checkpoint loaded after building the eval-class predictor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dataset: Dataset[Any] = CocoDetectionLite(args.eval_json, args.image_root, image_size=args.image_size)
    if args.max_eval_images > 0:
        dataset = Subset(dataset, list(range(min(args.max_eval_images, len(dataset)))))
    num_classes = max_category_id(dataset) + 1
    model = build_model(
        num_classes=num_classes,
        image_size=args.image_size,
        weights=args.weights,
        weights_file=args.weights_file,
    ).to(device)
    if args.resume_checkpoint is not None:
        load_checkpoint(args.resume_checkpoint, model, optimizer=None)
    model.eval()
    set_rpn_top_n(model, max(args.max_proposals, max(args.top_k)))
    rows = evaluate_proposal_recall(
        model,
        dataset,
        device=device,
        top_ks=tuple(sorted(set(args.top_k))),
        iou_thresholds=tuple(args.iou_thresholds),
        center_radius=args.center_radius,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_proposal_recall: {args.out}")
    for row in rows:
        if row["slice"] == "all" and row["iou_threshold"] in {0.5, 0.75}:
            print(
                f"all@{row['top_k']} IoU>={row['iou_threshold']}: "
                f"recall={row['recall']:.4f} mean_best_iou={row['mean_best_iou']:.4f}"
            )


@torch.no_grad()
def evaluate_proposal_recall(
    model: torch.nn.Module,
    dataset: Dataset[Any],
    device: torch.device,
    top_ks: tuple[int, ...] = (10, 50, 100, 150),
    iou_thresholds: tuple[float, ...] = (0.5, 0.75),
    center_radius: float = 0.25,
    small_area_ratio: float = 0.05,
    large_area_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    accumulators = {
        (slice_name, top_k): {"best_ious": []}
        for slice_name in SLICES
        for top_k in top_ks
    }
    for images, targets in DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_detection):
        image = images[0].to(device)
        target = targets[0]
        gt_boxes = target["boxes"].cpu()
        if gt_boxes.numel() == 0:
            continue
        proposals = rpn_proposals(model, image)
        slice_masks = proposal_slice_masks(
            gt_boxes,
            resized_size=target["resized_size"],
            center_radius=center_radius,
            small_area_ratio=small_area_ratio,
            large_area_ratio=large_area_ratio,
        )
        for top_k in top_ks:
            best_ious = best_iou_per_gt(gt_boxes, proposals[:top_k].cpu())
            for slice_name, mask in slice_masks.items():
                selected = best_ious[mask]
                accumulators[(slice_name, top_k)]["best_ious"].extend(selected.tolist())
    rows: list[dict[str, Any]] = []
    for slice_name in SLICES:
        for top_k in top_ks:
            values = torch.tensor(accumulators[(slice_name, top_k)]["best_ious"], dtype=torch.float32)
            gt_count = int(values.numel())
            mean_best_iou = float(values.mean().item()) if gt_count else 0.0
            for threshold in iou_thresholds:
                recall = float((values >= threshold).float().mean().item()) if gt_count else 0.0
                rows.append(
                    {
                        "slice": slice_name,
                        "top_k": top_k,
                        "iou_threshold": threshold,
                        "gt_count": gt_count,
                        "recall": recall,
                        "mean_best_iou": mean_best_iou,
                    }
                )
    return rows


def rpn_proposals(model: torch.nn.Module, image: Tensor) -> Tensor:
    transformed, _ = model.transform([image], None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, Tensor):
        features = OrderedDict([("0", features)])
    proposals, _ = model.rpn(transformed, features, None)
    return proposals[0].detach().cpu()


def best_iou_per_gt(gt_boxes: Tensor, proposals: Tensor) -> Tensor:
    if gt_boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    if proposals.numel() == 0:
        return torch.zeros((len(gt_boxes),), dtype=torch.float32)
    return box_iou(gt_boxes, proposals).max(dim=1).values


def proposal_slice_masks(
    boxes: Tensor,
    resized_size: Tensor,
    center_radius: float,
    small_area_ratio: float,
    large_area_ratio: float,
) -> dict[str, Tensor]:
    height, width = [float(value) for value in resized_size.tolist()]
    width = max(1.0, width)
    height = max(1.0, height)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    box_w = (x2 - x1).clamp(min=0.0)
    box_h = (y2 - y1).clamp(min=0.0)
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    distance = torch.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    area_ratio = box_w * box_h / (width * height)
    return {
        "all": torch.ones((len(boxes),), dtype=torch.bool),
        "offcenter": distance >= center_radius,
        "center": distance < center_radius,
        "small": area_ratio < small_area_ratio,
        "medium": (area_ratio >= small_area_ratio) & (area_ratio < large_area_ratio),
        "large": area_ratio >= large_area_ratio,
    }


def set_rpn_top_n(model: torch.nn.Module, top_n: int) -> None:
    if hasattr(model.rpn, "_post_nms_top_n"):
        model.rpn._post_nms_top_n["testing"] = int(top_n)
    if hasattr(model.rpn, "_pre_nms_top_n"):
        model.rpn._pre_nms_top_n["testing"] = max(int(top_n), int(model.rpn._pre_nms_top_n.get("testing", top_n)))


if __name__ == "__main__":
    main()
