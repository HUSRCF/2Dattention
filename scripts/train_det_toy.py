"""Train tiny DETR variants on a synthetic square-detection task."""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
import time
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import get_best_device  # noqa: E402
from attention2d.detection import DetectionCriterion, TinyAnchorRegionDETR  # noqa: E402
from attention2d.detection.matcher import box_cxcywh_to_xyxy, box_iou  # noqa: E402


MODEL_CONFIGS = {
    "learned": ("anchor", "learned", "none", "none", 0.1, "none"),
    "anchor": ("anchor", "anchor", "none", "none", 0.1, "none"),
    "local_learned": ("local", "learned", "none", "none", 0.1, "none"),
    "local_anchor": ("local", "anchor", "none", "none", 0.1, "none"),
    "local_anchor_detached": ("local", "anchor_detached", "none", "none", 0.1, "none"),
    "local_anchor_residual_query": ("local", "anchor_residual", "none", "none", 0.01, "none"),
    "local_anchor_residual_detached_query": (
        "local",
        "anchor_residual_detached",
        "none",
        "none",
        0.01,
        "none",
    ),
    "anchor_learned": ("anchor", "learned", "none", "none", 0.1, "none"),
    "anchor_anchor": ("anchor", "anchor", "none", "none", 0.1, "none"),
    "anchor_anchor_detached": ("anchor", "anchor_detached", "none", "none", 0.1, "none"),
    "anchor_anchor_residual_query": ("anchor", "anchor_residual", "none", "none", 0.01, "none"),
    "local_learned_maskaux": ("local", "learned", "none", "side", 0.1, "none"),
    "local_anchor_maskaux": ("local", "anchor", "none", "side", 0.1, "none"),
    "local_anchor_detached_maskaux": ("local", "anchor_detached", "none", "side", 0.1, "none"),
    "local_learned_maskpooled_query": ("local", "learned", "mask_pool", "pooled", 0.1, "none"),
    "local_anchor_maskpooled_query": ("local", "anchor", "mask_pool", "pooled", 0.1, "none"),
    "local_learned_mask_biased_attn": ("local", "learned", "mask_bias", "biased", 0.1, "none"),
    "local_anchor_mask_biased_attn": ("local", "anchor", "mask_bias", "biased", 0.1, "none"),
    "local_learned_mask_biased_attn_gate001": ("local", "learned", "mask_bias", "biased", 0.01, "none"),
    "local_anchor_mask_biased_attn_gate001": ("local", "anchor", "mask_bias", "biased", 0.01, "none"),
    "local_learned_mask_biased_attn_warmup": ("local", "learned", "mask_bias", "biased", 0.1, "linear"),
    "local_anchor_mask_biased_attn_warmup": ("local", "anchor", "mask_bias", "biased", 0.1, "linear"),
    "local_mask_proposal_query": ("local", "mask_proposal", "none", "proposal", 0.1, "none"),
    "anchor_mask_proposal_query": ("anchor", "mask_proposal", "none", "proposal", 0.1, "none"),
    "local_mask_proposal_nms_query": ("local", "mask_proposal_nms", "none", "proposal", 0.1, "none"),
    "anchor_mask_proposal_nms_query": ("anchor", "mask_proposal_nms", "none", "proposal", 0.1, "none"),
    "local_mask_proposal_nms_query_decode2": (
        "local",
        "mask_proposal_nms",
        "proposal_decode2",
        "proposal",
        0.1,
        "none",
    ),
    "local_mask_proposal_nms_query_reinject": (
        "local",
        "mask_proposal_nms",
        "proposal_reinject",
        "proposal",
        0.1,
        "none",
    ),
    "local_mask_proposal_nms_query_persistent": (
        "local",
        "mask_proposal_nms",
        "proposal_persistent",
        "proposal",
        0.1,
        "none",
    ),
    "local_mask_proposal_nms_query_persistent_gated": (
        "local",
        "mask_proposal_nms",
        "proposal_persistent",
        "proposal",
        0.01,
        "none",
    ),
    "local_mask_proposal_nms_query_late_persistent": (
        "local",
        "mask_proposal_nms",
        "proposal_late_persistent",
        "proposal",
        0.1,
        "none",
    ),
    "local_mask_proposal_nms_query_persistent_stopgrad": (
        "local",
        "mask_proposal_nms",
        "proposal_persistent_stopgrad",
        "proposal",
        0.1,
        "none",
    ),
    "local_learned_querymask": ("local", "learned", "query_mask", "query", 0.1, "none"),
    "local_anchor_residual_query_querymask": ("local", "anchor_residual", "query_mask", "query", 0.01, "none"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=["learned", "anchor"],
        help=(
            "Model variants. Legacy aliases: learned=anchor_learned, "
            "anchor=anchor_anchor."
        ),
    )
    parser.add_argument("--reference-model", choices=tuple(MODEL_CONFIGS), default="learned")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument(
        "--toy-mode",
        choices=("single", "multi", "distractor", "multi_distractor"),
        default="single",
    )
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--mask-aux-weight", type=float, default=0.5)
    parser.add_argument("--mask-dice-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/det_toy_compare.csv"))
    return parser.parse_args()


def mask_gate_scale(schedule: str, step: int, total_steps: int) -> float:
    if schedule == "none":
        return 1.0
    if schedule == "linear":
        return min(1.0, max(0.0, step / max(total_steps, 1)))
    raise ValueError(f"unknown mask gate schedule: {schedule}")


def main() -> None:
    args = parse_args()
    if args.max_objects > args.num_queries:
        raise ValueError("--max-objects must be <= --num-queries for the current brute-force matcher")
    device = get_best_device()
    rows = []
    print("device:", device)
    print(
        "model,run_seed,step,loss,eval_iou,eval_recall50,eval_ap50,"
        "eval_mask_iou,eval_mask_dice,best_iou,best_step,images_per_sec"
    )
    for seed_idx in range(args.seeds):
        run_seed = args.seed + seed_idx
        for model_name in args.models:
            feature_mode, query_init, query_refine, mask_aux_mode, gate_init, gate_schedule = MODEL_CONFIGS[
                model_name
            ]
            torch.manual_seed(run_seed)
            random.seed(run_seed)
            model = TinyAnchorRegionDETR(
                embed_dim=args.embed_dim,
                num_classes=1,
                num_queries=args.num_queries,
                feature_mode=feature_mode,
                query_init=query_init,
                query_refine=query_refine,
                query_mask_gate_init=gate_init,
            ).to(device)
            mask_head = DenseMaskAuxHead(args.embed_dim).to(device) if mask_aux_mode == "side" else None
            criterion = DetectionCriterion(num_classes=1).to(device)
            parameters = list(model.parameters())
            if mask_head is not None:
                parameters.extend(mask_head.parameters())
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-3)
            start = time.perf_counter()
            last_loss = 0.0
            best_iou = -1.0
            best_step = 0
            for step in range(1, args.steps + 1):
                random.seed(10_000_000 + run_seed * 100_000 + step)
                images, targets = sample_square_detection_batch(
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    device=device,
                    toy_mode=args.toy_mode,
                    max_objects=args.max_objects,
                    torch_seed=30_000_000 + run_seed * 100_000 + step,
                )
                model.set_query_mask_gate_scale(mask_gate_scale(gate_schedule, step, args.steps))
                outputs = model(images)
                losses = criterion(outputs, targets)
                if mask_aux_mode == "query":
                    mask_losses = query_mask_aux_loss(
                        outputs=outputs,
                        targets=targets,
                        criterion=criterion,
                        dice_weight=args.mask_dice_weight,
                    )
                    losses["loss"] = losses["loss"] + args.mask_aux_weight * mask_losses["loss_mask_aux"]
                elif mask_aux_mode != "none":
                    mask_losses = dense_mask_aux_loss(
                        outputs=outputs,
                        targets=targets,
                        mask_head=mask_head,
                        dice_weight=args.mask_dice_weight,
                    )
                    losses["loss"] = losses["loss"] + args.mask_aux_weight * mask_losses["loss_mask_aux"]
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
                last_loss = float(losses["loss"].item())
                if step % args.eval_every == 0 or step == args.steps:
                    metrics = evaluate_toy(
                        model,
                        args.batch_size,
                        args.image_size,
                        device,
                        toy_mode=args.toy_mode,
                        max_objects=args.max_objects,
                        mask_head=mask_head,
                        criterion=criterion,
                        seed=20_000_000 + run_seed,
                    )
                    if metrics["iou"] > best_iou:
                        best_iou = metrics["iou"]
                        best_step = step
                    speed = (step * args.batch_size) / max(time.perf_counter() - start, 1e-9)
                    row = {
                        "model": model_name,
                        "run_seed": run_seed,
                        "step": step,
                        "loss": last_loss,
                        "eval_iou": metrics["iou"],
                        "eval_recall50": metrics["recall50"],
                        "eval_ap50": metrics["ap50"],
                        "eval_mask_iou": metrics["mask_iou"],
                        "eval_mask_dice": metrics["mask_dice"],
                        "eval_small_iou": metrics["small_iou"],
                        "eval_medium_iou": metrics["medium_iou"],
                        "eval_large_iou": metrics["large_iou"],
                        "eval_center_iou": metrics["center_iou"],
                        "eval_offcenter_iou": metrics["offcenter_iou"],
                        "eval_small_recall50": metrics["small_recall50"],
                        "eval_medium_recall50": metrics["medium_recall50"],
                        "eval_large_recall50": metrics["large_recall50"],
                        "eval_center_recall50": metrics["center_recall50"],
                        "eval_offcenter_recall50": metrics["offcenter_recall50"],
                        "best_iou": best_iou,
                        "best_step": best_step,
                        "images_per_sec": speed,
                    }
                    rows.append(row)
                    print(
                        f"{model_name},{run_seed},{step},{last_loss:.4f},"
                        f"{metrics['iou']:.3f},{metrics['recall50']:.3f},{metrics['ap50']:.3f},"
                        f"{metrics['mask_iou']:.3f},{metrics['mask_dice']:.3f},"
                        f"{best_iou:.3f},{best_step},{speed:.2f}"
                    )
    write_rows(args.out, rows)
    print("saved_csv:", args.out)
    print_summary(rows)
    print_paired_summary(rows, reference_model=args.reference_model)


def sample_square_detection_batch(
    batch_size: int,
    image_size: int,
    device: torch.device,
    toy_mode: str = "single",
    max_objects: int = 3,
    torch_seed: int | None = None,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    targets = []
    for idx in range(batch_size):
        target_count, distractor_count = object_counts(toy_mode, max_objects)
        occupied_boxes: list[list[float]] = []
        boxes = []
        for _ in range(distractor_count):
            occupied_boxes.append(
                draw_square(
                    images[idx],
                    image_size,
                    device,
                    color=(0.85, 0.15, 0.15),
                    existing_boxes=occupied_boxes,
                )
            )
        for _ in range(target_count):
            box = draw_square(
                images[idx],
                image_size,
                device,
                color=(0.1, 0.8, 0.2),
                existing_boxes=occupied_boxes,
            )
            boxes.append(box)
            occupied_boxes.append(box)
        box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        targets.append(
            {
                "labels": torch.zeros(target_count, dtype=torch.long, device=device),
                "boxes": box_tensor,
            }
        )
    generator = None
    if torch_seed is not None:
        generator = torch.Generator(device=device).manual_seed(torch_seed)
    noise = torch.randn(images.shape, device=device, generator=generator) * 0.02
    return (images + noise).clamp(0, 1), targets


def object_counts(toy_mode: str, max_objects: int) -> tuple[int, int]:
    max_objects = max(1, max_objects)
    if toy_mode == "single":
        return 1, 0
    if toy_mode == "multi":
        return random.randint(1, max_objects), 0
    if toy_mode == "distractor":
        return 1, random.randint(1, max_objects)
    if toy_mode == "multi_distractor":
        return random.randint(1, max_objects), random.randint(1, max_objects)
    raise ValueError(f"unknown toy_mode: {toy_mode}")


def draw_square(
    image: Tensor,
    image_size: int,
    device: torch.device,
    color: tuple[float, float, float],
    existing_boxes: list[list[float]] | None = None,
    max_iou: float = 0.02,
) -> list[float]:
    existing_boxes = existing_boxes or []
    box = sample_non_overlapping_square(image_size, existing_boxes, max_iou=max_iou)
    cx, cy, width, height = box
    side = int(round(width * image_size))
    left = int(round((cx - width / 2) * image_size))
    top = int(round((cy - height / 2) * image_size))
    color_tensor = torch.tensor(color, device=device).view(3, 1, 1)
    image[:, top : top + side, left : left + side] = color_tensor
    return box


def sample_non_overlapping_square(
    image_size: int,
    existing_boxes: list[list[float]],
    max_iou: float,
    attempts: int = 100,
) -> list[float]:
    best_box = None
    best_overlap = float("inf")
    for _ in range(attempts):
        box = sample_square_box(image_size)
        overlap = max((cxcywh_iou(box, existing) for existing in existing_boxes), default=0.0)
        if overlap <= max_iou:
            return box
        if overlap < best_overlap:
            best_box = box
            best_overlap = overlap
    if best_box is None:
        return sample_square_box(image_size)
    return best_box


def sample_square_box(image_size: int) -> list[float]:
    side = random.randint(image_size // 8, image_size // 3)
    left = random.randint(0, image_size - side - 1)
    top = random.randint(0, image_size - side - 1)
    cx = (left + side / 2) / image_size
    cy = (top + side / 2) / image_size
    return [cx, cy, side / image_size, side / image_size]


def cxcywh_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = cxcywh_to_xyxy_list(box_a)
    bx1, by1, bx2, by2 = cxcywh_to_xyxy_list(box_b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def cxcywh_to_xyxy_list(box: list[float]) -> tuple[float, float, float, float]:
    cx, cy, width, height = box
    return (
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    )


@torch.no_grad()
def evaluate_toy(
    model: TinyAnchorRegionDETR,
    batch_size: int,
    image_size: int,
    device: torch.device,
    toy_mode: str = "single",
    max_objects: int = 3,
    mask_head: nn.Module | None = None,
    criterion: DetectionCriterion | None = None,
    batches: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    model.eval()
    if mask_head is not None:
        mask_head.eval()
    ious = []
    recalls = []
    aps = []
    mask_ious = []
    mask_dices = []
    strata: dict[str, list[Tensor]] = {
        "small": [],
        "medium": [],
        "large": [],
        "center": [],
        "offcenter": [],
    }
    for batch_idx in range(batches):
        random.seed(seed + batch_idx)
        images, targets = sample_square_detection_batch(
            batch_size,
            image_size,
            device,
            toy_mode=toy_mode,
            max_objects=max_objects,
            torch_seed=seed + batch_idx,
        )
        outputs = model(images)
        if "query_mask_logits_per_query" in outputs:
            if criterion is None:
                raise ValueError("criterion is required for query-mask metrics")
            mask_metrics = query_mask_aux_metrics(outputs, targets, criterion)
            mask_ious.append(mask_metrics["mask_iou"].cpu())
            mask_dices.append(mask_metrics["mask_dice"].cpu())
        elif mask_head is not None or "query_mask_logits" in outputs:
            mask_metrics = dense_mask_aux_metrics(outputs, targets, mask_head)
            mask_ious.append(mask_metrics["mask_iou"].cpu())
            mask_dices.append(mask_metrics["mask_dice"].cpu())
        for sample_idx, target in enumerate(targets):
            matched_iou = match_targets_by_iou(
                outputs["pred_boxes"][sample_idx],
                target["boxes"],
            )
            ious.append(matched_iou.cpu())
            recalls.append((matched_iou >= 0.5).float().cpu())
            update_stratified_iou_lists(strata, matched_iou.cpu(), target["boxes"].cpu())
            aps.append(
                ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                ).cpu()
            )
    model.train()
    if mask_head is not None:
        mask_head.train()
    all_ious = torch.cat(ious)
    all_recalls = torch.cat(recalls)
    stratified_metrics = summarize_stratified_iou(strata)
    if mask_ious:
        mean_mask_iou = float(torch.stack(mask_ious).mean().item())
        mean_mask_dice = float(torch.stack(mask_dices).mean().item())
    else:
        mean_mask_iou = 0.0
        mean_mask_dice = 0.0
    return {
        "iou": float(all_ious.mean().item()),
        "recall50": float(all_recalls.mean().item()),
        "ap50": float(torch.stack(aps).mean().item()),
        "mask_iou": mean_mask_iou,
        "mask_dice": mean_mask_dice,
        **stratified_metrics,
    }


def update_stratified_iou_lists(strata: dict[str, list[Tensor]], ious: Tensor, boxes: Tensor) -> None:
    areas = boxes[:, 2] * boxes[:, 3]
    center_distance = ((boxes[:, 0] - 0.5).square() + (boxes[:, 1] - 0.5).square()).sqrt()
    masks = {
        "small": areas < 0.04,
        "medium": (areas >= 0.04) & (areas < 0.075),
        "large": areas >= 0.075,
        "center": center_distance < 0.25,
        "offcenter": center_distance >= 0.25,
    }
    for name, mask in masks.items():
        if bool(mask.any()):
            strata[name].append(ious[mask])


def summarize_stratified_iou(strata: dict[str, list[Tensor]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, values in strata.items():
        if values:
            joined = torch.cat(values)
            metrics[f"{name}_iou"] = float(joined.mean().item())
            metrics[f"{name}_recall50"] = float((joined >= 0.5).float().mean().item())
        else:
            metrics[f"{name}_iou"] = 0.0
            metrics[f"{name}_recall50"] = 0.0
    return metrics


def match_targets_by_iou(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    target_count = target_boxes.shape[0]
    if target_count == 0:
        return target_boxes.new_zeros((0,))
    query_count = pred_boxes.shape[0]
    if target_count > query_count:
        raise ValueError("target_count must be <= query_count for match_targets_by_iou")
    matched_count = target_count
    iou_matrix = box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(target_boxes),
    )
    target_indices = torch.arange(matched_count, device=target_boxes.device)
    best_values = None
    best_score = None
    for query_indices_tuple in itertools.permutations(range(query_count), matched_count):
        query_indices = torch.tensor(query_indices_tuple, device=target_boxes.device)
        values = iou_matrix[query_indices, target_indices]
        score = values.sum()
        if best_score is None or bool(score > best_score):
            best_score = score
            best_values = values
    if best_values is None:
        return target_boxes.new_zeros((target_count,))
    return best_values


def ap50_for_image(pred_logits: Tensor, pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    target_count = target_boxes.shape[0]
    if target_count == 0:
        return pred_logits.new_tensor(0.0)
    scores = pred_logits.softmax(dim=-1)[:, 0]
    order = scores.argsort(descending=True)
    iou_matrix = box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(target_boxes),
    )
    matched_targets: set[int] = set()
    true_positives = []
    false_positives = []
    for query_idx_tensor in order:
        query_idx = int(query_idx_tensor.item())
        ious = iou_matrix[query_idx]
        best_iou, best_target_tensor = ious.max(dim=0)
        best_target = int(best_target_tensor.item())
        if float(best_iou.item()) >= 0.5 and best_target not in matched_targets:
            matched_targets.add(best_target)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)
    tp = torch.tensor(true_positives, device=pred_logits.device, dtype=pred_logits.dtype).cumsum(dim=0)
    fp = torch.tensor(false_positives, device=pred_logits.device, dtype=pred_logits.dtype).cumsum(dim=0)
    recall = tp / max(target_count, 1)
    precision = tp / (tp + fp).clamp_min(1e-8)
    return precision_recall_ap(recall, precision)


def precision_recall_ap(recall: Tensor, precision: Tensor) -> Tensor:
    zero = recall.new_tensor([0.0])
    one = recall.new_tensor([1.0])
    mrec = torch.cat((zero, recall, one), dim=0)
    mpre = torch.cat((zero, precision, zero), dim=0)
    for idx in range(mpre.numel() - 2, -1, -1):
        mpre[idx] = torch.maximum(mpre[idx], mpre[idx + 1])
    changes = mrec[1:] - mrec[:-1]
    return (changes * mpre[1:]).sum()


class DenseMaskAuxHead(nn.Module):
    """Predict a dense target-box mask from detector spatial features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        return self.proj(features).squeeze(1)


def dense_mask_aux_loss(
    outputs: dict[str, Tensor | list[Tensor]],
    targets: list[dict[str, Tensor]],
    mask_head: DenseMaskAuxHead | None,
    dice_weight: float,
) -> dict[str, Tensor]:
    logits = dense_mask_logits(outputs, mask_head)
    target_masks = dense_mask_targets_from_boxes(
        targets=targets,
        height=logits.shape[-2],
        width=logits.shape[-1],
        device=logits.device,
    )
    loss_bce = F.binary_cross_entropy_with_logits(logits, target_masks)
    probs = logits.sigmoid()
    loss_dice = 1.0 - soft_dice_score(probs, target_masks)
    return {
        "loss_mask_aux": loss_bce + dice_weight * loss_dice,
        "loss_mask_bce": loss_bce,
        "loss_mask_dice": loss_dice,
    }


def query_mask_aux_loss(
    outputs: dict[str, Tensor | list[Tensor]],
    targets: list[dict[str, Tensor]],
    criterion: DetectionCriterion,
    dice_weight: float,
) -> dict[str, Tensor]:
    """Supervise each Hungarian-matched query with its matched bbox mask."""

    logits = outputs["query_mask_logits_per_query"]
    matched_logits, matched_targets = matched_query_mask_tensors(outputs, targets, criterion)
    if matched_logits.numel() == 0:
        zero = logits.sum() * 0.0
        return {"loss_mask_aux": zero, "loss_mask_bce": zero, "loss_mask_dice": zero}
    loss_bce = F.binary_cross_entropy_with_logits(matched_logits, matched_targets)
    loss_dice = 1.0 - soft_dice_score(matched_logits.sigmoid(), matched_targets)
    return {
        "loss_mask_aux": loss_bce + dice_weight * loss_dice,
        "loss_mask_bce": loss_bce,
        "loss_mask_dice": loss_dice,
    }


@torch.no_grad()
def dense_mask_aux_metrics(
    outputs: dict[str, Tensor | list[Tensor]],
    targets: list[dict[str, Tensor]],
    mask_head: DenseMaskAuxHead | None,
    threshold: float = 0.5,
) -> dict[str, Tensor]:
    logits = dense_mask_logits(outputs, mask_head)
    target_masks = dense_mask_targets_from_boxes(
        targets=targets,
        height=logits.shape[-2],
        width=logits.shape[-1],
        device=logits.device,
    )
    probs = logits.sigmoid()
    pred = probs >= threshold
    target_bool = target_masks >= 0.5
    intersection = (pred & target_bool).float().flatten(1).sum(dim=1)
    union = (pred | target_bool).float().flatten(1).sum(dim=1).clamp_min(1e-8)
    pred_sum = pred.float().flatten(1).sum(dim=1)
    target_sum = target_bool.float().flatten(1).sum(dim=1)
    dice = (2 * intersection / (pred_sum + target_sum).clamp_min(1e-8)).mean()
    return {
        "mask_iou": (intersection / union).mean(),
        "mask_dice": dice,
    }


@torch.no_grad()
def query_mask_aux_metrics(
    outputs: dict[str, Tensor | list[Tensor]],
    targets: list[dict[str, Tensor]],
    criterion: DetectionCriterion,
    threshold: float = 0.5,
) -> dict[str, Tensor]:
    """Evaluate matched query-specific masks against matched bbox masks."""

    logits = outputs["query_mask_logits_per_query"]
    matched_logits, matched_targets = matched_query_mask_tensors(outputs, targets, criterion)
    if matched_logits.numel() == 0:
        zero = logits.new_tensor(0.0)
        return {"mask_iou": zero, "mask_dice": zero}
    pred = matched_logits.sigmoid() >= threshold
    target_bool = matched_targets >= 0.5
    intersection = (pred & target_bool).float().flatten(1).sum(dim=1)
    union = (pred | target_bool).float().flatten(1).sum(dim=1).clamp_min(1e-8)
    pred_sum = pred.float().flatten(1).sum(dim=1)
    target_sum = target_bool.float().flatten(1).sum(dim=1)
    dice = (2 * intersection / (pred_sum + target_sum).clamp_min(1e-8)).mean()
    return {
        "mask_iou": (intersection / union).mean(),
        "mask_dice": dice,
    }


def matched_query_mask_tensors(
    outputs: dict[str, Tensor | list[Tensor]],
    targets: list[dict[str, Tensor]],
    criterion: DetectionCriterion,
) -> tuple[Tensor, Tensor]:
    """Return matched per-query mask logits and matched bbox mask targets."""

    logits = outputs["query_mask_logits_per_query"]
    matches = criterion.matcher(outputs, targets)
    matched_logits = []
    matched_masks = []
    height, width = logits.shape[-2:]
    for batch_idx, (src_idx, target_idx) in enumerate(matches):
        if src_idx.numel() == 0:
            continue
        matched_logits.append(logits[batch_idx, src_idx])
        matched_target = [
            {"boxes": targets[batch_idx]["boxes"][target_id].unsqueeze(0)}
            for target_id in target_idx
        ]
        matched_masks.append(
            dense_mask_targets_from_boxes(
                targets=matched_target,
                height=height,
                width=width,
                device=logits.device,
            )
        )
    if not matched_logits:
        empty_logits = logits.new_zeros((0, height, width))
        return empty_logits, empty_logits
    return torch.cat(matched_logits, dim=0), torch.cat(matched_masks, dim=0)


def dense_mask_logits(
    outputs: dict[str, Tensor | list[Tensor]],
    mask_head: DenseMaskAuxHead | None,
) -> Tensor:
    if "query_mask_logits" in outputs:
        return outputs["query_mask_logits"]
    if mask_head is None:
        raise ValueError("mask_head is required when outputs do not include query_mask_logits")
    return mask_head(outputs["spatial_features"])


def dense_mask_targets_from_boxes(
    targets: list[dict[str, Tensor]],
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    masks = torch.zeros(len(targets), height, width, device=device)
    for batch_idx, target in enumerate(targets):
        for box in target["boxes"].to(device):
            cx, cy, box_w, box_h = box.tolist()
            x1 = max(0, int(torch.floor(torch.tensor((cx - box_w / 2) * width)).item()))
            y1 = max(0, int(torch.floor(torch.tensor((cy - box_h / 2) * height)).item()))
            x2 = min(width, int(torch.ceil(torch.tensor((cx + box_w / 2) * width)).item()))
            y2 = min(height, int(torch.ceil(torch.tensor((cy + box_h / 2) * height)).item()))
            if x2 > x1 and y2 > y1:
                masks[batch_idx, y1:y2, x1:x2] = 1.0
    return masks


def soft_dice_score(probs: Tensor, targets: Tensor) -> Tensor:
    probs_flat = probs.flatten(1)
    targets_flat = targets.flatten(1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    return ((2 * intersection + 1.0) / (denominator + 1.0)).mean()


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "run_seed",
                "step",
                "loss",
                "eval_iou",
                "eval_recall50",
                "eval_ap50",
                "eval_mask_iou",
                "eval_mask_dice",
                "eval_small_iou",
                "eval_medium_iou",
                "eval_large_iou",
                "eval_center_iou",
                "eval_offcenter_iou",
                "eval_small_recall50",
                "eval_medium_recall50",
                "eval_large_recall50",
                "eval_center_recall50",
                "eval_offcenter_recall50",
                "best_iou",
                "best_step",
                "images_per_sec",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    final_rows = final_rows_by_model_seed(rows)
    print(
        "summary_model,final_iou_mean,best_iou_mean,recall50_mean,ap50_mean,"
        "mask_iou_mean,mask_dice_mean,small_iou_mean,medium_iou_mean,large_iou_mean,"
        "center_iou_mean,offcenter_iou_mean,images_per_sec_mean"
    )
    for model in sorted({str(row["model"]) for row in final_rows}):
        model_rows = [row for row in final_rows if row["model"] == model]
        print(
            f"{model},"
            f"{mean([float(row['eval_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['best_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_recall50']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_mask_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_mask_dice']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_small_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_medium_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_large_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_center_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_offcenter_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['images_per_sec']) for row in model_rows]):.2f}"
        )


def print_paired_summary(rows: list[dict[str, float | int | str]], reference_model: str) -> None:
    final_rows = final_rows_by_model_seed(rows)
    if reference_model not in {str(row["model"]) for row in final_rows}:
        return
    print(f"paired_det_toy_vs,{reference_model}")
    print(
        "paired_model,final_iou_delta_mean,final_iou_wins,best_iou_delta_mean,"
        "best_iou_wins,ap50_delta_mean,ap50_wins,mask_iou_delta_mean,mask_iou_wins"
    )
    run_seeds = sorted({int(row["run_seed"]) for row in final_rows})
    for model in sorted({str(row["model"]) for row in final_rows}):
        if model == reference_model:
            continue
        final_deltas = []
        best_deltas = []
        ap_deltas = []
        mask_deltas = []
        for run_seed in run_seeds:
            ref = find_row(final_rows, model=reference_model, run_seed=run_seed)
            cur = find_row(final_rows, model=model, run_seed=run_seed)
            if ref is None or cur is None:
                continue
            final_deltas.append(float(cur["eval_iou"]) - float(ref["eval_iou"]))
            best_deltas.append(float(cur["best_iou"]) - float(ref["best_iou"]))
            ap_deltas.append(float(cur["eval_ap50"]) - float(ref["eval_ap50"]))
            mask_deltas.append(float(cur["eval_mask_iou"]) - float(ref["eval_mask_iou"]))
        if not final_deltas:
            continue
        print(
            f"{model},"
            f"{mean(final_deltas):.3f},{wins_higher(final_deltas)},"
            f"{mean(best_deltas):.3f},{wins_higher(best_deltas)},"
            f"{mean(ap_deltas):.3f},{wins_higher(ap_deltas)},"
            f"{mean(mask_deltas):.3f},{wins_higher(mask_deltas)}"
        )


def final_rows_by_model_seed(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    latest: dict[tuple[str, int], dict[str, float | int | str]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["run_seed"]))
        if key not in latest or int(row["step"]) > int(latest[key]["step"]):
            latest[key] = row
    return list(latest.values())


def find_row(
    rows: list[dict[str, float | int | str]],
    model: str,
    run_seed: int,
) -> dict[str, float | int | str] | None:
    for row in rows:
        if row["model"] == model and int(row["run_seed"]) == run_seed:
            return row
    return None


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def wins_higher(values: list[float]) -> str:
    return f"{sum(value > 0 for value in values)}/{len(values)}"


if __name__ == "__main__":
    main()
