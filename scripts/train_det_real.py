"""Train tiny DETR variants on real ILSVRC2013 DET validation boxes.

This is a mini-detector path, not a full RF-DETR reproduction. It reuses the
tiny detector controls from ``train_det_toy.py`` and replaces synthetic squares
with real images plus XML boxes. Metrics are lightweight smoke diagnostics:
target-matched IoU/recall and objectness AP50-lite.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from itertools import cycle, permutations
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from attention2d import get_best_device  # noqa: E402
from attention2d.detection import DetectionCriterion, TinyAnchorRegionDETR  # noqa: E402
from attention2d.detection.matcher import box_cxcywh_to_xyxy, box_iou  # noqa: E402
from train_det_toy import (  # noqa: E402
    MODEL_CONFIGS as BASE_MODEL_CONFIGS,
    dense_mask_aux_loss,
    dense_mask_aux_metrics,
    dense_mask_targets_from_boxes,
    mask_gate_scale,
    match_targets_by_iou,
    precision_recall_ap,
    query_mask_aux_loss,
    query_mask_aux_metrics,
    summarize_stratified_iou,
    update_stratified_iou_lists,
)


REAL_MODEL_CONFIGS = {
    **BASE_MODEL_CONFIGS,
    "local_learned_calib": ("local", "learned", "none", "none", 0.1, "none"),
    "local_anchor_calib": ("local", "anchor", "none", "none", 0.1, "none"),
    "local_anchor_residual_query_calib": ("local", "anchor_residual", "none", "none", 0.01, "none"),
    "local_learned_matchqual": ("local", "learned", "none", "none", 0.1, "none"),
    "local_anchor_matchqual": ("local", "anchor", "none", "none", 0.1, "none"),
    "local_anchor_residual_query_matchqual": ("local", "anchor_residual", "none", "none", 0.01, "none"),
    "local_learned_quality_head": ("local", "learned", "none", "none", 0.1, "none"),
    "local_anchor_quality_head": ("local", "anchor", "none", "none", 0.1, "none"),
    "local_anchor_residual_query_quality_head": ("local", "anchor_residual", "none", "none", 0.01, "none"),
    "local_anchor_residual_query_box_quality_head": ("local", "anchor_residual", "none", "none", 0.01, "none"),
    "local_anchor_residual_query_box_class_head": ("local", "anchor_residual", "none", "none", 0.01, "none"),
    "local_anchor_residual_query_box_class_quality_head": (
        "local",
        "anchor_residual",
        "none",
        "none",
        0.01,
        "none",
    ),
    "local_grid_residual_query": ("local", "grid_residual", "none", "none", 0.01, "none"),
    "local_grid_residual_query_quality_head": ("local", "grid_residual", "none", "none", 0.01, "none"),
    "local_edge_grid_residual_query": ("local", "edge_grid_residual", "none", "none", 0.01, "none"),
    "local_edge_grid_residual_query_quality_head": (
        "local",
        "edge_grid_residual",
        "none",
        "none",
        0.01,
        "none",
    ),
    "local_learned_querymask": ("local", "learned", "query_mask", "query", 0.1, "none"),
    "local_anchor_residual_query_querymask": ("local", "anchor_residual", "query_mask", "query", 0.01, "none"),
    "local_anchor_residual_query_querymask_refine": (
        "local",
        "anchor_residual",
        "query_mask_refine",
        "query",
        0.01,
        "none",
    ),
    "local_anchor_residual_query_querymask_quality_head": (
        "local",
        "anchor_residual",
        "query_mask",
        "query",
        0.01,
        "none",
    ),
    "local_mask_proposal_nms_query_quality_head": (
        "local",
        "mask_proposal_nms",
        "none",
        "proposal",
        0.1,
        "none",
    ),
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
    "local_mask_proposal_nms_query_reinject_g001": (
        "local",
        "mask_proposal_nms",
        "proposal_reinject",
        "proposal",
        0.01,
        "none",
    ),
    "local_mask_proposal_nms_query_reinject_g003": (
        "local",
        "mask_proposal_nms",
        "proposal_reinject",
        "proposal",
        0.03,
        "none",
    ),
    "local_mask_proposal_nms_query_reinject_g003_quality_head": (
        "local",
        "mask_proposal_nms",
        "proposal_reinject",
        "proposal",
        0.03,
        "none",
    ),
    "local_mask_proposal_nms_query_reinject_g003_box_quality_head": (
        "local",
        "mask_proposal_nms",
        "proposal_reinject",
        "proposal",
        0.03,
        "none",
    ),
    "local_mask_proposal_residual_nms_query_reinject_g003_quality_head": (
        "local",
        "mask_proposal_residual_nms",
        "proposal_reinject",
        "proposal",
        0.03,
        "none",
    ),
    "local_mask_proposal_nms_query_reinject_g03": (
        "local",
        "mask_proposal_nms",
        "proposal_reinject",
        "proposal",
        0.3,
        "none",
    ),
    "local_mask_proposal_nms_query_reinject_quality_head": (
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
    "local_mask_proposal_oracle_query": ("local", "mask_proposal_oracle", "none", "none", 0.1, "none"),
    "local_mask_proposal_oracle_nms_query": (
        "local",
        "mask_proposal_oracle_nms",
        "none",
        "none",
        0.1,
        "none",
    ),
    "local_mask_proposal_oracle_nms_query_decode2": (
        "local",
        "mask_proposal_oracle_nms",
        "proposal_decode2",
        "none",
        0.1,
        "none",
    ),
    "local_mask_proposal_oracle_nms_query_reinject": (
        "local",
        "mask_proposal_oracle_nms",
        "proposal_reinject",
        "none",
        0.1,
        "none",
    ),
    "local_mask_proposal_oracle_nms_query_reinject_quality_head": (
        "local",
        "mask_proposal_oracle_nms",
        "proposal_reinject",
        "none",
        0.1,
        "none",
    ),
    "local_mask_proposal_oracle_nms_query_persistent": (
        "local",
        "mask_proposal_oracle_nms",
        "proposal_persistent",
        "none",
        0.1,
        "none",
    ),
    "local_mask_proposal_oracle_nms_query_persistent_gated": (
        "local",
        "mask_proposal_oracle_nms",
        "proposal_persistent",
        "none",
        0.01,
        "none",
    ),
    "local_mask_proposal_oracle_nms_query_late_persistent": (
        "local",
        "mask_proposal_oracle_nms",
        "proposal_late_persistent",
        "none",
        0.1,
        "none",
    ),
    "local_mask_proposal_oracle_nms_query_persistent_stopgrad": (
        "local",
        "mask_proposal_oracle_nms",
        "proposal_persistent_stopgrad",
        "none",
        0.1,
        "none",
    ),
    "local_mask_proposal_oracle_nms_query_quality_head": (
        "local",
        "mask_proposal_oracle_nms",
        "none",
        "none",
        0.1,
        "none",
    ),
    "anchor_mask_proposal_oracle_nms_query": (
        "anchor",
        "mask_proposal_oracle_nms",
        "none",
        "none",
        0.1,
        "none",
    ),
}

CALIBRATED_MODELS = {
    "local_learned_calib",
    "local_anchor_calib",
    "local_anchor_residual_query_calib",
}

MATCH_QUALITY_MODELS = {
    "local_learned_matchqual",
    "local_anchor_matchqual",
    "local_anchor_residual_query_matchqual",
}

QUALITY_HEAD_MODELS = {
    "local_learned_quality_head",
    "local_anchor_quality_head",
    "local_anchor_residual_query_quality_head",
    "local_anchor_residual_query_box_quality_head",
    "local_anchor_residual_query_box_class_quality_head",
    "local_grid_residual_query_quality_head",
    "local_edge_grid_residual_query_quality_head",
    "local_anchor_residual_query_querymask_quality_head",
    "local_mask_proposal_nms_query_quality_head",
    "local_mask_proposal_nms_query_reinject_quality_head",
    "local_mask_proposal_nms_query_reinject_g003_quality_head",
    "local_mask_proposal_nms_query_reinject_g003_box_quality_head",
    "local_mask_proposal_residual_nms_query_reinject_g003_quality_head",
    "local_mask_proposal_oracle_nms_query_quality_head",
    "local_mask_proposal_oracle_nms_query_reinject_quality_head",
}

BOX_QUALITY_HEAD_MODELS = {
    "local_anchor_residual_query_box_quality_head",
    "local_anchor_residual_query_box_class_quality_head",
    "local_mask_proposal_nms_query_reinject_g003_box_quality_head",
}

BOX_CLASS_HEAD_MODELS = {
    "local_anchor_residual_query_box_class_head",
    "local_anchor_residual_query_box_class_quality_head",
}

QUALITY_SCORE_ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class RealBox:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class RealDetSample:
    image_id: str
    image_path: Path
    width: int
    height: int
    boxes: tuple[RealBox, ...]


class RealDetDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """ILSVRC DET image dataset with normalized cxcywh targets."""

    def __init__(
        self,
        samples: list[RealDetSample],
        label_to_id: dict[str, int],
        image_size: int,
        max_objects: int,
    ) -> None:
        self.samples = samples
        self.label_to_id = label_to_id
        self.max_objects = max_objects
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        ranked_boxes = sorted(sample.boxes, key=lambda box: box_area(box), reverse=True)[: self.max_objects]
        labels = []
        boxes = []
        for box in ranked_boxes:
            labels.append(self.label_to_id[box.label])
            boxes.append(normalize_box(box, sample.width, sample.height))
        return tensor, {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument(
        "--anno-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_bbox_val/ILSVRC2013_DET_bbox_val"),
    )
    parser.add_argument("--models", nargs="+", choices=tuple(REAL_MODEL_CONFIGS), default=["local_learned"])
    parser.add_argument("--reference-model", choices=tuple(REAL_MODEL_CONFIGS), default="local_learned")
    parser.add_argument("--top-classes", type=int, default=20)
    parser.add_argument(
        "--label-map-source",
        choices=("all", "train"),
        default="all",
        help=(
            "Source used to choose top classes. 'all' preserves legacy runs; "
            "'train' avoids held-out label-frequency leakage for strict runs."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--calibration-frac",
        type=float,
        default=0.0,
        help="Optional fraction of the held-out split used only to choose quality alpha.",
    )
    parser.add_argument(
        "--calibration-source",
        choices=("heldout", "train"),
        default="heldout",
        help=(
            "Where to carve the calibration split from. 'heldout' preserves legacy "
            "behavior; 'train' keeps the eval split independent of alpha selection."
        ),
    )
    parser.add_argument(
        "--calibration-slice-filter",
        choices=("none", "small", "medium", "large", "center", "offcenter"),
        default="none",
        help=(
            "Optionally choose the alpha-calibration subset only from images containing "
            "this robustness slice. Non-selected train-source candidates remain in train."
        ),
    )
    parser.add_argument(
        "--eval-slice-filter",
        choices=("none", "small", "medium", "large", "center", "offcenter"),
        default="none",
        help=(
            "Optionally restrict the final eval subset to images containing at least one "
            "object in this robustness slice. Training and calibration splits are unchanged."
        ),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--mask-aux-weight", type=float, default=0.5)
    parser.add_argument("--mask-dice-weight", type=float, default=1.0)
    parser.add_argument("--score-iou-weight", type=float, default=0.5)
    parser.add_argument("--quality-cls-weight", type=float, default=1.0)
    parser.add_argument("--quality-head-weight", type=float, default=1.0)
    parser.add_argument("--quality-head-start-step", type=int, default=1)
    parser.add_argument("--quality-head-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--fixed-quality-alpha",
        type=float,
        default=2.0,
        help="Pre-registered quality-score exponent used for fixed, non-post-hoc AP reporting.",
    )
    parser.add_argument(
        "--quality-score-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to quality logits before sigmoid for fixed/calibrated scoring.",
    )
    parser.add_argument(
        "--quality-head-only-after-start",
        action="store_true",
        help="After quality-head start, freeze the detector and train only the query quality head.",
    )
    parser.add_argument(
        "--restore-best-before-quality-head",
        action="store_true",
        help="Before quality-head-only training starts, restore the best detector checkpoint observed so far.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=0,
        help="Calibration batches for alpha selection; 0 reuses --eval-batches.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/det_real_mini_compare.csv"))
    parser.add_argument(
        "--label-map-out",
        type=Path,
        default=Path("results/det_real_mini_label_map.csv"),
    )
    parser.add_argument("--split-out", type=Path, default=Path("results/det_real_mini_split.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_objects < 1:
        raise ValueError("--max-objects must be >= 1")
    if args.max_objects > args.num_queries:
        raise ValueError("--max-objects must be <= --num-queries")
    if args.fixed_quality_alpha < 0:
        raise ValueError("--fixed-quality-alpha must be >= 0")
    if args.quality_score_temperature <= 0:
        raise ValueError("--quality-score-temperature must be > 0")
    if not 0.0 <= args.calibration_frac < 1.0:
        raise ValueError("--calibration-frac must be in [0, 1)")
    device = get_best_device()
    all_samples = load_real_det_samples(args.anno_root, args.image_root)
    label_source_samples = all_samples
    if args.label_map_source == "train":
        indices = torch.randperm(
            len(all_samples),
            generator=torch.Generator().manual_seed(args.seed),
        ).tolist()
        train_size = max(1, min(len(indices), int(len(indices) * args.train_frac)))
        label_source_samples = [all_samples[idx] for idx in indices[:train_size]]
    label_to_id = build_label_map(label_source_samples, top_classes=args.top_classes)
    samples = filter_samples(all_samples, set(label_to_id), max_samples=args.max_samples)
    if len(samples) < 2:
        raise ValueError("not enough real DET samples after filtering")
    write_label_map(args.label_map_out, label_to_id)

    print("device:", device)
    print("samples:", len(samples))
    print("classes:", len(label_to_id))
    print("max_objects:", args.max_objects)
    print("num_queries:", args.num_queries)
    print("label_map_source:", args.label_map_source)
    print("calibration_source:", args.calibration_source)
    print("calibration_slice_filter:", args.calibration_slice_filter)
    print("eval_slice_filter:", args.eval_slice_filter)
    print("label_map:", args.label_map_out)
    print(
        "model,run_seed,step,loss,eval_iou,eval_recall50,eval_ap50,eval_ap50_class,"
        "eval_ap50_q_best,eval_ap50_q_best_alpha,eval_ap50_oracle_iou,"
        "eval_ap50_q_best_oracle_closure,"
        "matched_assignment_class_acc,tp50_class_acc,score_iou_corr,objectness_auc,"
        "quality_iou_corr,quality_auc,combined_iou_corr,combined_auc,topk_fp_rate,"
        "eval_mask_iou,eval_mask_dice,best_iou,best_step,images_per_sec"
    )
    rows: list[dict[str, float | int | str]] = []
    split_rows: list[dict[str, int | str]] = []
    for seed_idx in range(args.seeds):
        run_seed = args.seed + seed_idx
        train_set, calibration_set, eval_set, seed_split_rows = build_splits(
            samples=samples,
            label_to_id=label_to_id,
            image_size=args.image_size,
            max_objects=args.max_objects,
            train_frac=args.train_frac,
            calibration_frac=args.calibration_frac,
            calibration_source=args.calibration_source,
            calibration_slice_filter=args.calibration_slice_filter,
            eval_slice_filter=args.eval_slice_filter,
            seed=run_seed,
        )
        split_rows.extend(seed_split_rows)
        calibration_loader = None
        if calibration_set is not None:
            calibration_loader = DataLoader(
                calibration_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=det_collate,
            )
        eval_loader = DataLoader(
            eval_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=det_collate,
        )
        for model_name in args.models:
            rows.extend(
                train_one_model(
                    args,
                    model_name,
                    run_seed,
                    train_set,
                    eval_loader,
                    device,
                    len(label_to_id),
                    calibration_loader=calibration_loader,
                )
            )
    write_rows(args.out, rows)
    write_split_manifest(args.split_out, split_rows)
    print("saved_csv:", args.out)
    print("saved_split:", args.split_out)
    print_summary(rows)
    print_proposal_oracle_gap_summary(rows)
    print_paired_summary(rows, args.reference_model)


def train_one_model(
    args: argparse.Namespace,
    model_name: str,
    run_seed: int,
    train_set: Dataset,
    eval_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    calibration_loader: DataLoader | None = None,
) -> list[dict[str, float | int | str]]:
    feature_mode, query_init, query_refine, mask_aux_mode, gate_init, gate_schedule = REAL_MODEL_CONFIGS[model_name]
    if "oracle" in query_init and mask_aux_mode != "none":
        raise ValueError("oracle proposal controls must not use mask auxiliary loss")
    torch.manual_seed(run_seed)
    random.seed(run_seed)
    model = TinyAnchorRegionDETR(
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        num_queries=args.num_queries,
        feature_mode=feature_mode,
        query_init=query_init,
        query_refine=query_refine,
        query_mask_gate_init=gate_init,
        quality_mode="box" if model_name in BOX_QUALITY_HEAD_MODELS else "query",
        class_mode="box" if model_name in BOX_CLASS_HEAD_MODELS else "query",
    ).to(device)
    criterion = DetectionCriterion(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=det_collate,
        generator=torch.Generator().manual_seed(10_000_000 + run_seed),
    )
    rows = []
    loader_iter = cycle(train_loader)
    best_iou = -1.0
    best_step = 0
    best_detector_state: dict[str, Tensor] | None = None
    last_loss = 0.0
    examples_seen = 0
    start = time.perf_counter()
    quality_only_enabled = False
    for step in range(1, args.steps + 1):
        if (
            model_name in QUALITY_HEAD_MODELS
            and args.quality_head_only_after_start
            and not quality_only_enabled
            and step >= args.quality_head_start_step
        ):
            if args.restore_best_before_quality_head and best_detector_state is not None:
                model.load_state_dict(best_detector_state)
            set_quality_head_only_trainable(model)
            quality_only_enabled = True
        images, targets = next(loader_iter)
        images = images.to(device)
        targets = targets_to_device(targets, device)
        examples_seen += int(images.shape[0])
        model.set_query_mask_gate_scale(mask_gate_scale(gate_schedule, step, args.steps))
        outputs = forward_real_detector(model, images, targets)
        losses = criterion(outputs, targets)
        loss_score_iou = outputs["pred_logits"].new_tensor(0.0)
        loss_quality_cls = outputs["pred_logits"].new_tensor(0.0)
        loss_quality_head = outputs["pred_logits"].new_tensor(0.0)
        quality_head_scale = quality_head_loss_scale(
            step=step,
            start_step=args.quality_head_start_step,
            warmup_steps=args.quality_head_warmup_steps,
        )
        if model_name in CALIBRATED_MODELS:
            loss_score_iou = score_iou_calibration_loss(outputs, targets)
            losses["loss_score_iou"] = loss_score_iou
            losses["loss"] = losses["loss"] + args.score_iou_weight * loss_score_iou
        if model_name in QUALITY_HEAD_MODELS:
            loss_quality_head = query_quality_head_loss(outputs, targets, criterion)
            losses["loss_quality_head"] = loss_quality_head
            scaled_quality_loss = args.quality_head_weight * quality_head_scale * loss_quality_head
            losses["loss"] = scaled_quality_loss if quality_only_enabled else losses["loss"] + scaled_quality_loss
        if model_name in MATCH_QUALITY_MODELS:
            loss_quality_cls = matcher_aware_quality_classification_loss(
                outputs,
                targets,
                criterion,
            )
            losses["loss_quality_cls"] = loss_quality_cls
            losses["loss"] = losses["loss"] + args.quality_cls_weight * loss_quality_cls
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
                mask_head=None,
                dice_weight=args.mask_dice_weight,
            )
            losses["loss"] = losses["loss"] + args.mask_aux_weight * mask_losses["loss_mask_aux"]
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        last_loss = float(losses["loss"].item())
        if step % args.eval_every == 0 or step == args.steps:
            fixed_quality_alpha = args.fixed_quality_alpha
            if model_name in QUALITY_HEAD_MODELS and calibration_loader is not None:
                calibration_metrics = evaluate_real(
                    model,
                    calibration_loader,
                    device,
                    batches=args.calibration_batches or args.eval_batches,
                    criterion=criterion,
                    use_quality_scores=True,
                    fixed_quality_alpha=args.fixed_quality_alpha,
                    quality_score_temperature=args.quality_score_temperature,
                )
                fixed_quality_alpha = calibration_metrics["ap50_q_best_alpha"]
            metrics = evaluate_real(
                model,
                eval_loader,
                device,
                batches=args.eval_batches,
                criterion=criterion,
                use_quality_scores=model_name in QUALITY_HEAD_MODELS,
                fixed_quality_alpha=fixed_quality_alpha,
                quality_score_temperature=args.quality_score_temperature,
            )
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                best_step = step
                if (
                    model_name in QUALITY_HEAD_MODELS
                    and args.restore_best_before_quality_head
                    and not quality_only_enabled
                ):
                    best_detector_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in copy.deepcopy(model.state_dict()).items()
                    }
            speed = examples_seen / max(time.perf_counter() - start, 1e-9)
            row = {
                "model": model_name,
                "run_seed": run_seed,
                "step": step,
                "loss": last_loss,
                "loss_score_iou": float(loss_score_iou.detach().item()),
                "loss_quality_cls": float(loss_quality_cls.detach().item()),
                "loss_quality_head": float(loss_quality_head.detach().item()),
                "quality_head_scale": quality_head_scale,
                "eval_iou": metrics["iou"],
                "eval_recall50": metrics["recall50"],
                "eval_ap50": metrics["ap50"],
                "eval_ap50_class": metrics["ap50_class"],
                "eval_ap75": metrics["ap75"],
                "eval_ap75_q_fixed": metrics["ap75_q_fixed"],
                "eval_ap75_oracle_iou": metrics["ap75_oracle_iou"],
                "eval_small_ap50": metrics["small_ap50"],
                "eval_medium_ap50": metrics["medium_ap50"],
                "eval_large_ap50": metrics["large_ap50"],
                "eval_center_ap50": metrics["center_ap50"],
                "eval_offcenter_ap50": metrics["offcenter_ap50"],
                "eval_small_ap50_q_fixed": metrics["small_ap50_q_fixed"],
                "eval_medium_ap50_q_fixed": metrics["medium_ap50_q_fixed"],
                "eval_large_ap50_q_fixed": metrics["large_ap50_q_fixed"],
                "eval_center_ap50_q_fixed": metrics["center_ap50_q_fixed"],
                "eval_offcenter_ap50_q_fixed": metrics["offcenter_ap50_q_fixed"],
                "eval_ap50_center_dist1": metrics["ap50_center_dist1"],
                "eval_ap50_q_fixed_center_dist1": metrics["ap50_q_fixed_center_dist1"],
                "eval_center_ap50_center_dist1": metrics["center_ap50_center_dist1"],
                "eval_offcenter_ap50_center_dist1": metrics["offcenter_ap50_center_dist1"],
                "eval_center_ap50_q_fixed_center_dist1": metrics[
                    "center_ap50_q_fixed_center_dist1"
                ],
                "eval_offcenter_ap50_q_fixed_center_dist1": metrics[
                    "offcenter_ap50_q_fixed_center_dist1"
                ],
                "eval_ap50_q025": metrics["ap50_q025"],
                "eval_ap50_q05": metrics["ap50_q05"],
                "eval_ap50_q1": metrics["ap50_q1"],
                "eval_ap50_q2": metrics["ap50_q2"],
                "eval_ap50_q4": metrics["ap50_q4"],
                "eval_ap50_q_fixed": metrics["ap50_q_fixed"],
                "eval_ap50_q_fixed_alpha": metrics["ap50_q_fixed_alpha"],
                "eval_ap50_q_fixed_temperature": metrics["ap50_q_fixed_temperature"],
                "eval_ap50_q_best": metrics["ap50_q_best"],
                "eval_ap50_q_best_alpha": metrics["ap50_q_best_alpha"],
                "eval_ap50_class_q025": metrics["ap50_class_q025"],
                "eval_ap50_class_q05": metrics["ap50_class_q05"],
                "eval_ap50_class_q1": metrics["ap50_class_q1"],
                "eval_ap50_class_q2": metrics["ap50_class_q2"],
                "eval_ap50_class_q4": metrics["ap50_class_q4"],
                "eval_ap50_class_q_fixed": metrics["ap50_class_q_fixed"],
                "eval_ap50_class_q_best": metrics["ap50_class_q_best"],
                "eval_ap50_class_q_best_alpha": metrics["ap50_class_q_best_alpha"],
                "eval_ap50_oracle_iou": metrics["ap50_oracle_iou"],
                "eval_ap50_class_oracle_iou": metrics["ap50_class_oracle_iou"],
                "eval_ap50_oracle_gap": metrics["ap50_oracle_gap"],
                "eval_ap50_q_fixed_gap_to_oracle": metrics["ap50_q_fixed_gap_to_oracle"],
                "eval_ap50_q_fixed_oracle_closure": metrics["ap50_q_fixed_oracle_closure"],
                "eval_ap50_q_best_gap_to_oracle": metrics["ap50_q_best_gap_to_oracle"],
                "eval_ap50_q_best_oracle_closure": metrics["ap50_q_best_oracle_closure"],
                "eval_ap50_class_oracle_gap": metrics["ap50_class_oracle_gap"],
                "eval_ap50_class_q_fixed_gap_to_oracle": metrics["ap50_class_q_fixed_gap_to_oracle"],
                "eval_ap50_class_q_fixed_oracle_closure": metrics["ap50_class_q_fixed_oracle_closure"],
                "eval_ap50_class_q_best_gap_to_oracle": metrics["ap50_class_q_best_gap_to_oracle"],
                "eval_ap50_class_q_best_oracle_closure": metrics["ap50_class_q_best_oracle_closure"],
                "eval_mask_iou": metrics["mask_iou"],
                "eval_mask_dice": metrics["mask_dice"],
                "eval_small_iou": metrics["small_iou"],
                "eval_medium_iou": metrics["medium_iou"],
                "eval_large_iou": metrics["large_iou"],
                "eval_center_iou": metrics["center_iou"],
                "eval_offcenter_iou": metrics["offcenter_iou"],
                "matched_assignment_class_acc": metrics["matched_assignment_class_acc"],
                "tp50_class_acc": metrics["tp50_class_acc"],
                "score_iou_corr": metrics["score_iou_corr"],
                "objectness_auc": metrics["objectness_auc"],
                "objectness_ece50": metrics["objectness_ece50"],
                "objectness_ece75": metrics["objectness_ece75"],
                "quality_iou_corr": metrics["quality_iou_corr"],
                "quality_auc": metrics["quality_auc"],
                "combined_iou_corr": metrics["combined_iou_corr"],
                "combined_auc": metrics["combined_auc"],
                "combined_ece50": metrics["combined_ece50"],
                "combined_ece75": metrics["combined_ece75"],
                "topk_fp_rate": metrics["topk_fp_rate"],
                "combined_topk_fp_rate": metrics["combined_topk_fp_rate"],
                "topk_center_distance": metrics["topk_center_distance"],
                "combined_topk_center_distance": metrics["combined_topk_center_distance"],
                "duplicate_per_gt": metrics["duplicate_per_gt"],
                "center_matched_assignment_class_acc": metrics["center_matched_assignment_class_acc"],
                "offcenter_matched_assignment_class_acc": metrics["offcenter_matched_assignment_class_acc"],
                "center_tp50_class_acc": metrics["center_tp50_class_acc"],
                "offcenter_tp50_class_acc": metrics["offcenter_tp50_class_acc"],
                "center_score_iou_corr": metrics["center_score_iou_corr"],
                "offcenter_score_iou_corr": metrics["offcenter_score_iou_corr"],
                "center_objectness_auc": metrics["center_objectness_auc"],
                "offcenter_objectness_auc": metrics["offcenter_objectness_auc"],
                "center_topk_fp_rate": metrics["center_topk_fp_rate"],
                "offcenter_topk_fp_rate": metrics["offcenter_topk_fp_rate"],
                "center_combined_topk_fp_rate": metrics["center_combined_topk_fp_rate"],
                "offcenter_combined_topk_fp_rate": metrics["offcenter_combined_topk_fp_rate"],
                "center_topk_slice_match_rate": metrics["center_topk_slice_match_rate"],
                "offcenter_topk_slice_match_rate": metrics["offcenter_topk_slice_match_rate"],
                "center_topk_non_slice_match_rate": metrics["center_topk_non_slice_match_rate"],
                "offcenter_topk_non_slice_match_rate": metrics["offcenter_topk_non_slice_match_rate"],
                "center_topk_no_gt_match_rate": metrics["center_topk_no_gt_match_rate"],
                "offcenter_topk_no_gt_match_rate": metrics["offcenter_topk_no_gt_match_rate"],
                "center_combined_topk_slice_match_rate": metrics["center_combined_topk_slice_match_rate"],
                "offcenter_combined_topk_slice_match_rate": metrics["offcenter_combined_topk_slice_match_rate"],
                "center_combined_topk_non_slice_match_rate": metrics[
                    "center_combined_topk_non_slice_match_rate"
                ],
                "offcenter_combined_topk_non_slice_match_rate": metrics[
                    "offcenter_combined_topk_non_slice_match_rate"
                ],
                "center_combined_topk_no_gt_match_rate": metrics["center_combined_topk_no_gt_match_rate"],
                "offcenter_combined_topk_no_gt_match_rate": metrics[
                    "offcenter_combined_topk_no_gt_match_rate"
                ],
                "center_topk_center_distance": metrics["center_topk_center_distance"],
                "offcenter_topk_center_distance": metrics["offcenter_topk_center_distance"],
                "center_combined_topk_center_distance": metrics["center_combined_topk_center_distance"],
                "offcenter_combined_topk_center_distance": metrics["offcenter_combined_topk_center_distance"],
                "center_duplicate_per_gt": metrics["center_duplicate_per_gt"],
                "offcenter_duplicate_per_gt": metrics["offcenter_duplicate_per_gt"],
                "query_assignment_entropy": metrics["query_assignment_entropy"],
                "best_iou": best_iou,
                "best_step": best_step,
                "images_per_sec": speed,
            }
            rows.append(row)
            print(
                f"{model_name},{run_seed},{step},{last_loss:.4f},"
                f"{metrics['iou']:.3f},{metrics['recall50']:.3f},"
                f"{metrics['ap50']:.3f},{metrics['ap50_class']:.3f},"
                f"{metrics['ap50_q_best']:.3f},{metrics['ap50_q_best_alpha']:.2f},"
                f"{metrics['ap50_oracle_iou']:.3f},{metrics['ap50_q_best_oracle_closure']:.3f},"
                f"{metrics['matched_assignment_class_acc']:.3f},{metrics['tp50_class_acc']:.3f},"
                f"{metrics['score_iou_corr']:.3f},{metrics['objectness_auc']:.3f},"
                f"{metrics['quality_iou_corr']:.3f},{metrics['quality_auc']:.3f},"
                f"{metrics['combined_iou_corr']:.3f},{metrics['combined_auc']:.3f},"
                f"{metrics['topk_fp_rate']:.3f},"
                f"{metrics['mask_iou']:.3f},{metrics['mask_dice']:.3f},"
                f"{best_iou:.3f},{best_step},{speed:.2f}"
            )
    return rows


def forward_real_detector(
    model: TinyAnchorRegionDETR,
    images: Tensor,
    targets: list[dict[str, Tensor]],
) -> dict[str, Tensor | list[Tensor]]:
    """Run real DET variants, injecting GT mask proposals only for oracle controls."""

    if model.query_init in {"mask_proposal_oracle", "mask_proposal_oracle_nms"}:
        mask_logits = oracle_query_mask_logits(
            targets=targets,
            feature_height=images.shape[-2] // model.patch_embed.proj.stride[0],
            feature_width=images.shape[-1] // model.patch_embed.proj.stride[1],
            device=images.device,
            dtype=images.dtype,
        )
        return model(images, query_mask_logits_override=mask_logits)
    return model(images)


def oracle_query_mask_logits(
    targets: list[dict[str, Tensor]],
    feature_height: int,
    feature_width: int,
    device: torch.device,
    dtype: torch.dtype,
    logit_abs: float = 8.0,
) -> Tensor:
    """Build high-confidence GT union-mask logits for oracle proposal-query controls."""

    masks = dense_mask_targets_from_boxes(
        targets=targets,
        height=feature_height,
        width=feature_width,
        device=device,
    ).to(dtype=dtype)
    positive = torch.full_like(masks, float(logit_abs))
    negative = torch.full_like(masks, -float(logit_abs))
    return torch.where(masks >= 0.5, positive, negative)


def score_iou_calibration_loss(
    outputs: dict[str, Tensor | list[Tensor]],
    targets: list[dict[str, Tensor]],
) -> Tensor:
    """Align query objectness logits with detached max-IoU targets."""

    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    batch_losses = []
    for batch_idx, target in enumerate(targets):
        target_boxes = target["boxes"].to(pred_boxes.device)
        if target_boxes.numel() == 0:
            iou_target = pred_boxes.new_zeros(pred_boxes.shape[1])
        else:
            with torch.no_grad():
                iou_target = box_iou(
                    box_cxcywh_to_xyxy(pred_boxes[batch_idx].detach()),
                    box_cxcywh_to_xyxy(target_boxes),
                ).max(dim=1).values.clamp(0.0, 1.0)
        objectness_logit = objectness_logits(pred_logits[batch_idx])
        batch_losses.append(F.binary_cross_entropy_with_logits(objectness_logit, iou_target))
    return torch.stack(batch_losses).mean()


def matcher_aware_quality_classification_loss(
    outputs: dict[str, Tensor | list[Tensor]],
    targets: list[dict[str, Tensor]],
    criterion: DetectionCriterion,
) -> Tensor:
    """Train matched class logits with IoU-quality targets and unmatched as background."""

    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    quality_targets = torch.zeros_like(pred_logits[:, :, :-1])
    matches = criterion.matcher(outputs, targets)
    for batch_idx, (src_idx, target_idx) in enumerate(matches):
        if src_idx.numel() == 0:
            continue
        labels = targets[batch_idx]["labels"][target_idx].to(pred_logits.device)
        target_boxes = targets[batch_idx]["boxes"][target_idx].to(pred_boxes.device)
        with torch.no_grad():
            matched_iou = box_iou(
                box_cxcywh_to_xyxy(pred_boxes[batch_idx, src_idx].detach()),
                box_cxcywh_to_xyxy(target_boxes),
            ).diag().clamp(0.0, 1.0)
        quality_targets[batch_idx, src_idx, labels] = matched_iou
    return F.binary_cross_entropy_with_logits(pred_logits[:, :, :-1], quality_targets)


def query_quality_head_loss(
    outputs: dict[str, Tensor | list[Tensor]],
    targets: list[dict[str, Tensor]],
    criterion: DetectionCriterion,
) -> Tensor:
    """Train an independent query-quality head with matched IoU targets."""

    pred_quality_logits = outputs["pred_quality_logits"]
    pred_boxes = outputs["pred_boxes"]
    quality_targets = torch.zeros_like(pred_quality_logits)
    matches = criterion.matcher(outputs, targets)
    for batch_idx, (src_idx, target_idx) in enumerate(matches):
        if src_idx.numel() == 0:
            continue
        target_boxes = targets[batch_idx]["boxes"][target_idx].to(pred_boxes.device)
        with torch.no_grad():
            matched_iou = box_iou(
                box_cxcywh_to_xyxy(pred_boxes[batch_idx, src_idx].detach()),
                box_cxcywh_to_xyxy(target_boxes),
            ).diag().clamp(0.0, 1.0)
        quality_targets[batch_idx, src_idx] = matched_iou
    return F.binary_cross_entropy_with_logits(pred_quality_logits, quality_targets)


def quality_head_loss_scale(step: int, start_step: int, warmup_steps: int) -> float:
    """Return the runtime multiplier for late-start quality-head training."""

    if step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    progress = (step - start_step + 1) / float(warmup_steps)
    return max(0.0, min(1.0, progress))


def set_quality_head_only_trainable(model: TinyAnchorRegionDETR) -> None:
    """Freeze the detector and leave only the query quality head trainable."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.quality_head.parameters():
        parameter.requires_grad_(True)


def objectness_logits(pred_logits: Tensor) -> Tensor:
    """Return object-vs-no-object logits from DETR class logits."""

    foreground = torch.logsumexp(pred_logits[:, :-1], dim=-1)
    background = pred_logits[:, -1]
    return foreground - background


@torch.no_grad()
def evaluate_real(
    model: TinyAnchorRegionDETR,
    loader: DataLoader,
    device: torch.device,
    batches: int,
    criterion: DetectionCriterion,
    use_quality_scores: bool = False,
    fixed_quality_alpha: float = 2.0,
    quality_score_temperature: float = 1.0,
) -> dict[str, float]:
    model.eval()
    ious = []
    recalls = []
    aps = []
    aps_class = []
    aps75 = []
    aps75_q_fixed = []
    aps75_oracle_iou = []
    slice_aps: dict[str, list[Tensor]] = {
        "small": [],
        "medium": [],
        "large": [],
        "center": [],
        "offcenter": [],
    }
    slice_aps_q_fixed: dict[str, list[Tensor]] = {
        "small": [],
        "medium": [],
        "large": [],
        "center": [],
        "offcenter": [],
    }
    slice_aps_center_dist1: dict[str, list[Tensor]] = {
        "small": [],
        "medium": [],
        "large": [],
        "center": [],
        "offcenter": [],
    }
    slice_aps_q_fixed_center_dist1: dict[str, list[Tensor]] = {
        "small": [],
        "medium": [],
        "large": [],
        "center": [],
        "offcenter": [],
    }
    aps_q025 = []
    aps_q05 = []
    aps_q1 = []
    aps_q2 = []
    aps_q4 = []
    aps_q_fixed = []
    aps_center_dist1 = []
    aps_q_fixed_center_dist1 = []
    aps_class_q025 = []
    aps_class_q05 = []
    aps_class_q1 = []
    aps_class_q2 = []
    aps_class_q4 = []
    aps_class_q_fixed = []
    aps_oracle_iou = []
    aps_class_oracle_iou = []
    matched_assignment_class_correct = []
    tp50_class_correct = []
    score_iou_corrs = []
    objectness_aucs = []
    objectness_ece50s = []
    objectness_ece75s = []
    quality_iou_corrs = []
    quality_aucs = []
    combined_iou_corrs = []
    combined_aucs = []
    combined_ece50s = []
    combined_ece75s = []
    topk_fp_rates = []
    combined_topk_fp_rates = []
    topk_center_distances = []
    combined_topk_center_distances = []
    duplicate_per_gt_values = []
    query_assignment_counts = torch.zeros(model.num_queries, dtype=torch.float32)
    slice_diag_vectors: dict[str, dict[str, list[Tensor]]] = {
        "center": {
            "matched_assignment_class_acc": [],
            "tp50_class_acc": [],
        },
        "offcenter": {
            "matched_assignment_class_acc": [],
            "tp50_class_acc": [],
        },
    }
    slice_diag_scalars: dict[str, dict[str, list[Tensor]]] = {
        "center": {
            "score_iou_corr": [],
            "objectness_auc": [],
            "topk_fp_rate": [],
            "combined_topk_fp_rate": [],
            "topk_slice_match_rate": [],
            "topk_non_slice_match_rate": [],
            "topk_no_gt_match_rate": [],
            "combined_topk_slice_match_rate": [],
            "combined_topk_non_slice_match_rate": [],
            "combined_topk_no_gt_match_rate": [],
            "topk_center_distance": [],
            "combined_topk_center_distance": [],
            "duplicate_per_gt": [],
        },
        "offcenter": {
            "score_iou_corr": [],
            "objectness_auc": [],
            "topk_fp_rate": [],
            "combined_topk_fp_rate": [],
            "topk_slice_match_rate": [],
            "topk_non_slice_match_rate": [],
            "topk_no_gt_match_rate": [],
            "combined_topk_slice_match_rate": [],
            "combined_topk_non_slice_match_rate": [],
            "combined_topk_no_gt_match_rate": [],
            "topk_center_distance": [],
            "combined_topk_center_distance": [],
            "duplicate_per_gt": [],
        },
    }
    mask_ious = []
    mask_dices = []
    strata: dict[str, list[Tensor]] = {
        "small": [],
        "medium": [],
        "large": [],
        "center": [],
        "offcenter": [],
    }
    for batch_idx, (images, targets_cpu) in enumerate(loader):
        if batch_idx >= batches:
            break
        images = images.to(device)
        targets = targets_to_device(targets_cpu, device)
        outputs = forward_real_detector(model, images, targets)
        if "query_mask_logits_per_query" in outputs:
            mask_metrics = query_mask_aux_metrics(outputs, targets, criterion)
            mask_ious.append(mask_metrics["mask_iou"].cpu())
            mask_dices.append(mask_metrics["mask_dice"].cpu())
        elif "query_mask_logits" in outputs:
            mask_metrics = dense_mask_aux_metrics(outputs, targets, mask_head=None)
            mask_ious.append(mask_metrics["mask_iou"].cpu())
            mask_dices.append(mask_metrics["mask_dice"].cpu())
        for sample_idx, target in enumerate(targets):
            matched_iou = match_targets_by_iou(
                outputs["pred_boxes"][sample_idx],
                target["boxes"],
            )
            quality_logits = (
                outputs["pred_quality_logits"][sample_idx]
                if use_quality_scores and "pred_quality_logits" in outputs
                else None
            )
            fixed_quality_scores = fixed_quality_score_multiplier(
                quality_logits,
                alpha=fixed_quality_alpha,
                temperature=quality_score_temperature,
            )
            center_distance_scores = center_distance_score_multiplier(outputs["pred_boxes"][sample_idx])
            q_fixed_center_distance_scores = (
                fixed_quality_scores * center_distance_scores
                if fixed_quality_scores is not None
                else center_distance_scores
            )
            diagnostics = query_ranking_diagnostics(
                pred_logits=outputs["pred_logits"][sample_idx],
                pred_boxes=outputs["pred_boxes"][sample_idx],
                target_boxes=target["boxes"],
                target_labels=target["labels"],
                quality_logits=quality_logits,
                combined_quality_scores=fixed_quality_scores,
            )
            ious.append(matched_iou.cpu())
            recalls.append((matched_iou >= 0.5).float().cpu())
            matched_assignment_class_correct.append(
                diagnostics["matched_assignment_class_correct"].cpu()
            )
            tp50_class_correct.append(diagnostics["tp50_class_correct"].cpu())
            score_iou_corrs.append(diagnostics["score_iou_corr"].cpu())
            objectness_aucs.append(diagnostics["objectness_auc"].cpu())
            objectness_ece50s.append(diagnostics["objectness_ece50"].cpu())
            objectness_ece75s.append(diagnostics["objectness_ece75"].cpu())
            quality_iou_corrs.append(diagnostics["quality_iou_corr"].cpu())
            quality_aucs.append(diagnostics["quality_auc"].cpu())
            combined_iou_corrs.append(diagnostics["combined_iou_corr"].cpu())
            combined_aucs.append(diagnostics["combined_auc"].cpu())
            combined_ece50s.append(diagnostics["combined_ece50"].cpu())
            combined_ece75s.append(diagnostics["combined_ece75"].cpu())
            topk_fp_rates.append(diagnostics["topk_fp_rate"].cpu())
            combined_topk_fp_rates.append(diagnostics["combined_topk_fp_rate"].cpu())
            topk_center_distances.append(diagnostics["topk_center_distance"].cpu())
            combined_topk_center_distances.append(diagnostics["combined_topk_center_distance"].cpu())
            duplicate_per_gt_values.append(diagnostics["duplicate_per_gt"].cpu())
            query_assignment_counts += diagnostics["matched_query_counts"].cpu()
            update_stratified_iou_lists(strata, matched_iou.cpu(), target["boxes"].cpu())
            aps.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                ).cpu()
            )
            aps75.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    iou_threshold=0.75,
                ).cpu()
            )
            aps_class.append(
                class_aware_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    target["labels"],
                ).cpu()
            )
            quality_scores = quality_score_multipliers(
                quality_logits,
            )
            oracle_iou_scores = oracle_iou_score_multiplier(
                outputs["pred_boxes"][sample_idx],
                target["boxes"],
            )
            for slice_name, slice_mask in box_slice_masks(target["boxes"]).items():
                if not bool(slice_mask.any()):
                    continue
                slice_boxes = target["boxes"][slice_mask]
                slice_labels = target["labels"][slice_mask]
                slice_aps[slice_name].append(
                    objectness_ap50_for_image(
                        outputs["pred_logits"][sample_idx],
                        outputs["pred_boxes"][sample_idx],
                        slice_boxes,
                    ).cpu()
                )
                slice_aps_q_fixed[slice_name].append(
                    objectness_ap50_for_image(
                        outputs["pred_logits"][sample_idx],
                        outputs["pred_boxes"][sample_idx],
                        slice_boxes,
                        score_multiplier=fixed_quality_scores,
                    ).cpu()
                )
                slice_aps_center_dist1[slice_name].append(
                    objectness_ap50_for_image(
                        outputs["pred_logits"][sample_idx],
                        outputs["pred_boxes"][sample_idx],
                        slice_boxes,
                        score_multiplier=center_distance_scores,
                    ).cpu()
                )
                slice_aps_q_fixed_center_dist1[slice_name].append(
                    objectness_ap50_for_image(
                        outputs["pred_logits"][sample_idx],
                        outputs["pred_boxes"][sample_idx],
                        slice_boxes,
                        score_multiplier=q_fixed_center_distance_scores,
                    ).cpu()
                )
                if slice_name in slice_diag_vectors:
                    slice_diagnostics = query_ranking_diagnostics(
                        pred_logits=outputs["pred_logits"][sample_idx],
                        pred_boxes=outputs["pred_boxes"][sample_idx],
                        target_boxes=slice_boxes,
                        target_labels=slice_labels,
                        quality_logits=quality_logits,
                        combined_quality_scores=fixed_quality_scores,
                        all_target_boxes=target["boxes"],
                    )
                    for metric_name, values in slice_diag_vectors[slice_name].items():
                        source_name = (
                            "matched_assignment_class_correct"
                            if metric_name == "matched_assignment_class_acc"
                            else "tp50_class_correct"
                        )
                        values.append(slice_diagnostics[source_name].cpu())
                    for metric_name, values in slice_diag_scalars[slice_name].items():
                        values.append(slice_diagnostics[metric_name].cpu())
            aps_q025.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=quality_scores[0.25],
                ).cpu()
            )
            aps_q05.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=quality_scores[0.5],
                ).cpu()
            )
            aps_q1.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=quality_scores[1.0],
                ).cpu()
            )
            aps_q2.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=quality_scores[2.0],
                ).cpu()
            )
            aps_q4.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=quality_scores[4.0],
                ).cpu()
            )
            aps_q_fixed.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=fixed_quality_scores,
                ).cpu()
            )
            aps_center_dist1.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=center_distance_scores,
                ).cpu()
            )
            aps_q_fixed_center_dist1.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=q_fixed_center_distance_scores,
                ).cpu()
            )
            aps75_q_fixed.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=fixed_quality_scores,
                    iou_threshold=0.75,
                ).cpu()
            )
            aps_class_q025.append(
                class_aware_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    target["labels"],
                    score_multiplier=quality_scores[0.25],
                ).cpu()
            )
            aps_class_q05.append(
                class_aware_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    target["labels"],
                    score_multiplier=quality_scores[0.5],
                ).cpu()
            )
            aps_class_q1.append(
                class_aware_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    target["labels"],
                    score_multiplier=quality_scores[1.0],
                ).cpu()
            )
            aps_class_q2.append(
                class_aware_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    target["labels"],
                    score_multiplier=quality_scores[2.0],
                ).cpu()
            )
            aps_class_q4.append(
                class_aware_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    target["labels"],
                    score_multiplier=quality_scores[4.0],
                ).cpu()
            )
            aps_class_q_fixed.append(
                class_aware_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    target["labels"],
                    score_multiplier=fixed_quality_scores,
                ).cpu()
            )
            aps_oracle_iou.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=oracle_iou_scores,
                ).cpu()
            )
            aps75_oracle_iou.append(
                objectness_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    score_multiplier=oracle_iou_scores,
                    iou_threshold=0.75,
                ).cpu()
            )
            aps_class_oracle_iou.append(
                class_aware_ap50_for_image(
                    outputs["pred_logits"][sample_idx],
                    outputs["pred_boxes"][sample_idx],
                    target["boxes"],
                    target["labels"],
                    score_multiplier=oracle_iou_scores,
                ).cpu()
            )
    model.train()
    all_ious = torch.cat(ious) if ious else torch.zeros(1)
    all_recalls = torch.cat(recalls) if recalls else torch.zeros(1)
    all_assignment_class_correct = (
        torch.cat(matched_assignment_class_correct)
        if matched_assignment_class_correct
        else torch.zeros(1)
    )
    all_tp50_class_correct = torch.cat(tp50_class_correct) if tp50_class_correct else torch.zeros(0)
    tp50_class_acc = (
        float(all_tp50_class_correct.float().mean().item())
        if all_tp50_class_correct.numel() > 0
        else 0.0
    )
    if mask_ious:
        mask_iou = float(torch.stack(mask_ious).mean().item())
        mask_dice = float(torch.stack(mask_dices).mean().item())
    else:
        mask_iou = 0.0
        mask_dice = 0.0
    ap50 = float(torch.stack(aps).mean().item()) if aps else 0.0
    ap50_class = float(torch.stack(aps_class).mean().item()) if aps_class else 0.0
    ap75 = float(torch.stack(aps75).mean().item()) if aps75 else 0.0
    ap75_q_fixed = float(torch.stack(aps75_q_fixed).mean().item()) if aps75_q_fixed else 0.0
    ap75_oracle_iou = float(torch.stack(aps75_oracle_iou).mean().item()) if aps75_oracle_iou else 0.0
    slice_ap50 = summarize_slice_ap(slice_aps)
    slice_ap50_q_fixed = summarize_slice_ap(slice_aps_q_fixed)
    slice_ap50_center_dist1 = summarize_slice_ap(slice_aps_center_dist1)
    slice_ap50_q_fixed_center_dist1 = summarize_slice_ap(slice_aps_q_fixed_center_dist1)
    ap50_q_by_alpha = {
        0.25: float(torch.stack(aps_q025).mean().item()) if aps_q025 else 0.0,
        0.5: float(torch.stack(aps_q05).mean().item()) if aps_q05 else 0.0,
        1.0: float(torch.stack(aps_q1).mean().item()) if aps_q1 else 0.0,
        2.0: float(torch.stack(aps_q2).mean().item()) if aps_q2 else 0.0,
        4.0: float(torch.stack(aps_q4).mean().item()) if aps_q4 else 0.0,
    }
    ap50_q_fixed = float(torch.stack(aps_q_fixed).mean().item()) if aps_q_fixed else 0.0
    ap50_center_dist1 = float(torch.stack(aps_center_dist1).mean().item()) if aps_center_dist1 else 0.0
    ap50_q_fixed_center_dist1 = (
        float(torch.stack(aps_q_fixed_center_dist1).mean().item())
        if aps_q_fixed_center_dist1
        else 0.0
    )
    ap50_class_q_by_alpha = {
        0.25: float(torch.stack(aps_class_q025).mean().item()) if aps_class_q025 else 0.0,
        0.5: float(torch.stack(aps_class_q05).mean().item()) if aps_class_q05 else 0.0,
        1.0: float(torch.stack(aps_class_q1).mean().item()) if aps_class_q1 else 0.0,
        2.0: float(torch.stack(aps_class_q2).mean().item()) if aps_class_q2 else 0.0,
        4.0: float(torch.stack(aps_class_q4).mean().item()) if aps_class_q4 else 0.0,
    }
    ap50_class_q_fixed = float(torch.stack(aps_class_q_fixed).mean().item()) if aps_class_q_fixed else 0.0
    ap50_oracle_iou = float(torch.stack(aps_oracle_iou).mean().item()) if aps_oracle_iou else 0.0
    ap50_class_oracle_iou = (
        float(torch.stack(aps_class_oracle_iou).mean().item()) if aps_class_oracle_iou else 0.0
    )
    best_q_alpha, best_q_ap50 = max(ap50_q_by_alpha.items(), key=lambda item: item[1])
    best_class_q_alpha, best_class_q_ap50 = max(ap50_class_q_by_alpha.items(), key=lambda item: item[1])
    if not use_quality_scores:
        best_q_alpha = 0.0
        best_class_q_alpha = 0.0
    return {
        "iou": float(all_ious.mean().item()),
        "recall50": float(all_recalls.mean().item()),
        "ap50": ap50,
        "ap50_class": ap50_class,
        "ap75": ap75,
        "ap75_q_fixed": ap75_q_fixed,
        "ap75_oracle_iou": ap75_oracle_iou,
        **{f"{name}_ap50": value for name, value in slice_ap50.items()},
        **{f"{name}_ap50_q_fixed": value for name, value in slice_ap50_q_fixed.items()},
        **{f"{name}_ap50_center_dist1": value for name, value in slice_ap50_center_dist1.items()},
        **{
            f"{name}_ap50_q_fixed_center_dist1": value
            for name, value in slice_ap50_q_fixed_center_dist1.items()
        },
        "ap50_q025": ap50_q_by_alpha[0.25],
        "ap50_q05": ap50_q_by_alpha[0.5],
        "ap50_q1": ap50_q_by_alpha[1.0],
        "ap50_q2": ap50_q_by_alpha[2.0],
        "ap50_q4": ap50_q_by_alpha[4.0],
        "ap50_q_fixed": ap50_q_fixed,
        "ap50_center_dist1": ap50_center_dist1,
        "ap50_q_fixed_center_dist1": ap50_q_fixed_center_dist1,
        "ap50_q_fixed_alpha": fixed_quality_alpha if use_quality_scores else 0.0,
        "ap50_q_fixed_temperature": quality_score_temperature if use_quality_scores else 0.0,
        "ap50_q_best": best_q_ap50,
        "ap50_q_best_alpha": best_q_alpha,
        "ap50_class_q025": ap50_class_q_by_alpha[0.25],
        "ap50_class_q05": ap50_class_q_by_alpha[0.5],
        "ap50_class_q1": ap50_class_q_by_alpha[1.0],
        "ap50_class_q2": ap50_class_q_by_alpha[2.0],
        "ap50_class_q4": ap50_class_q_by_alpha[4.0],
        "ap50_class_q_fixed": ap50_class_q_fixed,
        "ap50_class_q_best": best_class_q_ap50,
        "ap50_class_q_best_alpha": best_class_q_alpha,
        "ap50_oracle_iou": ap50_oracle_iou,
        "ap50_class_oracle_iou": ap50_class_oracle_iou,
        "ap50_oracle_gap": ap50_oracle_iou - ap50,
        "ap50_q_fixed_gap_to_oracle": ap50_oracle_iou - ap50_q_fixed,
        "ap50_q_fixed_oracle_closure": ranking_gap_closure(ap50, ap50_q_fixed, ap50_oracle_iou),
        "ap50_q_best_gap_to_oracle": ap50_oracle_iou - best_q_ap50,
        "ap50_q_best_oracle_closure": ranking_gap_closure(ap50, best_q_ap50, ap50_oracle_iou),
        "ap50_class_oracle_gap": ap50_class_oracle_iou - ap50_class,
        "ap50_class_q_fixed_gap_to_oracle": ap50_class_oracle_iou - ap50_class_q_fixed,
        "ap50_class_q_fixed_oracle_closure": ranking_gap_closure(
            ap50_class,
            ap50_class_q_fixed,
            ap50_class_oracle_iou,
        ),
        "ap50_class_q_best_gap_to_oracle": ap50_class_oracle_iou - best_class_q_ap50,
        "ap50_class_q_best_oracle_closure": ranking_gap_closure(
            ap50_class,
            best_class_q_ap50,
            ap50_class_oracle_iou,
        ),
        "matched_assignment_class_acc": float(all_assignment_class_correct.float().mean().item()),
        "tp50_class_acc": tp50_class_acc,
        "score_iou_corr": float(torch.stack(score_iou_corrs).mean().item()) if score_iou_corrs else 0.0,
        "objectness_auc": float(torch.stack(objectness_aucs).mean().item()) if objectness_aucs else 0.0,
        "objectness_ece50": float(torch.stack(objectness_ece50s).mean().item()) if objectness_ece50s else 0.0,
        "objectness_ece75": float(torch.stack(objectness_ece75s).mean().item()) if objectness_ece75s else 0.0,
        "quality_iou_corr": float(torch.stack(quality_iou_corrs).mean().item()) if quality_iou_corrs else 0.0,
        "quality_auc": float(torch.stack(quality_aucs).mean().item()) if quality_aucs else 0.0,
        "combined_iou_corr": float(torch.stack(combined_iou_corrs).mean().item()) if combined_iou_corrs else 0.0,
        "combined_auc": float(torch.stack(combined_aucs).mean().item()) if combined_aucs else 0.0,
        "combined_ece50": float(torch.stack(combined_ece50s).mean().item()) if combined_ece50s else 0.0,
        "combined_ece75": float(torch.stack(combined_ece75s).mean().item()) if combined_ece75s else 0.0,
        "topk_fp_rate": float(torch.stack(topk_fp_rates).mean().item()) if topk_fp_rates else 0.0,
        "combined_topk_fp_rate": (
            float(torch.stack(combined_topk_fp_rates).mean().item()) if combined_topk_fp_rates else 0.0
        ),
        "topk_center_distance": (
            float(torch.stack(topk_center_distances).mean().item()) if topk_center_distances else 0.0
        ),
        "combined_topk_center_distance": (
            float(torch.stack(combined_topk_center_distances).mean().item())
            if combined_topk_center_distances
            else 0.0
        ),
        "duplicate_per_gt": float(torch.stack(duplicate_per_gt_values).mean().item()) if duplicate_per_gt_values else 0.0,
        **summarize_slice_ranking_diagnostics(slice_diag_vectors, slice_diag_scalars),
        "query_assignment_entropy": assignment_entropy(query_assignment_counts),
        "mask_iou": mask_iou,
        "mask_dice": mask_dice,
        **summarize_stratified_iou(strata),
    }


def quality_score_multipliers(quality_logits: Tensor | None) -> dict[float, Tensor | None]:
    """Return quality score multipliers for AP ranking."""

    if quality_logits is None:
        return {alpha: None for alpha in QUALITY_SCORE_ALPHAS}
    quality = quality_logits.sigmoid().clamp(0.0, 1.0)
    return {alpha: quality.pow(alpha) for alpha in QUALITY_SCORE_ALPHAS}


def fixed_quality_score_multiplier(
    quality_logits: Tensor | None,
    alpha: float,
    temperature: float,
) -> Tensor | None:
    """Return pre-registered temperature-scaled quality score multipliers."""

    if quality_logits is None:
        return None
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    quality = (quality_logits / temperature).sigmoid().clamp(0.0, 1.0)
    return quality.pow(alpha)


def center_distance_score_multiplier(pred_boxes: Tensor, power: float = 1.0) -> Tensor:
    """Return an eval-only multiplier that downranks center-near predicted boxes."""

    if power < 0:
        raise ValueError("power must be >= 0")
    center = pred_boxes.new_tensor([0.5, 0.5])
    max_distance = float(0.5 * (2.0**0.5))
    distance = (pred_boxes[:, :2] - center).norm(dim=-1) / max_distance
    return distance.clamp(0.0, 1.0).pow(power)


def box_slice_masks(boxes: Tensor) -> dict[str, Tensor]:
    """Return object-size and center-position masks for normalized cxcywh boxes."""

    if boxes.numel() == 0:
        empty = torch.zeros(0, dtype=torch.bool, device=boxes.device)
        return {
            "small": empty,
            "medium": empty,
            "large": empty,
            "center": empty,
            "offcenter": empty,
        }
    areas = boxes[:, 2] * boxes[:, 3]
    center_distance = ((boxes[:, 0] - 0.5).square() + (boxes[:, 1] - 0.5).square()).sqrt()
    return {
        "small": areas < 0.04,
        "medium": (areas >= 0.04) & (areas < 0.075),
        "large": areas >= 0.075,
        "center": center_distance < 0.25,
        "offcenter": center_distance >= 0.25,
    }


def summarize_slice_ap(values: dict[str, list[Tensor]]) -> dict[str, float]:
    """Average per-image AP values for every robustness slice."""

    return {
        name: float(torch.stack(slice_values).mean().item()) if slice_values else 0.0
        for name, slice_values in values.items()
    }


def summarize_slice_ranking_diagnostics(
    vector_values: dict[str, dict[str, list[Tensor]]],
    scalar_values: dict[str, dict[str, list[Tensor]]],
) -> dict[str, float]:
    """Average query-ranking diagnostics for center/offcenter target subsets."""

    summary: dict[str, float] = {}
    for slice_name, metric_values in vector_values.items():
        for metric_name, values in metric_values.items():
            key = f"{slice_name}_{metric_name}"
            non_empty = [value for value in values if value.numel() > 0]
            if not non_empty:
                summary[key] = 0.0
                continue
            summary[key] = float(torch.cat(non_empty).float().mean().item())
    for slice_name, metric_values in scalar_values.items():
        for metric_name, values in metric_values.items():
            key = f"{slice_name}_{metric_name}"
            summary[key] = float(torch.stack(values).mean().item()) if values else 0.0
    return summary


def sample_matches_slice(sample: RealDetSample, max_objects: int, slice_name: str) -> bool:
    """Return whether ranked target boxes include the requested robustness slice."""

    if slice_name == "none":
        return True
    ranked_boxes = sorted(sample.boxes, key=lambda box: box_area(box), reverse=True)[:max_objects]
    normalized = torch.tensor(
        [normalize_box(box, sample.width, sample.height) for box in ranked_boxes],
        dtype=torch.float32,
    )
    if normalized.numel() == 0:
        return False
    masks = box_slice_masks(normalized)
    if slice_name not in masks:
        raise ValueError(f"unknown eval slice filter: {slice_name}")
    return bool(masks[slice_name].any())


def ranking_gap_closure(base_score: float, quality_score: float, oracle_score: float) -> float:
    """Return raw, unclamped fraction of the IoU-reference ranking gap closed."""

    gap = oracle_score - base_score
    if gap <= 1e-8:
        return 0.0
    return (quality_score - base_score) / gap


def oracle_iou_score_multiplier(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    """Return each query's true max IoU to any target for oracle ranking."""

    if target_boxes.numel() == 0:
        return pred_boxes.new_zeros(pred_boxes.shape[0])
    return box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(target_boxes),
    ).max(dim=1).values.clamp(0.0, 1.0)


def objectness_ap50_for_image(
    pred_logits: Tensor,
    pred_boxes: Tensor,
    target_boxes: Tensor,
    score_multiplier: Tensor | None = None,
    iou_threshold: float = 0.5,
) -> Tensor:
    if target_boxes.numel() == 0:
        return pred_logits.new_tensor(0.0)
    scores = pred_logits.softmax(dim=-1)[:, :-1].max(dim=-1).values
    if score_multiplier is not None:
        scores = scores * score_multiplier
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
        if float(best_iou.item()) >= iou_threshold and best_target not in matched_targets:
            matched_targets.add(best_target)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)
    tp = torch.tensor(true_positives, device=pred_logits.device, dtype=pred_logits.dtype).cumsum(dim=0)
    fp = torch.tensor(false_positives, device=pred_logits.device, dtype=pred_logits.dtype).cumsum(dim=0)
    recall = tp / max(int(target_boxes.shape[0]), 1)
    precision = tp / (tp + fp).clamp_min(1e-8)
    return precision_recall_ap(recall, precision)


def query_ranking_diagnostics(
    pred_logits: Tensor,
    pred_boxes: Tensor,
    target_boxes: Tensor,
    target_labels: Tensor,
    quality_logits: Tensor | None = None,
    combined_quality_scores: Tensor | None = None,
    all_target_boxes: Tensor | None = None,
) -> dict[str, Tensor]:
    objectness = pred_logits.softmax(dim=-1)[:, :-1].max(dim=-1).values
    quality_scores = quality_logits.sigmoid() if quality_logits is not None else None
    if combined_quality_scores is not None:
        combined_scores = objectness * combined_quality_scores
    elif quality_scores is not None:
        combined_scores = objectness * quality_scores
    else:
        combined_scores = objectness
    pred_labels = pred_logits.softmax(dim=-1)[:, :-1].argmax(dim=-1)
    target_count = int(target_boxes.shape[0])
    if target_count == 0:
        return {
            "matched_assignment_class_correct": pred_logits.new_zeros((0,)),
            "tp50_class_correct": pred_logits.new_zeros((0,)),
            "score_iou_corr": pred_logits.new_tensor(0.0),
            "objectness_auc": pred_logits.new_tensor(0.5),
            "objectness_ece50": calibration_ece(objectness, torch.zeros_like(objectness, dtype=torch.bool)),
            "objectness_ece75": calibration_ece(objectness, torch.zeros_like(objectness, dtype=torch.bool)),
            "quality_iou_corr": pred_logits.new_tensor(0.0),
            "quality_auc": pred_logits.new_tensor(0.5),
            "combined_iou_corr": pred_logits.new_tensor(0.0),
            "combined_auc": pred_logits.new_tensor(0.5),
            "combined_ece50": calibration_ece(combined_scores, torch.zeros_like(combined_scores, dtype=torch.bool)),
            "combined_ece75": calibration_ece(combined_scores, torch.zeros_like(combined_scores, dtype=torch.bool)),
            "topk_fp_rate": pred_logits.new_tensor(0.0),
            "combined_topk_fp_rate": pred_logits.new_tensor(0.0),
            "topk_slice_match_rate": pred_logits.new_tensor(0.0),
            "topk_non_slice_match_rate": pred_logits.new_tensor(0.0),
            "topk_no_gt_match_rate": pred_logits.new_tensor(0.0),
            "combined_topk_slice_match_rate": pred_logits.new_tensor(0.0),
            "combined_topk_non_slice_match_rate": pred_logits.new_tensor(0.0),
            "combined_topk_no_gt_match_rate": pred_logits.new_tensor(0.0),
            "topk_center_distance": pred_logits.new_tensor(0.0),
            "combined_topk_center_distance": pred_logits.new_tensor(0.0),
            "duplicate_per_gt": pred_logits.new_tensor(0.0),
            "matched_query_counts": torch.zeros(
                pred_boxes.shape[0],
                device=pred_boxes.device,
            ),
        }
    iou_matrix = box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(target_boxes),
    )
    max_iou_per_query = iou_matrix.max(dim=1).values
    matched_queries, matched_targets = best_iou_assignment_indices(iou_matrix)
    if matched_queries.numel() == 0:
        class_correct = pred_logits.new_zeros((0,))
    else:
        class_correct = (
            pred_labels[matched_queries] == target_labels[matched_targets]
        ).float()
    tp50_mask = (
        max_iou_per_query[matched_queries] >= 0.5
        if matched_queries.numel() > 0
        else torch.zeros(0, dtype=torch.bool, device=pred_boxes.device)
    )
    tp50_class_correct = class_correct[tp50_mask]
    topk = objectness.argsort(descending=True)[:target_count]
    topk_fp_rate = (max_iou_per_query[topk] < 0.5).float().mean()
    combined_topk = combined_scores.argsort(descending=True)[:target_count]
    combined_topk_fp_rate = (max_iou_per_query[combined_topk] < 0.5).float().mean()
    topk_match_rates = topk_target_match_rates(
        pred_boxes=pred_boxes,
        slice_target_boxes=target_boxes,
        all_target_boxes=all_target_boxes,
        topk=topk,
    )
    combined_topk_match_rates = topk_target_match_rates(
        pred_boxes=pred_boxes,
        slice_target_boxes=target_boxes,
        all_target_boxes=all_target_boxes,
        topk=combined_topk,
    )
    center_reference = pred_boxes.new_tensor([0.5, 0.5])
    pred_center_distance = (pred_boxes[:, :2] - center_reference).norm(dim=-1)
    topk_center_distance = pred_center_distance[topk].mean()
    combined_topk_center_distance = pred_center_distance[combined_topk].mean()
    duplicate_per_gt = duplicate_predictions_per_gt(iou_matrix, threshold=0.5)
    positive = max_iou_per_query >= 0.5
    positive75 = max_iou_per_query >= 0.75
    matched_query_counts = torch.zeros(pred_boxes.shape[0], device=pred_boxes.device)
    if matched_queries.numel() > 0:
        matched_query_counts.scatter_add_(
            0,
            matched_queries,
            torch.ones_like(matched_queries, dtype=matched_query_counts.dtype),
        )
    return {
        "matched_assignment_class_correct": class_correct,
        "tp50_class_correct": tp50_class_correct,
        "score_iou_corr": pearson_corr(objectness, max_iou_per_query),
        "objectness_auc": binary_auc(objectness, positive),
        "objectness_ece50": calibration_ece(objectness, positive),
        "objectness_ece75": calibration_ece(objectness, positive75),
        "quality_iou_corr": pearson_corr(quality_scores, max_iou_per_query)
        if quality_scores is not None
        else pred_logits.new_tensor(0.0),
        "quality_auc": binary_auc(quality_scores, positive)
        if quality_scores is not None
        else pred_logits.new_tensor(0.5),
        "combined_iou_corr": pearson_corr(combined_scores, max_iou_per_query),
        "combined_auc": binary_auc(combined_scores, positive),
        "combined_ece50": calibration_ece(combined_scores, positive),
        "combined_ece75": calibration_ece(combined_scores, positive75),
        "topk_fp_rate": topk_fp_rate,
        "combined_topk_fp_rate": combined_topk_fp_rate,
        "topk_slice_match_rate": topk_match_rates["slice"],
        "topk_non_slice_match_rate": topk_match_rates["non_slice"],
        "topk_no_gt_match_rate": topk_match_rates["no_gt"],
        "combined_topk_slice_match_rate": combined_topk_match_rates["slice"],
        "combined_topk_non_slice_match_rate": combined_topk_match_rates["non_slice"],
        "combined_topk_no_gt_match_rate": combined_topk_match_rates["no_gt"],
        "topk_center_distance": topk_center_distance,
        "combined_topk_center_distance": combined_topk_center_distance,
        "duplicate_per_gt": duplicate_per_gt,
        "matched_query_counts": matched_query_counts,
    }


def topk_target_match_rates(
    pred_boxes: Tensor,
    slice_target_boxes: Tensor,
    all_target_boxes: Tensor | None,
    topk: Tensor,
    iou_threshold: float = 0.5,
) -> dict[str, Tensor]:
    """Split top-k predictions by slice-target, non-slice-target, and no-GT matches."""

    if topk.numel() == 0:
        zero = pred_boxes.new_tensor(0.0)
        return {"slice": zero, "non_slice": zero, "no_gt": zero}
    slice_iou = box_iou(
        box_cxcywh_to_xyxy(pred_boxes[topk]),
        box_cxcywh_to_xyxy(slice_target_boxes),
    )
    slice_match = slice_iou.max(dim=1).values >= iou_threshold
    if all_target_boxes is None or all_target_boxes.numel() == 0:
        all_match = slice_match
    else:
        all_iou = box_iou(
            box_cxcywh_to_xyxy(pred_boxes[topk]),
            box_cxcywh_to_xyxy(all_target_boxes),
        )
        all_match = all_iou.max(dim=1).values >= iou_threshold
    non_slice_match = all_match & ~slice_match
    no_gt_match = ~all_match
    return {
        "slice": slice_match.float().mean(),
        "non_slice": non_slice_match.float().mean(),
        "no_gt": no_gt_match.float().mean(),
    }


def best_iou_assignment_indices(iou_matrix: Tensor) -> tuple[Tensor, Tensor]:
    query_count, target_count = iou_matrix.shape
    if target_count == 0:
        empty = torch.empty(0, dtype=torch.long, device=iou_matrix.device)
        return empty, empty
    if target_count > query_count:
        raise ValueError("target_count must be <= query_count for diagnostic assignment")
    if query_count > 8 or target_count > 4:
        return greedy_iou_assignment_indices(iou_matrix)
    target_indices = torch.arange(target_count, device=iou_matrix.device)
    best_score = None
    best_queries = None
    for query_indices_tuple in permutations(range(query_count), target_count):
        query_indices = torch.tensor(query_indices_tuple, device=iou_matrix.device)
        score = iou_matrix[query_indices, target_indices].sum()
        if best_score is None or bool(score > best_score):
            best_score = score
            best_queries = query_indices
    if best_queries is None:
        empty = torch.empty(0, dtype=torch.long, device=iou_matrix.device)
        return empty, empty
    return best_queries, target_indices


def greedy_iou_assignment_indices(iou_matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Approximate smoke-test assignment for larger query/target counts."""
    query_count, target_count = iou_matrix.shape
    pairs: list[tuple[float, int, int]] = []
    for query_idx in range(query_count):
        for target_idx in range(target_count):
            pairs.append(
                (
                    float(iou_matrix[query_idx, target_idx].item()),
                    query_idx,
                    target_idx,
                )
            )
    used_queries: set[int] = set()
    used_targets: set[int] = set()
    selected_queries: list[int] = []
    selected_targets: list[int] = []
    for _, query_idx, target_idx in sorted(pairs, reverse=True):
        if query_idx in used_queries or target_idx in used_targets:
            continue
        used_queries.add(query_idx)
        used_targets.add(target_idx)
        selected_queries.append(query_idx)
        selected_targets.append(target_idx)
        if len(selected_targets) == target_count:
            break
    return (
        torch.tensor(selected_queries, dtype=torch.long, device=iou_matrix.device),
        torch.tensor(selected_targets, dtype=torch.long, device=iou_matrix.device),
    )


def duplicate_predictions_per_gt(iou_matrix: Tensor, threshold: float) -> Tensor:
    if iou_matrix.shape[1] == 0:
        return iou_matrix.new_tensor(0.0)
    duplicates = (iou_matrix >= threshold).float().sum(dim=0).sub(1.0).clamp_min(0.0)
    return duplicates.mean()


def pearson_corr(x: Tensor, y: Tensor) -> Tensor:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom.item()) <= 1e-12:
        return x.new_tensor(0.0)
    return (x_centered * y_centered).sum() / denom


def binary_auc(scores: Tensor, positive: Tensor) -> Tensor:
    positive_scores = scores[positive]
    negative_scores = scores[~positive]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return scores.new_tensor(0.5)
    greater = positive_scores[:, None] > negative_scores[None, :]
    equal = positive_scores[:, None] == negative_scores[None, :]
    return greater.float().mean() + 0.5 * equal.float().mean()


def calibration_ece(scores: Tensor, positive: Tensor, bins: int = 10) -> Tensor:
    """Expected calibration error for query-level detection confidence."""

    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    clipped = scores.clamp(0.0, 1.0)
    positive_float = positive.to(device=scores.device, dtype=scores.dtype)
    ece = scores.new_tensor(0.0)
    for bin_idx in range(bins):
        lower = bin_idx / bins
        upper = (bin_idx + 1) / bins
        if bin_idx == bins - 1:
            in_bin = (clipped >= lower) & (clipped <= upper)
        else:
            in_bin = (clipped >= lower) & (clipped < upper)
        if not bool(in_bin.any()):
            continue
        confidence = clipped[in_bin].mean()
        accuracy = positive_float[in_bin].mean()
        ece = ece + in_bin.float().mean() * (confidence - accuracy).abs()
    return ece


def assignment_entropy(counts: Tensor) -> float:
    total = counts.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    probs = counts / total
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    denom = torch.log(torch.tensor(float(counts.numel()), dtype=counts.dtype))
    return float((entropy / denom.clamp_min(1e-12)).item())


def class_aware_ap50_for_image(
    pred_logits: Tensor,
    pred_boxes: Tensor,
    target_boxes: Tensor,
    target_labels: Tensor,
    score_multiplier: Tensor | None = None,
    iou_threshold: float = 0.5,
) -> Tensor:
    if target_boxes.numel() == 0:
        return pred_logits.new_tensor(0.0)
    class_probs = pred_logits.softmax(dim=-1)[:, :-1]
    scores, pred_labels = class_probs.max(dim=-1)
    if score_multiplier is not None:
        scores = scores * score_multiplier
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
        same_class = target_labels == pred_labels[query_idx]
        if bool(same_class.any()):
            masked_ious = iou_matrix[query_idx].masked_fill(~same_class, -1.0)
            best_iou, best_target_tensor = masked_ious.max(dim=0)
            best_target = int(best_target_tensor.item())
        else:
            best_iou = pred_logits.new_tensor(-1.0)
            best_target = -1
        if float(best_iou.item()) >= iou_threshold and best_target not in matched_targets:
            matched_targets.add(best_target)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)
    tp = torch.tensor(true_positives, device=pred_logits.device, dtype=pred_logits.dtype).cumsum(dim=0)
    fp = torch.tensor(false_positives, device=pred_logits.device, dtype=pred_logits.dtype).cumsum(dim=0)
    recall = tp / max(int(target_boxes.shape[0]), 1)
    precision = tp / (tp + fp).clamp_min(1e-8)
    return precision_recall_ap(recall, precision)


def load_real_det_samples(anno_root: Path, image_root: Path) -> list[RealDetSample]:
    samples = []
    for xml_path in sorted(anno_root.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        image_id = required_text(root, "filename")
        image_path = image_root / f"{image_id}.JPEG"
        if not image_path.exists():
            continue
        size = root.find("size")
        if size is None:
            continue
        width = int(required_text(size, "width"))
        height = int(required_text(size, "height"))
        boxes = []
        for obj in root.findall("object"):
            label = required_text(obj, "name")
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = clamp_int(required_text(bbox, "xmin"), 0, width)
            ymin = clamp_int(required_text(bbox, "ymin"), 0, height)
            xmax = clamp_int(required_text(bbox, "xmax"), 0, width)
            ymax = clamp_int(required_text(bbox, "ymax"), 0, height)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append(RealBox(label=label, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
        if boxes:
            samples.append(RealDetSample(image_id=image_id, image_path=image_path, width=width, height=height, boxes=tuple(boxes)))
    if not samples:
        raise ValueError(f"no DET samples found under {anno_root}")
    return samples


def build_label_map(samples: list[RealDetSample], top_classes: int) -> dict[str, int]:
    counts = Counter(box.label for sample in samples for box in sample.boxes)
    if top_classes <= 0:
        labels = sorted(counts)
    else:
        labels = [label for label, _ in counts.most_common(top_classes)]
    return {label: idx for idx, label in enumerate(labels)}


def filter_samples(
    samples: list[RealDetSample],
    selected_labels: set[str],
    max_samples: int,
) -> list[RealDetSample]:
    filtered = []
    for sample in samples:
        boxes = tuple(box for box in sample.boxes if box.label in selected_labels)
        if not boxes:
            continue
        filtered.append(
            RealDetSample(
                image_id=sample.image_id,
                image_path=sample.image_path,
                width=sample.width,
                height=sample.height,
                boxes=boxes,
            )
        )
        if max_samples > 0 and len(filtered) >= max_samples:
            break
    return filtered


def build_splits(
    samples: list[RealDetSample],
    label_to_id: dict[str, int],
    image_size: int,
    max_objects: int,
    train_frac: float,
    calibration_frac: float,
    calibration_source: str = "heldout",
    calibration_slice_filter: str = "none",
    eval_slice_filter: str = "none",
    seed: int = 0,
) -> tuple[Subset, Subset | None, Subset, list[dict[str, int | str]]]:
    if calibration_source not in {"heldout", "train"}:
        raise ValueError("calibration_source must be 'heldout' or 'train'")
    if calibration_slice_filter not in {"none", "small", "medium", "large", "center", "offcenter"}:
        raise ValueError("unknown calibration slice filter")
    if eval_slice_filter not in {"none", "small", "medium", "large", "center", "offcenter"}:
        raise ValueError("unknown eval slice filter")
    dataset = RealDetDataset(samples, label_to_id, image_size=image_size, max_objects=max_objects)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    train_size = max(1, min(len(indices) - 1, int(len(indices) * train_frac)))
    train_indices = indices[:train_size]
    heldout_indices = indices[train_size:]
    calibration_indices: list[int] = []
    eval_indices = heldout_indices
    if calibration_frac > 0.0 and calibration_source == "heldout" and len(heldout_indices) > 1:
        if calibration_slice_filter == "none":
            calibration_size = max(1, min(len(heldout_indices) - 1, int(len(heldout_indices) * calibration_frac)))
            calibration_indices = heldout_indices[:calibration_size]
        else:
            calibration_candidates = [
                idx
                for idx in heldout_indices
                if sample_matches_slice(samples[idx], max_objects=max_objects, slice_name=calibration_slice_filter)
            ]
            if not calibration_candidates:
                raise ValueError(
                    f"calibration slice filter produced an empty calibration split: {calibration_slice_filter}"
                )
            max_calibration = min(len(calibration_candidates), len(heldout_indices) - 1)
            calibration_size = max(1, min(max_calibration, int(max_calibration * calibration_frac)))
            calibration_indices = calibration_candidates[:calibration_size]
        calibration_index_set = set(calibration_indices)
        eval_indices = [idx for idx in heldout_indices if idx not in calibration_index_set]
    elif calibration_frac > 0.0 and calibration_source == "train" and len(train_indices) > 1:
        if calibration_slice_filter == "none":
            calibration_size = max(1, min(len(train_indices) - 1, int(len(train_indices) * calibration_frac)))
            calibration_indices = train_indices[-calibration_size:]
        else:
            calibration_candidates = [
                idx
                for idx in train_indices
                if sample_matches_slice(samples[idx], max_objects=max_objects, slice_name=calibration_slice_filter)
            ]
            if not calibration_candidates:
                raise ValueError(
                    f"calibration slice filter produced an empty calibration split: {calibration_slice_filter}"
                )
            max_calibration = min(len(calibration_candidates), len(train_indices) - 1)
            calibration_size = max(1, min(max_calibration, int(max_calibration * calibration_frac)))
            calibration_indices = calibration_candidates[-calibration_size:]
        calibration_index_set = set(calibration_indices)
        train_indices = [idx for idx in train_indices if idx not in calibration_index_set]
    if eval_slice_filter != "none":
        eval_indices = [
            idx
            for idx in eval_indices
            if sample_matches_slice(samples[idx], max_objects=max_objects, slice_name=eval_slice_filter)
        ]
        if not eval_indices:
            raise ValueError(f"eval slice filter produced an empty eval split: {eval_slice_filter}")
    split_rows = []
    split_defs = [("train", train_indices)]
    if calibration_indices:
        split_defs.append(("calibration", calibration_indices))
    split_defs.append(("eval", eval_indices))
    for split, split_indices in split_defs:
        for idx in split_indices:
            split_rows.append(
                {
                    "run_seed": seed,
                    "split": split,
                    "index": idx,
                    "image_id": samples[idx].image_id,
                    "object_count": len(samples[idx].boxes),
                }
            )
    calibration_subset = Subset(dataset, calibration_indices) if calibration_indices else None
    return Subset(dataset, train_indices), calibration_subset, Subset(dataset, eval_indices), split_rows


def det_collate(batch: list[tuple[Tensor, dict[str, Tensor]]]) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def targets_to_device(targets: list[dict[str, Tensor]], device: torch.device) -> list[dict[str, Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def normalize_box(box: RealBox, width: int, height: int) -> list[float]:
    xmin = box.xmin / width
    xmax = box.xmax / width
    ymin = box.ymin / height
    ymax = box.ymax / height
    return [
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        max(1e-6, xmax - xmin),
        max(1e-6, ymax - ymin),
    ]


def box_area(box: RealBox) -> int:
    return max(0, box.xmax - box.xmin) * max(0, box.ymax - box.ymin)


def required_text(root: ET.Element, tag: str) -> str:
    child = root.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"missing XML tag: {tag}")
    return child.text.strip()


def clamp_int(text: str, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(text.strip())))


def write_label_map(path: Path, label_to_id: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "id"])
        for label, idx in sorted(label_to_id.items(), key=lambda item: item[1]):
            writer.writerow([label, idx])


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "run_seed",
        "step",
        "loss",
        "loss_score_iou",
        "loss_quality_cls",
        "loss_quality_head",
        "quality_head_scale",
        "eval_iou",
        "eval_recall50",
        "eval_ap50",
        "eval_ap50_class",
        "eval_ap75",
        "eval_ap75_q_fixed",
        "eval_ap75_oracle_iou",
        "eval_small_ap50",
        "eval_medium_ap50",
        "eval_large_ap50",
        "eval_center_ap50",
        "eval_offcenter_ap50",
        "eval_small_ap50_q_fixed",
        "eval_medium_ap50_q_fixed",
        "eval_large_ap50_q_fixed",
        "eval_center_ap50_q_fixed",
        "eval_offcenter_ap50_q_fixed",
        "eval_ap50_center_dist1",
        "eval_ap50_q_fixed_center_dist1",
        "eval_center_ap50_center_dist1",
        "eval_offcenter_ap50_center_dist1",
        "eval_center_ap50_q_fixed_center_dist1",
        "eval_offcenter_ap50_q_fixed_center_dist1",
        "eval_ap50_q025",
        "eval_ap50_q05",
        "eval_ap50_q1",
        "eval_ap50_q2",
        "eval_ap50_q4",
        "eval_ap50_q_fixed",
        "eval_ap50_q_fixed_alpha",
        "eval_ap50_q_fixed_temperature",
        "eval_ap50_q_best",
        "eval_ap50_q_best_alpha",
        "eval_ap50_class_q025",
        "eval_ap50_class_q05",
        "eval_ap50_class_q1",
        "eval_ap50_class_q2",
        "eval_ap50_class_q4",
        "eval_ap50_class_q_fixed",
        "eval_ap50_class_q_best",
        "eval_ap50_class_q_best_alpha",
        "eval_ap50_oracle_iou",
        "eval_ap50_class_oracle_iou",
        "eval_ap50_oracle_gap",
        "eval_ap50_q_fixed_gap_to_oracle",
        "eval_ap50_q_fixed_oracle_closure",
        "eval_ap50_q_best_gap_to_oracle",
        "eval_ap50_q_best_oracle_closure",
        "eval_ap50_class_oracle_gap",
        "eval_ap50_class_q_fixed_gap_to_oracle",
        "eval_ap50_class_q_fixed_oracle_closure",
        "eval_ap50_class_q_best_gap_to_oracle",
        "eval_ap50_class_q_best_oracle_closure",
        "eval_mask_iou",
        "eval_mask_dice",
        "eval_small_iou",
        "eval_medium_iou",
        "eval_large_iou",
        "eval_center_iou",
        "eval_offcenter_iou",
        "matched_assignment_class_acc",
        "tp50_class_acc",
        "score_iou_corr",
        "objectness_auc",
        "objectness_ece50",
        "objectness_ece75",
        "quality_iou_corr",
        "quality_auc",
        "combined_iou_corr",
        "combined_auc",
        "combined_ece50",
        "combined_ece75",
        "topk_fp_rate",
        "combined_topk_fp_rate",
        "topk_center_distance",
        "combined_topk_center_distance",
        "duplicate_per_gt",
        "center_matched_assignment_class_acc",
        "offcenter_matched_assignment_class_acc",
        "center_tp50_class_acc",
        "offcenter_tp50_class_acc",
        "center_score_iou_corr",
        "offcenter_score_iou_corr",
        "center_objectness_auc",
        "offcenter_objectness_auc",
        "center_topk_fp_rate",
        "offcenter_topk_fp_rate",
        "center_combined_topk_fp_rate",
        "offcenter_combined_topk_fp_rate",
        "center_topk_slice_match_rate",
        "offcenter_topk_slice_match_rate",
        "center_topk_non_slice_match_rate",
        "offcenter_topk_non_slice_match_rate",
        "center_topk_no_gt_match_rate",
        "offcenter_topk_no_gt_match_rate",
        "center_combined_topk_slice_match_rate",
        "offcenter_combined_topk_slice_match_rate",
        "center_combined_topk_non_slice_match_rate",
        "offcenter_combined_topk_non_slice_match_rate",
        "center_combined_topk_no_gt_match_rate",
        "offcenter_combined_topk_no_gt_match_rate",
        "center_topk_center_distance",
        "offcenter_topk_center_distance",
        "center_combined_topk_center_distance",
        "offcenter_combined_topk_center_distance",
        "center_duplicate_per_gt",
        "offcenter_duplicate_per_gt",
        "query_assignment_entropy",
        "best_iou",
        "best_step",
        "images_per_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_split_manifest(path: Path, rows: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_seed", "split", "index", "image_id", "object_count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    final_rows = final_rows_by_model_seed(rows)
    print(
        "summary_model,final_iou_mean,best_iou_mean,recall50_mean,ap50_mean,ap50_class_mean,"
        "ap75_mean,ap75_q_fixed_mean,ap75_iou_reference_mean,"
        "small_ap50_mean,medium_ap50_mean,large_ap50_mean,center_ap50_mean,offcenter_ap50_mean,"
        "small_ap50_q_fixed_mean,medium_ap50_q_fixed_mean,large_ap50_q_fixed_mean,"
        "center_ap50_q_fixed_mean,offcenter_ap50_q_fixed_mean,"
        "ap50_q025_mean,ap50_q05_mean,ap50_q1_mean,ap50_q2_mean,ap50_q4_mean,"
        "ap50_q_fixed_mean,ap50_q_fixed_alpha_mean,ap50_q_fixed_temperature_mean,"
        "ap50_q_best_mean,ap50_q_selected_alpha_mean,"
        "ap50_class_q025_mean,ap50_class_q05_mean,ap50_class_q1_mean,ap50_class_q2_mean,"
        "ap50_class_q4_mean,ap50_class_q_fixed_mean,ap50_class_q_best_mean,"
        "ap50_class_q_selected_alpha_mean,"
        "ap50_iou_reference_mean,ap50_class_iou_reference_mean,ap50_iou_reference_gap_mean,"
        "ap50_q_fixed_gap_to_iou_reference_mean,ap50_q_fixed_raw_iou_reference_closure_mean,"
        "ap50_q_best_gap_to_iou_reference_mean,ap50_q_best_raw_iou_reference_closure_mean,"
        "ap50_class_iou_reference_gap_mean,ap50_class_q_fixed_gap_to_iou_reference_mean,"
        "ap50_class_q_fixed_raw_iou_reference_closure_mean,"
        "ap50_class_q_best_gap_to_iou_reference_mean,"
        "ap50_class_q_best_raw_iou_reference_closure_mean,"
        "matched_assignment_class_acc_mean,tp50_class_acc_mean,"
        "score_iou_corr_mean,objectness_auc_mean,objectness_ece50_mean,objectness_ece75_mean,"
        "quality_iou_corr_mean,quality_auc_mean,"
        "combined_iou_corr_mean,combined_auc_mean,combined_ece50_mean,combined_ece75_mean,"
        "topk_fp_rate_mean,"
        "duplicate_per_gt_mean,query_assignment_entropy_mean,"
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
            f"{mean([float(row['eval_ap50_class']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap75']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap75_q_fixed']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap75_oracle_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_small_ap50']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_medium_ap50']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_large_ap50']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_center_ap50']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_offcenter_ap50']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_small_ap50_q_fixed']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_medium_ap50_q_fixed']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_large_ap50_q_fixed']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_center_ap50_q_fixed']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_offcenter_ap50_q_fixed']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q025']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q05']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q1']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q2']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q4']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q_fixed']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q_fixed_alpha']) for row in model_rows]):.2f},"
            f"{mean([float(row['eval_ap50_q_fixed_temperature']) for row in model_rows]):.2f},"
            f"{mean([float(row['eval_ap50_q_best']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q_best_alpha']) for row in model_rows]):.2f},"
            f"{mean([float(row['eval_ap50_class_q025']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q05']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q1']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q2']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q4']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q_fixed']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q_best']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q_best_alpha']) for row in model_rows]):.2f},"
            f"{mean([float(row['eval_ap50_oracle_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_oracle_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_oracle_gap']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q_fixed_gap_to_oracle']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q_fixed_oracle_closure']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q_best_gap_to_oracle']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_q_best_oracle_closure']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_oracle_gap']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q_fixed_gap_to_oracle']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q_fixed_oracle_closure']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q_best_gap_to_oracle']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_ap50_class_q_best_oracle_closure']) for row in model_rows]):.3f},"
            f"{mean([float(row['matched_assignment_class_acc']) for row in model_rows]):.3f},"
            f"{mean([float(row['tp50_class_acc']) for row in model_rows]):.3f},"
            f"{mean([float(row['score_iou_corr']) for row in model_rows]):.3f},"
            f"{mean([float(row['objectness_auc']) for row in model_rows]):.3f},"
            f"{mean([float(row['objectness_ece50']) for row in model_rows]):.3f},"
            f"{mean([float(row['objectness_ece75']) for row in model_rows]):.3f},"
            f"{mean([float(row['quality_iou_corr']) for row in model_rows]):.3f},"
            f"{mean([float(row['quality_auc']) for row in model_rows]):.3f},"
            f"{mean([float(row['combined_iou_corr']) for row in model_rows]):.3f},"
            f"{mean([float(row['combined_auc']) for row in model_rows]):.3f},"
            f"{mean([float(row['combined_ece50']) for row in model_rows]):.3f},"
            f"{mean([float(row['combined_ece75']) for row in model_rows]):.3f},"
            f"{mean([float(row['topk_fp_rate']) for row in model_rows]):.3f},"
            f"{mean([float(row['duplicate_per_gt']) for row in model_rows]):.3f},"
            f"{mean([float(row['query_assignment_entropy']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_mask_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_mask_dice']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_small_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_medium_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_large_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_center_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_offcenter_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['images_per_sec']) for row in model_rows]):.2f}"
        )


def print_proposal_oracle_gap_summary(rows: list[dict[str, float | int | str]]) -> None:
    """Print predicted-vs-oracle proposal gap and closure diagnostics."""

    final_rows = final_rows_by_model_seed(rows)
    row_models = {str(row["model"]) for row in final_rows}
    baseline_pred = "local_mask_proposal_nms_query"
    baseline_oracle = "local_mask_proposal_oracle_nms_query"
    if baseline_pred not in row_models or baseline_oracle not in row_models:
        return
    pairs = [
        (
            "init_only",
            "local_mask_proposal_nms_query",
            "local_mask_proposal_oracle_nms_query",
        ),
        (
            "decode2",
            "local_mask_proposal_nms_query_decode2",
            "local_mask_proposal_oracle_nms_query_decode2",
        ),
        (
            "reinject",
            "local_mask_proposal_nms_query_reinject",
            "local_mask_proposal_oracle_nms_query_reinject",
        ),
        (
            "persistent",
            "local_mask_proposal_nms_query_persistent",
            "local_mask_proposal_oracle_nms_query_persistent",
        ),
    ]
    metrics = [
        ("final_iou", "eval_iou"),
        ("best_iou", "best_iou"),
        ("ap50", "eval_ap50"),
        ("class_ap50", "eval_ap50_class"),
    ]
    print(
        "proposal_oracle_gap_mode,metric,pred_model,oracle_model,"
        "pred_mean,oracle_mean,direct_gap_mean,closure_vs_init_oracle_gap_mean"
    )
    run_seeds = sorted({int(row["run_seed"]) for row in final_rows})
    for mode, pred_model, oracle_model in pairs:
        if pred_model not in row_models or oracle_model not in row_models:
            continue
        for metric_name, field in metrics:
            pred_values = []
            oracle_values = []
            gap_values = []
            closure_values = []
            for run_seed in run_seeds:
                pred = find_row(final_rows, model=pred_model, run_seed=run_seed)
                oracle = find_row(final_rows, model=oracle_model, run_seed=run_seed)
                base_pred = find_row(final_rows, model=baseline_pred, run_seed=run_seed)
                base_oracle = find_row(final_rows, model=baseline_oracle, run_seed=run_seed)
                if pred is None or oracle is None or base_pred is None or base_oracle is None:
                    continue
                pred_value = float(pred[field])
                oracle_value = float(oracle[field])
                base_pred_value = float(base_pred[field])
                base_oracle_value = float(base_oracle[field])
                pred_values.append(pred_value)
                oracle_values.append(oracle_value)
                gap_values.append(oracle_value - pred_value)
                closure_values.append(
                    proposal_gap_closure(
                        base_pred=base_pred_value,
                        candidate=pred_value,
                        oracle=base_oracle_value,
                    )
                )
            if pred_values:
                print(
                    f"{mode},{metric_name},{pred_model},{oracle_model},"
                    f"{mean(pred_values):.3f},{mean(oracle_values):.3f},"
                    f"{mean(gap_values):.3f},{mean(closure_values):.3f}"
                )


def proposal_gap_closure(base_pred: float, candidate: float, oracle: float) -> float:
    """Return raw fraction of the init-only predicted-vs-oracle gap closed."""

    gap = oracle - base_pred
    if gap <= 1e-8:
        return 0.0
    return (candidate - base_pred) / gap


def print_paired_summary(rows: list[dict[str, float | int | str]], reference_model: str) -> None:
    final_rows = final_rows_by_model_seed(rows)
    if reference_model not in {str(row["model"]) for row in final_rows}:
        return
    print(f"paired_det_real_vs,{reference_model}")
    print(
        "paired_model,final_iou_delta_mean,final_iou_wins,best_iou_delta_mean,best_iou_wins,"
        "ap50_delta_mean,ap50_wins,ap50_class_delta_mean,ap50_class_wins,"
        "ap75_delta_mean,ap75_wins,ap75_q_fixed_delta_mean,ap75_q_fixed_wins,"
        "ap75_oracle_iou_delta_mean,ap75_oracle_iou_wins,"
        "ap50_q1_delta_mean,ap50_q1_wins,ap50_q_fixed_delta_mean,ap50_q_fixed_wins,"
        "ap50_q_best_delta_mean,ap50_q_best_wins,"
        "ap50_class_q1_delta_mean,ap50_class_q1_wins,"
        "ap50_class_q_fixed_delta_mean,ap50_class_q_fixed_wins,"
        "ap50_class_q_best_delta_mean,ap50_class_q_best_wins,"
        "ap50_oracle_iou_delta_mean,ap50_oracle_iou_wins,"
        "ap50_class_oracle_iou_delta_mean,ap50_class_oracle_iou_wins,"
        "ap50_q_best_closure_delta_mean,ap50_class_q_best_closure_delta_mean"
    )
    run_seeds = sorted({int(row["run_seed"]) for row in final_rows})
    for model in sorted({str(row["model"]) for row in final_rows}):
        if model == reference_model:
            continue
        final_deltas = []
        best_deltas = []
        ap_deltas = []
        ap_class_deltas = []
        ap75_deltas = []
        ap75_q_fixed_deltas = []
        ap75_oracle_deltas = []
        ap_q1_deltas = []
        ap_q_fixed_deltas = []
        ap_q_best_deltas = []
        ap_class_q1_deltas = []
        ap_class_q_fixed_deltas = []
        ap_class_q_best_deltas = []
        ap_oracle_deltas = []
        ap_class_oracle_deltas = []
        closure_deltas = []
        class_closure_deltas = []
        for run_seed in run_seeds:
            ref = find_row(final_rows, model=reference_model, run_seed=run_seed)
            cur = find_row(final_rows, model=model, run_seed=run_seed)
            if ref is None or cur is None:
                continue
            final_deltas.append(float(cur["eval_iou"]) - float(ref["eval_iou"]))
            best_deltas.append(float(cur["best_iou"]) - float(ref["best_iou"]))
            ap_deltas.append(float(cur["eval_ap50"]) - float(ref["eval_ap50"]))
            ap_class_deltas.append(float(cur["eval_ap50_class"]) - float(ref["eval_ap50_class"]))
            ap75_deltas.append(float(cur["eval_ap75"]) - float(ref["eval_ap75"]))
            ap75_q_fixed_deltas.append(float(cur["eval_ap75_q_fixed"]) - float(ref["eval_ap75_q_fixed"]))
            ap75_oracle_deltas.append(float(cur["eval_ap75_oracle_iou"]) - float(ref["eval_ap75_oracle_iou"]))
            ap_q1_deltas.append(float(cur["eval_ap50_q1"]) - float(ref["eval_ap50_q1"]))
            ap_q_fixed_deltas.append(float(cur["eval_ap50_q_fixed"]) - float(ref["eval_ap50_q_fixed"]))
            ap_q_best_deltas.append(float(cur["eval_ap50_q_best"]) - float(ref["eval_ap50_q_best"]))
            ap_class_q1_deltas.append(float(cur["eval_ap50_class_q1"]) - float(ref["eval_ap50_class_q1"]))
            ap_class_q_fixed_deltas.append(
                float(cur["eval_ap50_class_q_fixed"]) - float(ref["eval_ap50_class_q_fixed"])
            )
            ap_class_q_best_deltas.append(
                float(cur["eval_ap50_class_q_best"]) - float(ref["eval_ap50_class_q_best"])
            )
            ap_oracle_deltas.append(float(cur["eval_ap50_oracle_iou"]) - float(ref["eval_ap50_oracle_iou"]))
            ap_class_oracle_deltas.append(
                float(cur["eval_ap50_class_oracle_iou"]) - float(ref["eval_ap50_class_oracle_iou"])
            )
            closure_deltas.append(
                float(cur["eval_ap50_q_best_oracle_closure"])
                - float(ref["eval_ap50_q_best_oracle_closure"])
            )
            class_closure_deltas.append(
                float(cur["eval_ap50_class_q_best_oracle_closure"])
                - float(ref["eval_ap50_class_q_best_oracle_closure"])
            )
        if final_deltas:
            print(
                f"{model},"
                f"{mean(final_deltas):.3f},{wins_higher(final_deltas)},"
                f"{mean(best_deltas):.3f},{wins_higher(best_deltas)},"
                f"{mean(ap_deltas):.3f},{wins_higher(ap_deltas)},"
                f"{mean(ap_class_deltas):.3f},{wins_higher(ap_class_deltas)},"
                f"{mean(ap75_deltas):.3f},{wins_higher(ap75_deltas)},"
                f"{mean(ap75_q_fixed_deltas):.3f},{wins_higher(ap75_q_fixed_deltas)},"
                f"{mean(ap75_oracle_deltas):.3f},{wins_higher(ap75_oracle_deltas)},"
                f"{mean(ap_q1_deltas):.3f},{wins_higher(ap_q1_deltas)},"
                f"{mean(ap_q_fixed_deltas):.3f},{wins_higher(ap_q_fixed_deltas)},"
                f"{mean(ap_q_best_deltas):.3f},{wins_higher(ap_q_best_deltas)},"
                f"{mean(ap_class_q1_deltas):.3f},{wins_higher(ap_class_q1_deltas)},"
                f"{mean(ap_class_q_fixed_deltas):.3f},{wins_higher(ap_class_q_fixed_deltas)},"
                f"{mean(ap_class_q_best_deltas):.3f},{wins_higher(ap_class_q_best_deltas)},"
                f"{mean(ap_oracle_deltas):.3f},{wins_higher(ap_oracle_deltas)},"
                f"{mean(ap_class_oracle_deltas):.3f},{wins_higher(ap_class_oracle_deltas)},"
                f"{mean(closure_deltas):.3f},{mean(class_closure_deltas):.3f}"
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
