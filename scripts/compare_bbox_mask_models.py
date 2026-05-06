"""Compare small models on DET-derived bbox-mask heatmap localization."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from attention2d import get_best_device  # noqa: E402
from compare_bbox_center_regression_models import bbox_center_target  # noqa: E402
from compare_bbox_heatmap_models import (  # noqa: E402
    HEATMAP_MODEL_NAMES,
    HeatmapWrapper,
    collect_heatmap_mechanism_stats,
)
from compare_bbox_probe_models import BBoxSample, load_largest_bbox_samples  # noqa: E402
from compare_imagefolder_models import (  # noqa: E402
    MODEL_SEED_OFFSETS,
    build_model,
    count_parameters,
    get_mps_memory_mb,
    split_dataset,
)


@dataclass(frozen=True)
class MaskResultRow:
    dataset: str
    device: str
    run_seed: int
    model: str
    seed: int
    steps: int
    params: int
    train_loss: float
    eval_loss: float
    iou: float
    small_iou: float
    medium_iou: float
    large_iou: float
    dice: float
    cell_balanced_acc: float
    pos_recall: float
    neg_recall: float
    center_mean_l2: float
    center_median_l2: float
    pck_010: float
    pck_020: float
    area_abs_error: float
    mean_prior_loss: float
    mean_prior_iou: float
    center_box_prior_iou: float
    uniform_prior_loss: float
    best_iou: float
    best_step: int
    images_per_sec: float
    mps_current_mem_mb: float
    mps_driver_mem_mb: float
    gate_mean: float
    gate_max_abs: float
    read_norm_mean: float
    state_norm_mean: float
    scaled_read_ratio_mean: float
    block_stats_json: str


class BBoxMaskDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Return image, normalized largest-box center, and a dense bbox mask."""

    def __init__(self, samples: list[BBoxSample], image_size: int, mask_size: int) -> None:
        self.samples = samples
        self.centers = [bbox_center_target(sample) for sample in samples]
        self.masks = [bbox_mask_target(sample, mask_size) for sample in samples]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        center = torch.tensor(self.centers[index], dtype=torch.float32)
        return tensor, center, self.masks[index].clone()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument(
        "--anno-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_bbox_val/ILSVRC2013_DET_bbox_val"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=HEATMAP_MODEL_NAMES,
        default=list(HEATMAP_MODEL_NAMES),
        help="models that expose a 2D memories list or spatial_features",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--mask-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--reference-model", choices=HEATMAP_MODEL_NAMES, default="no_prefill_local_mix")
    parser.add_argument(
        "--reference-models",
        nargs="+",
        choices=HEATMAP_MODEL_NAMES,
        default=None,
        help="optional paired references; defaults to --reference-model",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="optional first-N sample limit for smoke or overfit sanity checks",
    )
    parser.add_argument(
        "--eval-on-train",
        action="store_true",
        help="use the train subset as eval subset for overfit sanity checks",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/bbox_mask_compare.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    samples = load_largest_bbox_samples(args.anno_root, args.image_root)
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    dataset = BBoxMaskDataset(samples=samples, image_size=args.image_size, mask_size=args.mask_size)

    print("device:", device)
    print("task: bbox_mask")
    print("images:", len(dataset))
    print("mask_size:", args.mask_size)
    print("steps:", args.steps)
    print("seeds:", args.seeds)
    print("eval_on_train:", args.eval_on_train)
    print(
        "model,seed,params,train_loss,eval_loss,iou,dice,cell_balanced_acc,"
        "small_iou,medium_iou,large_iou,pos_recall,neg_recall,center_l2,"
        "pck_010,area_abs_error,best_iou,best_step,images_per_sec"
    )

    rows: list[MaskResultRow] = []
    for seed_idx in range(args.seeds):
        split_seed = args.seed + seed_idx
        train_set, eval_set = split_dataset(dataset, args.train_frac, split_seed)
        if args.eval_on_train:
            eval_set = train_set
        eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        eval_centers, eval_masks = targets_for_dataset(eval_set)
        prior = build_mask_priors(eval_masks, mask_size=args.mask_size)

        for model_name in args.models:
            train_loader = build_mask_train_loader(
                train_set=train_set,
                batch_size=args.batch_size,
                seed=10_000_000 + split_seed,
            )
            model_seed = split_seed + MODEL_SEED_OFFSETS[model_name]
            torch.manual_seed(model_seed)
            backbone = build_model(
                name=model_name,
                embed_dim=args.embed_dim,
                image_size=args.image_size,
                num_classes=2,
            )
            model = HeatmapWrapper(backbone=backbone, embed_dim=args.embed_dim, heatmap_size=args.mask_size).to(device)
            params = count_parameters(model)
            train_result = train_mask_model(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                steps=args.steps,
                lr=args.lr,
                eval_every=args.eval_every,
                dice_weight=args.dice_weight,
                threshold=args.threshold,
            )
            eval_metrics = evaluate_mask(
                model=model,
                loader=eval_loader,
                device=device,
                dice_weight=args.dice_weight,
                threshold=args.threshold,
                mask_size=args.mask_size,
            )
            mechanism_stats = collect_heatmap_mechanism_stats(model, eval_loader, device)
            mps_current_mem_mb, mps_driver_mem_mb = get_mps_memory_mb()
            row = MaskResultRow(
                dataset="bbox_mask",
                device=str(device),
                run_seed=split_seed,
                model=model_name,
                seed=model_seed,
                steps=args.steps,
                params=params,
                train_loss=train_result["train_loss"],
                eval_loss=eval_metrics["loss"],
                iou=eval_metrics["iou"],
                small_iou=eval_metrics["small_iou"],
                medium_iou=eval_metrics["medium_iou"],
                large_iou=eval_metrics["large_iou"],
                dice=eval_metrics["dice"],
                cell_balanced_acc=eval_metrics["cell_balanced_acc"],
                pos_recall=eval_metrics["pos_recall"],
                neg_recall=eval_metrics["neg_recall"],
                center_mean_l2=eval_metrics["center_mean_l2"],
                center_median_l2=eval_metrics["center_median_l2"],
                pck_010=eval_metrics["pck_010"],
                pck_020=eval_metrics["pck_020"],
                area_abs_error=eval_metrics["area_abs_error"],
                mean_prior_loss=prior["mean_prior_loss"],
                mean_prior_iou=prior["mean_prior_iou"],
                center_box_prior_iou=prior["center_box_prior_iou"],
                uniform_prior_loss=prior["uniform_prior_loss"],
                best_iou=train_result["best_iou"],
                best_step=train_result["best_step"],
                images_per_sec=train_result["images_per_sec"],
                mps_current_mem_mb=mps_current_mem_mb,
                mps_driver_mem_mb=mps_driver_mem_mb,
                gate_mean=mechanism_stats["gate_mean"],
                gate_max_abs=mechanism_stats["gate_max_abs"],
                read_norm_mean=mechanism_stats["read_norm_mean"],
                state_norm_mean=mechanism_stats["state_norm_mean"],
                scaled_read_ratio_mean=mechanism_stats["scaled_read_ratio_mean"],
                block_stats_json=mechanism_stats["block_stats_json"],
            )
            rows.append(row)
            print(
                f"{model_name},{model_seed},{params},"
                f"{train_result['train_loss']:.4f},{eval_metrics['loss']:.4f},"
                f"{eval_metrics['iou']:.3f},{eval_metrics['dice']:.3f},"
                f"{eval_metrics['cell_balanced_acc']:.3f},"
                f"{eval_metrics['small_iou']:.3f},"
                f"{eval_metrics['medium_iou']:.3f},"
                f"{eval_metrics['large_iou']:.3f},"
                f"{eval_metrics['pos_recall']:.3f},{eval_metrics['neg_recall']:.3f},"
                f"{eval_metrics['center_mean_l2']:.4f},"
                f"{eval_metrics['pck_010']:.3f},"
                f"{eval_metrics['area_abs_error']:.3f},"
                f"{train_result['best_iou']:.3f},{train_result['best_step']},"
                f"{train_result['images_per_sec']:.2f}"
            )

    write_mask_csv(args.out, rows)
    print("saved_csv:", args.out)
    print_mask_summary(rows)
    reference_models = args.reference_models or [args.reference_model]
    for reference_model in dict.fromkeys(reference_models):
        print_mask_paired_summary(rows, reference_model=reference_model)


def build_mask_train_loader(train_set: Dataset[tuple[Tensor, Tensor, Tensor]], batch_size: int, seed: int) -> DataLoader:
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def train_mask_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    eval_every: int,
    dice_weight: float,
    threshold: float,
) -> dict[str, float | int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    loader_iter = cycle(train_loader)
    loss_value = 0.0
    best_iou = -1.0
    best_step = 0
    examples_seen = 0
    start = time.perf_counter()

    for step in range(1, steps + 1):
        images, _, targets = next(loader_iter)
        images = images.to(device)
        targets = targets.to(device)
        examples_seen += int(targets.shape[0])
        logits = model(images)
        loss = mask_loss(logits, targets, dice_weight=dice_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            eval_metrics = evaluate_mask(
                model=model,
                loader=eval_loader,
                device=device,
                dice_weight=dice_weight,
                threshold=threshold,
                mask_size=targets.shape[-1],
            )
            if eval_metrics["iou"] > best_iou:
                best_iou = eval_metrics["iou"]
                best_step = step
            model.train()

    elapsed = max(time.perf_counter() - start, 1e-9)
    if eval_every <= 0:
        best_iou = float("nan")
        best_step = 0
    return {
        "train_loss": loss_value,
        "best_iou": best_iou,
        "best_step": best_step,
        "images_per_sec": examples_seen / elapsed,
    }


def evaluate_mask(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dice_weight: float,
    threshold: float,
    mask_size: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    probs_all = []
    centers_all = []
    targets_all = []
    with torch.no_grad():
        for images, centers, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += mask_loss(logits, targets, dice_weight=dice_weight).item() * int(images.shape[0])
            probs_all.append(logits.sigmoid().cpu())
            centers_all.append(centers.cpu())
            targets_all.append(targets.cpu())
            total_examples += int(images.shape[0])

    probs = torch.cat(probs_all, dim=0)
    centers = torch.cat(centers_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    metrics = mask_metrics_from_probs(probs, centers, targets, threshold=threshold, mask_size=mask_size)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics


def mask_loss(logits: Tensor, targets: Tensor, dice_weight: float) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    if dice_weight <= 0:
        return bce
    probs = logits.sigmoid()
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6)).mean()
    return bce + dice_weight * dice_loss


def bbox_mask_target(sample: BBoxSample, size: int) -> Tensor:
    xmin = max(0.0, min(1.0, sample.xmin / sample.width))
    xmax = max(0.0, min(1.0, sample.xmax / sample.width))
    ymin = max(0.0, min(1.0, sample.ymin / sample.height))
    ymax = max(0.0, min(1.0, sample.ymax / sample.height))
    coords = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    mask = ((xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)).float()
    if float(mask.sum()) == 0.0:
        cx, cy = bbox_center_target(sample)
        x = min(size - 1, max(0, int(cx * size)))
        y = min(size - 1, max(0, int(cy * size)))
        mask[y, x] = 1.0
    return mask


def center_box_prior_mask(size: int, area_fraction: float) -> Tensor:
    side = max(1.0 / size, min(1.0, area_fraction ** 0.5))
    xmin = 0.5 - side / 2.0
    xmax = 0.5 + side / 2.0
    ymin = 0.5 - side / 2.0
    ymax = 0.5 + side / 2.0
    coords = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return ((xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)).float()


def mask_metrics_from_probs(
    probs: Tensor,
    centers: Tensor,
    targets: Tensor,
    threshold: float,
    mask_size: int,
) -> dict[str, float]:
    pred = (probs >= threshold).float()
    target = (targets > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    pred_area = pred.sum(dim=dims)
    target_area = target.sum(dim=dims)
    union = pred_area + target_area - intersection
    sample_iou = (intersection + 1e-6) / (union + 1e-6)
    iou = sample_iou.mean()
    dice = ((2.0 * intersection + 1e-6) / (pred_area + target_area + 1e-6)).mean()

    tp = float((pred * target).sum().item())
    tn = float(((1.0 - pred) * (1.0 - target)).sum().item())
    fp = float((pred * (1.0 - target)).sum().item())
    fn = float(((1.0 - pred) * target).sum().item())
    pos_recall = tp / max(1e-12, tp + fn)
    neg_recall = tn / max(1e-12, tn + fp)
    balanced_acc = 0.5 * (pos_recall + neg_recall)

    pred_centers = mask_centroid_from_probs(probs, mask_size)
    center_l2 = (pred_centers - centers).norm(dim=1)
    area_abs_error = (probs.flatten(1).mean(dim=1) - targets.flatten(1).mean(dim=1)).abs().mean()
    target_area_fraction = target.flatten(1).mean(dim=1)
    return {
        "iou": float(iou.item()),
        "small_iou": mean_masked(sample_iou, target_area_fraction < 0.10),
        "medium_iou": mean_masked(
            sample_iou,
            (target_area_fraction >= 0.10) & (target_area_fraction < 0.30),
        ),
        "large_iou": mean_masked(sample_iou, target_area_fraction >= 0.30),
        "dice": float(dice.item()),
        "cell_balanced_acc": float(balanced_acc),
        "pos_recall": float(pos_recall),
        "neg_recall": float(neg_recall),
        "center_mean_l2": float(center_l2.mean().item()),
        "center_median_l2": float(center_l2.median().item()),
        "pck_010": float((center_l2 < 0.10).float().mean().item()),
        "pck_020": float((center_l2 < 0.20).float().mean().item()),
        "area_abs_error": float(area_abs_error.item()),
    }


def mean_masked(values: Tensor, mask: Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(values[mask].mean().item())


def mask_centroid_from_probs(probs: Tensor, size: int) -> Tensor:
    coords = (torch.arange(size, dtype=probs.dtype, device=probs.device) + 0.5) / size
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    flat = probs.flatten(1)
    weights = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights @ grid


def targets_for_dataset(dataset: Dataset[tuple[Tensor, Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent_centers, parent_masks = targets_for_dataset(dataset.dataset)
        indices = torch.tensor(dataset.indices, dtype=torch.long)
        return parent_centers[indices], parent_masks[indices]
    if isinstance(dataset, BBoxMaskDataset):
        return (
            torch.tensor(dataset.centers, dtype=torch.float32),
            torch.stack(dataset.masks, dim=0),
        )
    centers = []
    masks = []
    for idx in range(len(dataset)):
        _, center, mask = dataset[idx]
        centers.append(center)
        masks.append(mask)
    return torch.stack(centers, dim=0), torch.stack(masks, dim=0)


def build_mask_priors(targets: Tensor, mask_size: int) -> dict[str, float]:
    centers = torch.full((targets.shape[0], 2), 0.5)
    mean_prior = targets.mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
    area_fraction = float(targets.flatten(1).mean().item())
    center_prior = center_box_prior_mask(mask_size, area_fraction).clamp(1e-4, 1.0 - 1e-4)
    uniform_prior = torch.full_like(mean_prior, max(1e-4, min(1.0 - 1e-4, area_fraction)))
    mean_metrics = mask_metrics_from_probs(
        mean_prior.unsqueeze(0).expand_as(targets),
        centers,
        targets,
        threshold=0.5,
        mask_size=mask_size,
    )
    center_metrics = mask_metrics_from_probs(
        center_prior.unsqueeze(0).expand_as(targets),
        centers,
        targets,
        threshold=0.5,
        mask_size=mask_size,
    )
    return {
        "mean_prior_loss": float(F.binary_cross_entropy(mean_prior.expand_as(targets), targets).item()),
        "mean_prior_iou": mean_metrics["iou"],
        "center_box_prior_iou": center_metrics["iou"],
        "uniform_prior_loss": float(F.binary_cross_entropy(uniform_prior.expand_as(targets), targets).item()),
    }


def write_mask_csv(path: Path, rows: list[MaskResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MaskResultRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_mask_summary(rows: list[MaskResultRow]) -> None:
    print(
        "summary_model,params_mean,iou_mean,dice_mean,cell_balanced_acc_mean,"
        "small_iou_mean,medium_iou_mean,large_iou_mean,"
        "pos_recall_mean,neg_recall_mean,center_l2_mean,pck010_mean,pck020_mean,"
        "area_abs_error_mean,best_iou_mean,mean_prior_iou_mean,center_box_prior_iou_mean,"
        "images_per_sec_mean,gate_mean,scaled_read_ratio_mean"
    )
    for model in sorted({row.model for row in rows}):
        model_rows = [row for row in rows if row.model == model]
        print(
            f"{model},{mean([row.params for row in model_rows]):.0f},"
            f"{mean([row.iou for row in model_rows]):.3f},"
            f"{mean([row.dice for row in model_rows]):.3f},"
            f"{mean([row.cell_balanced_acc for row in model_rows]):.3f},"
            f"{mean_or_nan([row.small_iou for row in model_rows if not is_nan(row.small_iou)]):.3f},"
            f"{mean_or_nan([row.medium_iou for row in model_rows if not is_nan(row.medium_iou)]):.3f},"
            f"{mean_or_nan([row.large_iou for row in model_rows if not is_nan(row.large_iou)]):.3f},"
            f"{mean([row.pos_recall for row in model_rows]):.3f},"
            f"{mean([row.neg_recall for row in model_rows]):.3f},"
            f"{mean([row.center_mean_l2 for row in model_rows]):.4f},"
            f"{mean([row.pck_010 for row in model_rows]):.3f},"
            f"{mean([row.pck_020 for row in model_rows]):.3f},"
            f"{mean([row.area_abs_error for row in model_rows]):.3f},"
            f"{mean([row.best_iou for row in model_rows]):.3f},"
            f"{mean([row.mean_prior_iou for row in model_rows]):.3f},"
            f"{mean([row.center_box_prior_iou for row in model_rows]):.3f},"
            f"{mean([row.images_per_sec for row in model_rows]):.2f},"
            f"{mean_or_nan([row.gate_mean for row in model_rows if not is_nan(row.gate_mean)]):.5f},"
            f"{mean_or_nan([row.scaled_read_ratio_mean for row in model_rows if not is_nan(row.scaled_read_ratio_mean)]):.5f}"
        )


def print_mask_paired_summary(rows: list[MaskResultRow], reference_model: str) -> None:
    if reference_model not in {row.model for row in rows}:
        return
    print(f"paired_mask_vs,{reference_model}")
    print(
        "paired_model,iou_delta_mean,iou_delta_std,iou_wins,"
        "dice_delta_mean,dice_delta_std,dice_wins,"
        "center_l2_delta_mean,center_l2_delta_std,center_l2_wins"
    )
    run_seeds = sorted({row.run_seed for row in rows})
    for model in sorted({row.model for row in rows}):
        if model == reference_model:
            continue
        iou_deltas = []
        dice_deltas = []
        center_deltas = []
        for run_seed in run_seeds:
            ref = find_row(rows, run_seed=run_seed, model=reference_model)
            cur = find_row(rows, run_seed=run_seed, model=model)
            if ref is None or cur is None:
                continue
            iou_deltas.append(cur.iou - ref.iou)
            dice_deltas.append(cur.dice - ref.dice)
            center_deltas.append(cur.center_mean_l2 - ref.center_mean_l2)
        if not iou_deltas:
            continue
        print(
            f"{model},"
            f"{mean(iou_deltas):.3f},{pstdev(iou_deltas):.3f},{wins_higher(iou_deltas)},"
            f"{mean(dice_deltas):.3f},{pstdev(dice_deltas):.3f},{wins_higher(dice_deltas)},"
            f"{mean(center_deltas):.4f},{pstdev(center_deltas):.4f},{wins_lower(center_deltas)}"
        )


def find_row(rows: list[MaskResultRow], run_seed: int, model: str) -> MaskResultRow | None:
    for row in rows:
        if row.run_seed == run_seed and row.model == model:
            return row
    return None


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values))


def mean_or_nan(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5


def wins_higher(values: list[float]) -> str:
    return f"{sum(value > 0 for value in values)}/{len(values)}"


def wins_lower(values: list[float]) -> str:
    return f"{sum(value < 0 for value in values)}/{len(values)}"


def is_nan(value: float) -> bool:
    return value != value


if __name__ == "__main__":
    main()
