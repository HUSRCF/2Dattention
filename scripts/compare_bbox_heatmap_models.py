"""Compare small models on explicit DET bbox-center heatmap localization."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from compare_bbox_center_regression_models import (  # noqa: E402
    bbox_center_target,
    baseline_l2,
)
from compare_bbox_probe_models import BBoxSample, load_largest_bbox_samples  # noqa: E402
from compare_imagefolder_models import (  # noqa: E402
    MODEL_NAMES,
    MODEL_SEED_OFFSETS,
    build_model,
    count_parameters,
    get_mps_memory_mb,
    split_dataset,
)

HEATMAP_MODEL_NAMES = tuple(
    name for name in MODEL_NAMES if name not in {"conv_only", "tiny_vit", "multi_cls_vit"}
)


@dataclass(frozen=True)
class HeatmapResultRow:
    dataset: str
    device: str
    run_seed: int
    model: str
    seed: int
    steps: int
    params: int
    train_loss: float
    eval_loss: float
    argmax_mean_l2: float
    argmax_median_l2: float
    softargmax_mean_l2: float
    softargmax_median_l2: float
    pck_005: float
    pck_010: float
    pck_020: float
    top1_cell_acc: float
    entropy: float
    center_baseline_l2: float
    mean_baseline_l2: float
    uniform_loss: float
    center_prior_loss: float
    best_argmax_l2: float
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


class BBoxHeatmapDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Return image, normalized center, and a soft heatmap target."""

    def __init__(
        self,
        samples: list[BBoxSample],
        image_size: int,
        heatmap_size: int,
        sigma: float,
    ) -> None:
        self.samples = samples
        self.centers = [bbox_center_target(sample) for sample in samples]
        self.heatmap_size = heatmap_size
        self.sigma = sigma
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
        heatmap = gaussian_heatmap(center, self.heatmap_size, self.sigma)
        return tensor, center, heatmap


class HeatmapWrapper(nn.Module):
    """Attach a 1x1 heatmap head to the last 2D state of an existing model."""

    def __init__(self, backbone: nn.Module, embed_dim: int, heatmap_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.heatmap_size = heatmap_size
        self.heatmap_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        output = self.backbone(x)
        if not isinstance(output, dict) or "memories" not in output:
            raise TypeError("heatmap wrapper expects a dict output with memories")
        spatial_features = output.get("spatial_features")
        if isinstance(spatial_features, Tensor):
            state = spatial_features
        else:
            memories = output["memories"]
            if not isinstance(memories, list) or not memories:
                raise TypeError("model output['memories'] must be a non-empty list")
            state = memories[-1]
        if not isinstance(state, Tensor):
            raise TypeError("heatmap spatial feature must be a tensor")
        logits = self.heatmap_head(state)
        if logits.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            logits = F.interpolate(
                logits,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits.squeeze(1)


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
        help="models that expose a 2D memories list for heatmap heads",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--heatmap-size", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--reference-model",
        choices=HEATMAP_MODEL_NAMES,
        default="no_prefill_local_mix",
    )
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
        "--out",
        type=Path,
        default=Path("results/bbox_heatmap_compare.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    samples = load_largest_bbox_samples(args.anno_root, args.image_root)
    dataset = BBoxHeatmapDataset(
        samples=samples,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
    )

    print("device:", device)
    print("task: bbox_heatmap")
    print("images:", len(dataset))
    print("heatmap_size:", args.heatmap_size)
    print("sigma:", args.sigma)
    print("steps:", args.steps)
    print("seeds:", args.seeds)
    print(
        "model,seed,params,train_loss,eval_loss,argmax_l2,softargmax_l2,"
        "pck_010,top1_cell_acc,entropy,best_argmax_l2,best_step,images_per_sec"
    )

    rows: list[HeatmapResultRow] = []
    for seed_idx in range(args.seeds):
        split_seed = args.seed + seed_idx
        train_set, eval_set = split_dataset(dataset, args.train_frac, split_seed)
        eval_loader = DataLoader(
            eval_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        eval_targets = centers_for_dataset(eval_set)
        center_baseline_l2 = baseline_l2(eval_targets, torch.tensor([0.5, 0.5]))
        mean_baseline_l2 = baseline_l2(eval_targets, eval_targets.mean(dim=0))
        uniform_loss = baseline_heatmap_loss(
            eval_set,
            args.heatmap_size,
            None,
            args.sigma,
        )
        center_prior_loss = baseline_heatmap_loss(
            eval_set,
            args.heatmap_size,
            eval_targets.mean(dim=0),
            args.sigma,
        )

        for model_name in args.models:
            train_loader = build_heatmap_train_loader(
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
            model = HeatmapWrapper(
                backbone=backbone,
                embed_dim=args.embed_dim,
                heatmap_size=args.heatmap_size,
            ).to(device)
            params = count_parameters(model)
            train_result = train_heatmap_model(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                steps=args.steps,
                lr=args.lr,
                eval_every=args.eval_every,
                heatmap_size=args.heatmap_size,
            )
            eval_metrics = evaluate_heatmap(
                model=model,
                loader=eval_loader,
                device=device,
                heatmap_size=args.heatmap_size,
            )
            mechanism_stats = collect_heatmap_mechanism_stats(model, eval_loader, device)
            mps_current_mem_mb, mps_driver_mem_mb = get_mps_memory_mb()
            row = HeatmapResultRow(
                dataset="bbox_heatmap",
                device=str(device),
                run_seed=split_seed,
                model=model_name,
                seed=model_seed,
                steps=args.steps,
                params=params,
                train_loss=train_result["train_loss"],
                eval_loss=eval_metrics["loss"],
                argmax_mean_l2=eval_metrics["argmax_mean_l2"],
                argmax_median_l2=eval_metrics["argmax_median_l2"],
                softargmax_mean_l2=eval_metrics["softargmax_mean_l2"],
                softargmax_median_l2=eval_metrics["softargmax_median_l2"],
                pck_005=eval_metrics["pck_005"],
                pck_010=eval_metrics["pck_010"],
                pck_020=eval_metrics["pck_020"],
                top1_cell_acc=eval_metrics["top1_cell_acc"],
                entropy=eval_metrics["entropy"],
                center_baseline_l2=center_baseline_l2,
                mean_baseline_l2=mean_baseline_l2,
                uniform_loss=uniform_loss,
                center_prior_loss=center_prior_loss,
                best_argmax_l2=train_result["best_argmax_l2"],
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
                f"{train_result['train_loss']:.4f},"
                f"{eval_metrics['loss']:.4f},"
                f"{eval_metrics['argmax_mean_l2']:.4f},"
                f"{eval_metrics['softargmax_mean_l2']:.4f},"
                f"{eval_metrics['pck_010']:.3f},"
                f"{eval_metrics['top1_cell_acc']:.3f},"
                f"{eval_metrics['entropy']:.3f},"
                f"{train_result['best_argmax_l2']:.4f},"
                f"{train_result['best_step']},"
                f"{train_result['images_per_sec']:.2f}"
            )

    write_heatmap_csv(args.out, rows)
    print("saved_csv:", args.out)
    print_heatmap_summary(rows)
    reference_models = args.reference_models or [args.reference_model]
    for reference_model in dict.fromkeys(reference_models):
        print_heatmap_paired_summary(rows, reference_model=reference_model)


def build_heatmap_train_loader(
    train_set: Dataset[tuple[Tensor, Tensor, Tensor]],
    batch_size: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def train_heatmap_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    eval_every: int,
    heatmap_size: int,
) -> dict[str, float | int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    loader_iter = cycle(train_loader)
    loss_value = 0.0
    best_argmax_l2 = float("inf")
    best_step = 0
    examples_seen = 0
    start = time.perf_counter()

    for step in range(1, steps + 1):
        images, _, targets = next(loader_iter)
        images = images.to(device)
        targets = targets.to(device)
        examples_seen += int(targets.shape[0])
        logits = model(images)
        loss = soft_heatmap_ce(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            eval_metrics = evaluate_heatmap(model, eval_loader, device, heatmap_size)
            if eval_metrics["argmax_mean_l2"] < best_argmax_l2:
                best_argmax_l2 = eval_metrics["argmax_mean_l2"]
                best_step = step
            model.train()

    elapsed = max(time.perf_counter() - start, 1e-9)
    if eval_every <= 0:
        best_argmax_l2 = float("nan")
        best_step = 0
    return {
        "train_loss": loss_value,
        "best_argmax_l2": best_argmax_l2,
        "best_step": best_step,
        "images_per_sec": examples_seen / elapsed,
    }


def evaluate_heatmap(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    heatmap_size: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    argmax_predictions = []
    soft_predictions = []
    targets_all = []
    entropy_values = []
    top1_correct = 0
    with torch.no_grad():
        for images, centers, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += soft_heatmap_ce(logits, targets).item() * int(images.shape[0])
            probs = logits.flatten(1).softmax(dim=1)
            argmax_xy = argmax_coords(probs, heatmap_size)
            soft_xy = softargmax_coords(probs, heatmap_size)
            centers = centers.to(dtype=argmax_xy.dtype)
            target_cells = center_to_cell_index(centers, heatmap_size)
            top1_correct += int((probs.argmax(dim=1).cpu() == target_cells).sum().item())
            argmax_predictions.append(argmax_xy.cpu())
            soft_predictions.append(soft_xy.cpu())
            targets_all.append(centers.cpu())
            entropy_values.append(entropy(probs).cpu())
            total_examples += int(images.shape[0])

    argmax_tensor = torch.cat(argmax_predictions, dim=0)
    soft_tensor = torch.cat(soft_predictions, dim=0)
    targets_tensor = torch.cat(targets_all, dim=0)
    entropy_tensor = torch.cat(entropy_values, dim=0)
    argmax_l2 = (argmax_tensor - targets_tensor).norm(dim=1)
    soft_l2 = (soft_tensor - targets_tensor).norm(dim=1)
    return {
        "loss": total_loss / max(1, total_examples),
        "argmax_mean_l2": float(argmax_l2.mean().item()),
        "argmax_median_l2": float(argmax_l2.median().item()),
        "softargmax_mean_l2": float(soft_l2.mean().item()),
        "softargmax_median_l2": float(soft_l2.median().item()),
        "pck_005": float((argmax_l2 < 0.05).float().mean().item()),
        "pck_010": float((argmax_l2 < 0.10).float().mean().item()),
        "pck_020": float((argmax_l2 < 0.20).float().mean().item()),
        "top1_cell_acc": top1_correct / max(1, total_examples),
        "entropy": float(entropy_tensor.mean().item()),
    }


def soft_heatmap_ce(logits: Tensor, targets: Tensor) -> Tensor:
    logits_flat = logits.flatten(1)
    targets_flat = targets.flatten(1)
    log_probs = F.log_softmax(logits_flat, dim=1)
    return -(targets_flat * log_probs).sum(dim=1).mean()


def gaussian_heatmap(center: Tensor, size: int, sigma: float) -> Tensor:
    coords = torch.linspace(0.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    heatmap = torch.exp(-dist2 / (2 * sigma_normalized(sigma, size) ** 2))
    return heatmap / heatmap.sum().clamp_min(1e-12)


def sigma_normalized(sigma: float, size: int) -> float:
    return sigma / max(1, size - 1)


def argmax_coords(probs: Tensor, size: int) -> Tensor:
    indices = probs.argmax(dim=1)
    y = torch.div(indices, size, rounding_mode="floor")
    x = indices % size
    denom = max(1, size - 1)
    return torch.stack((x.float() / denom, y.float() / denom), dim=1)


def softargmax_coords(probs: Tensor, size: int) -> Tensor:
    coords = torch.linspace(0.0, 1.0, size, device=probs.device, dtype=probs.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return probs @ grid


def center_to_cell_index(centers: Tensor, size: int) -> Tensor:
    x = torch.clamp((centers[:, 0] * size).long(), min=0, max=size - 1)
    y = torch.clamp((centers[:, 1] * size).long(), min=0, max=size - 1)
    return (y * size + x).cpu()


def entropy(probs: Tensor) -> Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)


def centers_for_dataset(dataset: Dataset[tuple[Tensor, Tensor, Tensor]]) -> Tensor:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent = centers_for_dataset(dataset.dataset)
        return parent[torch.tensor(dataset.indices, dtype=torch.long)]
    if isinstance(dataset, BBoxHeatmapDataset):
        return torch.tensor(dataset.centers, dtype=torch.float32)
    return torch.stack([dataset[idx][1] for idx in range(len(dataset))])


def baseline_heatmap_loss(
    dataset: Dataset[tuple[Tensor, Tensor, Tensor]],
    size: int,
    center: Tensor | None,
    sigma: float,
) -> float:
    heatmaps = [dataset[idx][2] for idx in range(len(dataset))]
    targets = torch.stack(heatmaps, dim=0)
    if center is None:
        probs = torch.full((size, size), 1.0 / (size * size))
    else:
        probs = gaussian_heatmap(center.to(dtype=torch.float32), size, sigma)
    loss = -(targets.flatten(1) * probs.flatten().clamp_min(1e-12).log()).sum(dim=1)
    return float(loss.mean().item())


def collect_heatmap_mechanism_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | str]:
    read_blocks = [module for module in model.modules() if hasattr(module, "read_gate")]
    if not read_blocks:
        return empty_mechanism_stats()

    model.eval()
    with torch.no_grad():
        images, _, _ = next(iter(loader))
        _ = model(images.to(device))

    gates = []
    read_norms = []
    state_norms = []
    ratios = []
    block_stats = []
    for block_idx, block in enumerate(read_blocks):
        if not hasattr(block, "last_gate"):
            continue
        gate = float(block.last_gate)
        read_norm = float(block.last_read_norm)
        state_norm = float(block.last_state_norm)
        ratio = float(block.last_scaled_read_ratio)
        gates.append(gate)
        read_norms.append(read_norm)
        state_norms.append(state_norm)
        ratios.append(ratio)
        block_stats.append(
            {
                "block": block_idx,
                "gate": gate,
                "read_norm": read_norm,
                "state_norm": state_norm,
                "scaled_read_ratio": ratio,
            }
        )

    if not gates:
        return empty_mechanism_stats()

    return {
        "gate_mean": mean(gates),
        "gate_max_abs": max(abs(value) for value in gates),
        "read_norm_mean": mean(read_norms),
        "state_norm_mean": mean(state_norms),
        "scaled_read_ratio_mean": mean(ratios),
        "block_stats_json": json.dumps(block_stats, separators=(",", ":")),
    }


def empty_mechanism_stats() -> dict[str, float | str]:
    return {
        "gate_mean": float("nan"),
        "gate_max_abs": float("nan"),
        "read_norm_mean": float("nan"),
        "state_norm_mean": float("nan"),
        "scaled_read_ratio_mean": float("nan"),
        "block_stats_json": "[]",
    }


def write_heatmap_csv(path: Path, rows: list[HeatmapResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(HeatmapResultRow.__dataclass_fields__),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_heatmap_summary(rows: list[HeatmapResultRow]) -> None:
    print(
        "summary_model,params_mean,argmax_l2_mean,softargmax_l2_mean,"
        "pck005_mean,pck010_mean,pck020_mean,top1_cell_acc_mean,"
        "entropy_mean,best_argmax_l2_mean,center_baseline_l2_mean,"
        "mean_baseline_l2_mean,uniform_loss_mean,center_prior_loss_mean,"
        "images_per_sec_mean,gate_mean,scaled_read_ratio_mean"
    )
    for model in sorted({row.model for row in rows}):
        model_rows = [row for row in rows if row.model == model]
        print(
            f"{model},{mean([row.params for row in model_rows]):.0f},"
            f"{mean([row.argmax_mean_l2 for row in model_rows]):.4f},"
            f"{mean([row.softargmax_mean_l2 for row in model_rows]):.4f},"
            f"{mean([row.pck_005 for row in model_rows]):.3f},"
            f"{mean([row.pck_010 for row in model_rows]):.3f},"
            f"{mean([row.pck_020 for row in model_rows]):.3f},"
            f"{mean([row.top1_cell_acc for row in model_rows]):.3f},"
            f"{mean([row.entropy for row in model_rows]):.3f},"
            f"{mean([row.best_argmax_l2 for row in model_rows]):.4f},"
            f"{mean([row.center_baseline_l2 for row in model_rows]):.4f},"
            f"{mean([row.mean_baseline_l2 for row in model_rows]):.4f},"
            f"{mean([row.uniform_loss for row in model_rows]):.4f},"
            f"{mean([row.center_prior_loss for row in model_rows]):.4f},"
            f"{mean([row.images_per_sec for row in model_rows]):.2f},"
            f"{mean_or_nan([row.gate_mean for row in model_rows if not is_nan(row.gate_mean)]):.5f},"
            f"{mean_or_nan([row.scaled_read_ratio_mean for row in model_rows if not is_nan(row.scaled_read_ratio_mean)]):.5f}"
        )


def print_heatmap_paired_summary(
    rows: list[HeatmapResultRow],
    reference_model: str,
) -> None:
    if reference_model not in {row.model for row in rows}:
        return
    print(f"paired_heatmap_vs,{reference_model}")
    print(
        "paired_model,argmax_l2_delta_mean,argmax_l2_delta_std,argmax_l2_wins,"
        "softargmax_l2_delta_mean,softargmax_l2_delta_std,softargmax_l2_wins,"
        "pck010_delta_mean,pck010_delta_std,pck010_wins"
    )
    run_seeds = sorted({row.run_seed for row in rows})
    for model in sorted({row.model for row in rows}):
        if model == reference_model:
            continue
        argmax_l2_deltas = []
        soft_l2_deltas = []
        pck010_deltas = []
        for run_seed in run_seeds:
            ref = find_row(rows, run_seed=run_seed, model=reference_model)
            cur = find_row(rows, run_seed=run_seed, model=model)
            if ref is None or cur is None:
                continue
            argmax_l2_deltas.append(cur.argmax_mean_l2 - ref.argmax_mean_l2)
            soft_l2_deltas.append(cur.softargmax_mean_l2 - ref.softargmax_mean_l2)
            pck010_deltas.append(cur.pck_010 - ref.pck_010)
        if not argmax_l2_deltas:
            continue
        print(
            f"{model},"
            f"{mean(argmax_l2_deltas):.4f},{pstdev(argmax_l2_deltas):.4f},{wins_lower(argmax_l2_deltas)},"
            f"{mean(soft_l2_deltas):.4f},{pstdev(soft_l2_deltas):.4f},{wins_lower(soft_l2_deltas)},"
            f"{mean(pck010_deltas):.3f},{pstdev(pck010_deltas):.3f},{wins_higher(pck010_deltas)}"
        )


def find_row(
    rows: list[HeatmapResultRow],
    run_seed: int,
    model: str,
) -> HeatmapResultRow | None:
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


def wins_lower(values: list[float]) -> str:
    return f"{sum(value < 0 for value in values)}/{len(values)}"


def wins_higher(values: list[float]) -> str:
    return f"{sum(value > 0 for value in values)}/{len(values)}"


def is_nan(value: float) -> bool:
    return value != value


if __name__ == "__main__":
    main()
