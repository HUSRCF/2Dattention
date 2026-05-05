"""Compare small models on DET-derived bbox center regression."""

from __future__ import annotations

import argparse
import csv
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
from compare_bbox_probe_models import (  # noqa: E402
    BBoxSample,
    load_largest_bbox_samples,
)
from compare_imagefolder_models import (  # noqa: E402
    MODEL_NAMES,
    MODEL_SEED_OFFSETS,
    build_model,
    build_train_loader,
    collect_mechanism_stats,
    count_parameters,
    get_mps_memory_mb,
    split_dataset,
)


@dataclass(frozen=True)
class RegressionResultRow:
    dataset: str
    device: str
    run_seed: int
    model: str
    seed: int
    steps: int
    params: int
    train_loss: float
    train_mae_x: float
    train_mae_y: float
    eval_loss: float
    mae_x: float
    mae_y: float
    mean_l2: float
    median_l2: float
    pck_005: float
    pck_010: float
    pck_020: float
    center_baseline_l2: float
    mean_baseline_l2: float
    best_mean_l2: float
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


class BBoxCenterDataset(Dataset[tuple[Tensor, Tensor]]):
    """One largest-object normalized bbox center target per DET validation image."""

    def __init__(self, samples: list[BBoxSample], image_size: int) -> None:
        self.samples = samples
        self.targets = [bbox_center_target(sample) for sample in samples]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, torch.tensor(self.targets[index], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument(
        "--anno-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_bbox_val/ILSVRC2013_DET_bbox_val"),
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--reference-model", choices=MODEL_NAMES, default="no_prefill_local_mix")
    parser.add_argument(
        "--reference-models",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="optional paired references; defaults to --reference-model",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/bbox_center_regression_compare.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    samples = load_largest_bbox_samples(args.anno_root, args.image_root)
    dataset = BBoxCenterDataset(samples=samples, image_size=args.image_size)

    print("device:", device)
    print("task: bbox_center_regression")
    print("images:", len(dataset))
    print("steps:", args.steps)
    print("seeds:", args.seeds)
    print(
        "model,seed,params,train_loss,train_mae_x,train_mae_y,"
        "eval_loss,mae_x,mae_y,mean_l2,median_l2,pck_010,best_mean_l2,best_step,images_per_sec"
    )

    rows: list[RegressionResultRow] = []
    for seed_idx in range(args.seeds):
        split_seed = args.seed + seed_idx
        train_set, eval_set = split_dataset(dataset, args.train_frac, split_seed)
        eval_loader = DataLoader(
            eval_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        eval_targets = targets_for_dataset(eval_set)
        center_baseline_l2 = baseline_l2(eval_targets, torch.tensor([0.5, 0.5]))
        mean_baseline_l2 = baseline_l2(eval_targets, eval_targets.mean(dim=0))

        for model_name in args.models:
            train_loader = build_train_loader(
                train_set=train_set,
                batch_size=args.batch_size,
                seed=10_000_000 + split_seed,
            )
            model_seed = split_seed + MODEL_SEED_OFFSETS[model_name]
            torch.manual_seed(model_seed)
            model = build_model(
                name=model_name,
                embed_dim=args.embed_dim,
                image_size=args.image_size,
                num_classes=2,
            ).to(device)
            params = count_parameters(model)
            train_result = train_regression_model(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                steps=args.steps,
                lr=args.lr,
                eval_every=args.eval_every,
            )
            eval_metrics = evaluate_regression(model, eval_loader, device)
            mechanism_stats = collect_mechanism_stats(model, eval_loader, device)
            mps_current_mem_mb, mps_driver_mem_mb = get_mps_memory_mb()
            row = RegressionResultRow(
                dataset="bbox_center_regression",
                device=str(device),
                run_seed=split_seed,
                model=model_name,
                seed=model_seed,
                steps=args.steps,
                params=params,
                train_loss=train_result["train_loss"],
                train_mae_x=train_result["train_mae_x"],
                train_mae_y=train_result["train_mae_y"],
                eval_loss=eval_metrics["loss"],
                mae_x=eval_metrics["mae_x"],
                mae_y=eval_metrics["mae_y"],
                mean_l2=eval_metrics["mean_l2"],
                median_l2=eval_metrics["median_l2"],
                pck_005=eval_metrics["pck_005"],
                pck_010=eval_metrics["pck_010"],
                pck_020=eval_metrics["pck_020"],
                center_baseline_l2=center_baseline_l2,
                mean_baseline_l2=mean_baseline_l2,
                best_mean_l2=train_result["best_mean_l2"],
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
                f"{train_result['train_mae_x']:.4f},{train_result['train_mae_y']:.4f},"
                f"{eval_metrics['loss']:.4f},"
                f"{eval_metrics['mae_x']:.4f},{eval_metrics['mae_y']:.4f},"
                f"{eval_metrics['mean_l2']:.4f},{eval_metrics['median_l2']:.4f},"
                f"{eval_metrics['pck_010']:.3f},"
                f"{train_result['best_mean_l2']:.4f},{train_result['best_step']},"
                f"{train_result['images_per_sec']:.2f}"
            )

    write_regression_csv(args.out, rows)
    print("saved_csv:", args.out)
    print_regression_summary(rows)
    reference_models = args.reference_models or [args.reference_model]
    for reference_model in dict.fromkeys(reference_models):
        print_regression_paired_summary(rows, reference_model=reference_model)


def train_regression_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    eval_every: int,
) -> dict[str, float | int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    loader_iter = cycle(train_loader)
    loss_value = 0.0
    mae_x = 0.0
    mae_y = 0.0
    best_mean_l2 = float("inf")
    best_step = 0
    examples_seen = 0
    start = time.perf_counter()

    for step in range(1, steps + 1):
        images, targets = next(loader_iter)
        images = images.to(device)
        targets = targets.to(device)
        examples_seen += int(targets.shape[0])
        predictions = regression_from_output(model(images))
        loss = F.smooth_l1_loss(predictions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        error = (predictions.detach() - targets).abs()
        loss_value = float(loss.item())
        mae_x = float(error[:, 0].mean().item())
        mae_y = float(error[:, 1].mean().item())

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            eval_metrics = evaluate_regression(model, eval_loader, device)
            if eval_metrics["mean_l2"] < best_mean_l2:
                best_mean_l2 = eval_metrics["mean_l2"]
                best_step = step
            model.train()

    elapsed = max(time.perf_counter() - start, 1e-9)
    if eval_every <= 0:
        best_mean_l2 = float("nan")
        best_step = 0

    return {
        "train_loss": loss_value,
        "train_mae_x": mae_x,
        "train_mae_y": mae_y,
        "best_mean_l2": best_mean_l2,
        "best_step": best_step,
        "images_per_sec": examples_seen / elapsed,
    }


def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions_all = []
    targets_all = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = regression_from_output(model(images))
            total_loss += F.smooth_l1_loss(predictions, targets, reduction="sum").item()
            total_examples += int(targets.shape[0])
            predictions_all.append(predictions.cpu())
            targets_all.append(targets.cpu())

    predictions_tensor = torch.cat(predictions_all, dim=0)
    targets_tensor = torch.cat(targets_all, dim=0)
    diff = predictions_tensor - targets_tensor
    abs_diff = diff.abs()
    l2 = diff.norm(dim=1)
    return {
        "loss": total_loss / max(1, total_examples * 2),
        "mae_x": float(abs_diff[:, 0].mean().item()),
        "mae_y": float(abs_diff[:, 1].mean().item()),
        "mean_l2": float(l2.mean().item()),
        "median_l2": float(l2.median().item()),
        "pck_005": float((l2 < 0.05).float().mean().item()),
        "pck_010": float((l2 < 0.10).float().mean().item()),
        "pck_020": float((l2 < 0.20).float().mean().item()),
    }


def regression_from_output(output: Tensor | dict[str, Tensor | list[Tensor]]) -> Tensor:
    if isinstance(output, dict):
        logits = output["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output['logits'] must be a tensor")
    else:
        logits = output
    if logits.shape[-1] != 2:
        raise ValueError(f"regression head must output 2 values, got {tuple(logits.shape)}")
    return torch.sigmoid(logits)


def bbox_center_target(sample: BBoxSample) -> tuple[float, float]:
    center_x = ((sample.xmin + sample.xmax) * 0.5) / max(1, sample.width)
    center_y = ((sample.ymin + sample.ymax) * 0.5) / max(1, sample.height)
    center_x = min(max(center_x, 0.0), 1.0)
    center_y = min(max(center_y, 0.0), 1.0)
    return center_x, center_y


def targets_for_dataset(dataset: Dataset[tuple[Tensor, Tensor]]) -> Tensor:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent = targets_for_dataset(dataset.dataset)
        return parent[torch.tensor(dataset.indices, dtype=torch.long)]
    if isinstance(dataset, BBoxCenterDataset):
        return torch.tensor(dataset.targets, dtype=torch.float32)
    return torch.stack([dataset[idx][1] for idx in range(len(dataset))])


def baseline_l2(targets: Tensor, prediction: Tensor) -> float:
    prediction = prediction.to(dtype=targets.dtype)
    return float((targets - prediction.view(1, 2)).norm(dim=1).mean().item())


def write_regression_csv(path: Path, rows: list[RegressionResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(RegressionResultRow.__dataclass_fields__),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_regression_summary(rows: list[RegressionResultRow]) -> None:
    print(
        "summary_model,params_mean,mae_x_mean,mae_y_mean,mean_l2_mean,"
        "median_l2_mean,pck_005_mean,pck_010_mean,pck_020_mean,"
        "best_mean_l2_mean,center_baseline_l2_mean,mean_baseline_l2_mean,"
        "images_per_sec_mean,gate_mean,scaled_read_ratio_mean"
    )
    for model in sorted({row.model for row in rows}):
        model_rows = [row for row in rows if row.model == model]
        print(
            f"{model},{mean([row.params for row in model_rows]):.0f},"
            f"{mean([row.mae_x for row in model_rows]):.4f},"
            f"{mean([row.mae_y for row in model_rows]):.4f},"
            f"{mean([row.mean_l2 for row in model_rows]):.4f},"
            f"{mean([row.median_l2 for row in model_rows]):.4f},"
            f"{mean([row.pck_005 for row in model_rows]):.3f},"
            f"{mean([row.pck_010 for row in model_rows]):.3f},"
            f"{mean([row.pck_020 for row in model_rows]):.3f},"
            f"{mean([row.best_mean_l2 for row in model_rows]):.4f},"
            f"{mean([row.center_baseline_l2 for row in model_rows]):.4f},"
            f"{mean([row.mean_baseline_l2 for row in model_rows]):.4f},"
            f"{mean([row.images_per_sec for row in model_rows]):.2f},"
            f"{mean_or_nan([row.gate_mean for row in model_rows if not is_nan(row.gate_mean)]):.5f},"
            f"{mean_or_nan([row.scaled_read_ratio_mean for row in model_rows if not is_nan(row.scaled_read_ratio_mean)]):.5f}"
        )


def print_regression_paired_summary(
    rows: list[RegressionResultRow],
    reference_model: str,
) -> None:
    if reference_model not in {row.model for row in rows}:
        return
    print(f"paired_regression_vs,{reference_model}")
    print(
        "paired_model,mean_l2_delta_mean,mean_l2_delta_std,mean_l2_wins,"
        "pck010_delta_mean,pck010_delta_std,pck010_wins,"
        "pck020_delta_mean,pck020_delta_std,pck020_wins"
    )
    run_seeds = sorted({row.run_seed for row in rows})
    for model in sorted({row.model for row in rows}):
        if model == reference_model:
            continue
        l2_deltas = []
        pck010_deltas = []
        pck020_deltas = []
        for run_seed in run_seeds:
            ref = find_row(rows, run_seed=run_seed, model=reference_model)
            cur = find_row(rows, run_seed=run_seed, model=model)
            if ref is None or cur is None:
                continue
            l2_deltas.append(cur.mean_l2 - ref.mean_l2)
            pck010_deltas.append(cur.pck_010 - ref.pck_010)
            pck020_deltas.append(cur.pck_020 - ref.pck_020)
        if not l2_deltas:
            continue
        print(
            f"{model},"
            f"{mean(l2_deltas):.4f},{pstdev(l2_deltas):.4f},{wins_lower(l2_deltas)},"
            f"{mean(pck010_deltas):.3f},{pstdev(pck010_deltas):.3f},{wins_higher(pck010_deltas)},"
            f"{mean(pck020_deltas):.3f},{pstdev(pck020_deltas):.3f},{wins_higher(pck020_deltas)}"
        )


def find_row(
    rows: list[RegressionResultRow],
    run_seed: int,
    model: str,
) -> RegressionResultRow | None:
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
