"""Compare small models on a supervised torchvision ImageFolder dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import (  # noqa: E402
    ConvOnlyClassifier,
    MultiCLSViTClassifier,
    TinyAnchorOnlyAttnResClassifier,
    TinyAnchorPrefillLatticeAttnRes,
    TinyGraphPrefillLatticeAttnRes,
    TinyPrefillLocalMixClassifier,
    TinyPrefillLatticeAttnRes,
    TinyViTClassifier,
    TinyXAttnResClassifier,
    get_best_device,
)


MODEL_NAMES = (
    "conv_only",
    "tiny_vit",
    "multi_cls_vit",
    "xattnres_style",
    "xattnres_equal_params",
    "prefill_local_mix",
    "no_prefill_local_mix",
    "anchor_read_only_no_lattice",
    "lattice_only_no_anchor",
    "prefill_lattice_attnres",
    "anchor_prefill_attnres",
    "anchor_no_prefill",
    "anchor_fixed_gamma_0",
    "graph_prefill_attnres",
)

MODEL_SEED_OFFSETS = {
    "conv_only": 0,
    "tiny_vit": 10_000,
    "multi_cls_vit": 20_000,
    "xattnres_style": 30_000,
    "xattnres_equal_params": 40_000,
    "prefill_local_mix": 50_000,
    "no_prefill_local_mix": 60_000,
    "anchor_read_only_no_lattice": 70_000,
    "lattice_only_no_anchor": 80_000,
    "prefill_lattice_attnres": 90_000,
    "anchor_prefill_attnres": 100_000,
    "anchor_no_prefill": 110_000,
    "anchor_fixed_gamma_0": 120_000,
    "graph_prefill_attnres": 130_000,
}


@dataclass(frozen=True)
class ResultRow:
    dataset: str
    device: str
    run_seed: int
    model: str
    seed: int
    steps: int
    params: int
    train_loss: float
    train_acc: float
    eval_loss: float
    eval_acc: float
    best_eval_loss: float
    best_eval_acc: float
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_val_supervised/single_label_imagefolder"),
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--reference-model", choices=MODEL_NAMES, default="xattnres_style")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="evaluate during training every N steps; 0 disables intermediate eval",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/imagefolder_compare.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    dataset = build_dataset(args.data_root, args.image_size)
    if len(dataset.classes) < 2:
        raise ValueError("ImageFolder must contain at least two class folders")

    print("device:", device)
    print("data_root:", args.data_root)
    print("classes:", len(dataset.classes))
    print("images:", len(dataset))
    print("steps:", args.steps)
    print("seeds:", args.seeds)
    print("model,seed,params,train_loss,train_acc,eval_loss,eval_acc,best_eval_acc,best_step,images_per_sec")

    rows: list[ResultRow] = []
    for seed_idx in range(args.seeds):
        split_seed = args.seed + seed_idx
        train_set, eval_set = split_dataset(dataset, args.train_frac, split_seed)
        eval_loader = DataLoader(
            eval_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

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
                num_classes=len(dataset.classes),
            ).to(device)
            params = count_parameters(model)
            train_result = train_model(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                steps=args.steps,
                lr=args.lr,
                eval_every=args.eval_every,
            )
            eval_loss, eval_acc = evaluate(model, eval_loader, device)
            mechanism_stats = collect_mechanism_stats(model, eval_loader, device)
            mps_current_mem_mb, mps_driver_mem_mb = get_mps_memory_mb()
            row = ResultRow(
                dataset=str(args.data_root),
                device=str(device),
                run_seed=split_seed,
                model=model_name,
                seed=model_seed,
                steps=args.steps,
                params=params,
                train_loss=train_result["train_loss"],
                train_acc=train_result["train_acc"],
                eval_loss=eval_loss,
                eval_acc=eval_acc,
                best_eval_loss=train_result["best_eval_loss"],
                best_eval_acc=train_result["best_eval_acc"],
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
                f"{train_result['train_loss']:.4f},{train_result['train_acc']:.3f},"
                f"{eval_loss:.4f},{eval_acc:.3f},"
                f"{train_result['best_eval_acc']:.3f},{train_result['best_step']},"
                f"{train_result['images_per_sec']:.2f},"
                f"gate={mechanism_stats['gate_mean']:.5f},"
                f"ratio={mechanism_stats['scaled_read_ratio_mean']:.5f}"
            )

    write_csv(args.out, rows)
    print("saved_csv:", args.out)
    print_summary(rows)
    print_paired_summary(rows, reference_model=args.reference_model)


def build_dataset(data_root: Path, image_size: int) -> datasets.ImageFolder:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return datasets.ImageFolder(data_root, transform=transform)


def split_dataset(
    dataset: datasets.ImageFolder,
    train_frac: float,
    seed: int,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    train_size = max(2, int(len(dataset) * train_frac))
    eval_size = len(dataset) - train_size
    if eval_size == 0:
        train_size -= 1
        eval_size = 1
    return random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(seed),
    )


def build_train_loader(
    train_set: torch.utils.data.Subset,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Build a fresh loader so every model sees the same shuffled batch order."""

    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def build_model(
    name: str,
    embed_dim: int,
    image_size: int,
    num_classes: int,
) -> nn.Module:
    builders = {
        "conv_only": lambda: ConvOnlyClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "tiny_vit": lambda: TinyViTClassifier(
            embed_dim=embed_dim,
            image_size=image_size,
            num_classes=num_classes,
        ),
        "multi_cls_vit": lambda: MultiCLSViTClassifier(
            embed_dim=embed_dim,
            image_size=image_size,
            num_classes=num_classes,
            global_tokens=4,
        ),
        "xattnres_style": lambda: TinyXAttnResClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "xattnres_equal_params": lambda: TinyXAttnResClassifier(
            embed_dim=36,
            num_classes=num_classes,
        ),
        "prefill_local_mix": lambda: TinyPrefillLocalMixClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "no_prefill_local_mix": lambda: make_no_prefill_local_mix(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "anchor_read_only_no_lattice": lambda: TinyAnchorOnlyAttnResClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "lattice_only_no_anchor": lambda: TinyPrefillLatticeAttnRes(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "prefill_lattice_attnres": lambda: TinyPrefillLatticeAttnRes(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "anchor_prefill_attnres": lambda: TinyAnchorPrefillLatticeAttnRes(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "anchor_no_prefill": lambda: make_anchor_no_prefill(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "anchor_fixed_gamma_0": lambda: make_anchor_fixed_gamma_0(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "graph_prefill_attnres": lambda: TinyGraphPrefillLatticeAttnRes(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
    }
    return builders[name]()


class IdentityPrefill(nn.Module):
    """Return only the patch lattice so anchor reads cannot use prefilled states."""

    def forward(self, x: Tensor) -> list[Tensor]:
        return [x]


def make_anchor_no_prefill(embed_dim: int, num_classes: int) -> nn.Module:
    model = TinyAnchorPrefillLatticeAttnRes(
        embed_dim=embed_dim,
        num_classes=num_classes,
    )
    model.prefill = IdentityPrefill()
    return model


def make_no_prefill_local_mix(embed_dim: int, num_classes: int) -> nn.Module:
    model = TinyPrefillLocalMixClassifier(
        embed_dim=embed_dim,
        num_classes=num_classes,
    )
    model.prefill = IdentityPrefill()
    return model


def make_anchor_fixed_gamma_0(embed_dim: int, num_classes: int) -> nn.Module:
    model = TinyAnchorPrefillLatticeAttnRes(
        embed_dim=embed_dim,
        num_classes=num_classes,
        gate_init=0.0,
    )
    for module in model.modules():
        if hasattr(module, "read_gate"):
            module.read_gate.requires_grad_(False)
    return model


def train_model(
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
    accuracy = 0.0
    best_eval_loss = float("inf")
    best_eval_acc = 0.0
    best_step = 0
    examples_seen = 0
    start = time.perf_counter()

    for step in range(1, steps + 1):
        images, labels = next(loader_iter)
        images = images.to(device)
        labels = labels.to(device)
        examples_seen += int(labels.numel())
        logits = logits_from_output(model(images))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

        if eval_every > 0 and (step % eval_every == 0 or step == steps):
            eval_loss, eval_acc = evaluate(model, eval_loader, device)
            if eval_acc > best_eval_acc:
                best_eval_loss = eval_loss
                best_eval_acc = eval_acc
                best_step = step
            model.train()

    elapsed = max(time.perf_counter() - start, 1e-9)
    if eval_every <= 0:
        best_eval_loss = float("nan")
        best_eval_acc = float("nan")
        best_step = 0

    return {
        "train_loss": loss_value,
        "train_acc": accuracy,
        "best_eval_loss": best_eval_loss,
        "best_eval_acc": best_eval_acc,
        "best_step": best_step,
        "images_per_sec": examples_seen / elapsed,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = logits_from_output(model(images))
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += int(labels.numel())
    return total_loss / total_examples, total_correct / total_examples


def logits_from_output(output: Tensor | dict[str, Tensor | list[Tensor]]) -> Tensor:
    if isinstance(output, dict):
        logits = output["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output['logits'] must be a tensor")
        return logits
    return output


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "device",
                "run_seed",
                "model",
                "seed",
                "steps",
                "params",
                "train_loss",
                "train_acc",
                "eval_loss",
                "eval_acc",
                "best_eval_loss",
                "best_eval_acc",
                "best_step",
                "images_per_sec",
                "mps_current_mem_mb",
                "mps_driver_mem_mb",
                "gate_mean",
                "gate_max_abs",
                "read_norm_mean",
                "state_norm_mean",
                "scaled_read_ratio_mean",
                "block_stats_json",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_summary(rows: list[ResultRow]) -> None:
    models = sorted({row.model for row in rows})
    print("summary_model,params_mean,eval_acc_mean,eval_acc_std,best_eval_acc_mean,images_per_sec_mean,gate_mean,scaled_read_ratio_mean")
    for model in models:
        model_rows = [row for row in rows if row.model == model]
        accuracies = [row.eval_acc for row in model_rows]
        best_accuracies = [
            row.best_eval_acc
            for row in model_rows
            if not torch.isnan(torch.tensor(row.best_eval_acc))
        ]
        params = [row.params for row in model_rows]
        speeds = [row.images_per_sec for row in model_rows]
        gates = [row.gate_mean for row in model_rows if not torch.isnan(torch.tensor(row.gate_mean))]
        ratios = [
            row.scaled_read_ratio_mean
            for row in model_rows
            if not torch.isnan(torch.tensor(row.scaled_read_ratio_mean))
        ]
        best_mean = mean(best_accuracies) if best_accuracies else float("nan")
        gate_mean = mean(gates) if gates else float("nan")
        ratio_mean = mean(ratios) if ratios else float("nan")
        print(
            f"{model},{mean(params):.0f},{mean(accuracies):.3f},"
            f"{pstdev(accuracies):.3f},{best_mean:.3f},{mean(speeds):.2f},"
            f"{gate_mean:.5f},{ratio_mean:.5f}"
        )


def print_paired_summary(rows: list[ResultRow], reference_model: str) -> None:
    if reference_model not in {row.model for row in rows}:
        return
    print(f"paired_vs,{reference_model}")
    print("paired_model,final_delta_mean,final_delta_std,final_wins,best_delta_mean,best_delta_std,best_wins")
    run_seeds = sorted({row.run_seed for row in rows})
    for model in sorted({row.model for row in rows}):
        if model == reference_model:
            continue
        final_deltas = []
        best_deltas = []
        for run_seed in run_seeds:
            ref = find_row(rows, run_seed=run_seed, model=reference_model)
            cur = find_row(rows, run_seed=run_seed, model=model)
            if ref is None or cur is None:
                continue
            final_deltas.append(cur.eval_acc - ref.eval_acc)
            best_deltas.append(cur.best_eval_acc - ref.best_eval_acc)
        if not final_deltas:
            continue
        final_wins = sum(delta > 0 for delta in final_deltas)
        best_wins = sum(delta > 0 for delta in best_deltas)
        final_std = pstdev(final_deltas) if len(final_deltas) > 1 else 0.0
        best_std = pstdev(best_deltas) if len(best_deltas) > 1 else 0.0
        print(
            f"{model},{mean(final_deltas):.3f},{final_std:.3f},"
            f"{final_wins}/{len(final_deltas)},"
            f"{mean(best_deltas):.3f},{best_std:.3f},"
            f"{best_wins}/{len(best_deltas)}"
        )


def find_row(rows: list[ResultRow], run_seed: int, model: str) -> ResultRow | None:
    for row in rows:
        if row.run_seed == run_seed and row.model == model:
            return row
    return None


def get_mps_memory_mb() -> tuple[float, float]:
    if not torch.backends.mps.is_available():
        return 0.0, 0.0
    current = torch.mps.current_allocated_memory() / (1024 * 1024)
    driver = torch.mps.driver_allocated_memory() / (1024 * 1024)
    return current, driver


def collect_mechanism_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    read_blocks = [
        module for module in model.modules() if hasattr(module, "read_gate")
    ]
    if not read_blocks:
        return {
            "gate_mean": float("nan"),
            "gate_max_abs": float("nan"),
            "read_norm_mean": float("nan"),
            "state_norm_mean": float("nan"),
            "scaled_read_ratio_mean": float("nan"),
            "block_stats_json": "[]",
        }

    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images.to(device)
        _ = logits_from_output(model(images))

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
        return {
            "gate_mean": float("nan"),
            "gate_max_abs": float("nan"),
            "read_norm_mean": float("nan"),
            "state_norm_mean": float("nan"),
            "scaled_read_ratio_mean": float("nan"),
            "block_stats_json": "[]",
        }

    return {
        "gate_mean": mean(gates),
        "gate_max_abs": max(abs(value) for value in gates),
        "read_norm_mean": mean(read_norms),
        "state_norm_mean": mean(state_norms),
        "scaled_read_ratio_mean": mean(ratios),
        "block_stats_json": json.dumps(block_stats, separators=(",", ":")),
    }


if __name__ == "__main__":
    main()
