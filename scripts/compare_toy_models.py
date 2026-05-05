"""Compare the prototype against simple baselines on synthetic 2D tasks."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import (  # noqa: E402
    ConvOnlyClassifier,
    SequenceTransformerClassifier,
    TinyAnchorPrefillLatticeAttnRes,
    TinyGraphPrefillLatticeAttnRes,
    TinyPrefillLatticeAttnRes,
    TinyViTClassifier,
    TinyXAttnResClassifier,
    get_best_device,
    sample_aligned_pair_batch,
    sample_distractor_aligned_pair_batch,
    sample_oriented_pair_batch,
)


Sampler = Callable[..., tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class ResultRow:
    task: str
    model: str
    seed: int
    steps: int
    train_loss: float
    train_acc: float
    eval_loss: float
    eval_acc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("oriented_pair", "aligned_pair", "distractor_aligned_pair"),
        default="oriented_pair",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/toy_compare.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    sampler: Sampler = {
        "oriented_pair": sample_oriented_pair_batch,
        "aligned_pair": sample_aligned_pair_batch,
        "distractor_aligned_pair": sample_distractor_aligned_pair_batch,
    }[args.task]

    print("device:", device)
    print("task:", args.task)
    print("steps:", args.steps)
    print("seeds:", args.seeds)

    builders = {
        "conv_only": lambda: ConvOnlyClassifier(embed_dim=args.embed_dim),
        "seq_transformer": lambda: SequenceTransformerClassifier(embed_dim=args.embed_dim),
        "tiny_vit": lambda: TinyViTClassifier(embed_dim=args.embed_dim),
        "xattnres_style": lambda: TinyXAttnResClassifier(
            embed_dim=args.embed_dim,
            num_classes=2,
            prefill_rounds=2,
            read_blocks=2,
        ),
        "prefill_lattice_attnres": lambda: TinyPrefillLatticeAttnRes(
            embed_dim=args.embed_dim,
            num_classes=2,
            prefill_rounds=2,
            read_blocks=2,
        ),
        "anchor_prefill_attnres": lambda: TinyAnchorPrefillLatticeAttnRes(
            embed_dim=args.embed_dim,
            num_classes=2,
            prefill_rounds=2,
            read_blocks=2,
        ),
        "graph_prefill_attnres": lambda: TinyGraphPrefillLatticeAttnRes(
            embed_dim=args.embed_dim,
            num_classes=2,
            prefill_rounds=2,
            read_blocks=2,
            graph_k=4,
        ),
    }

    rows: list[ResultRow] = []
    print("model,seed,train_loss,train_acc,eval_loss,eval_acc")
    for seed_idx in range(args.seeds):
        run_seed = args.seed + seed_idx
        for model_idx, (name, builder) in enumerate(builders.items()):
            model_seed = run_seed + model_idx * 10_000
            torch.manual_seed(model_seed)
            generator = torch.Generator().manual_seed(model_seed)
            model = builder().to(device)
            train_loss, train_acc = train_model(
                model=model,
                sampler=sampler,
                generator=generator,
                device=device,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            eval_loss, eval_acc = evaluate_model(
                model=model,
                sampler=sampler,
                generator=generator,
                device=device,
                batch_size=args.eval_batch_size,
            )
            row = ResultRow(
                task=args.task,
                model=name,
                seed=model_seed,
                steps=args.steps,
                train_loss=train_loss,
                train_acc=train_acc,
                eval_loss=eval_loss,
                eval_acc=eval_acc,
            )
            rows.append(row)
            print(
                f"{name},{model_seed},{train_loss:.4f},{train_acc:.3f},"
                f"{eval_loss:.4f},{eval_acc:.3f}"
            )

    write_csv(args.out, rows)
    print("saved_csv:", args.out)
    print_summary(rows)


def train_model(
    model: nn.Module,
    sampler: Sampler,
    generator: torch.Generator,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    loss_value = 0.0
    accuracy = 0.0

    for _ in range(steps):
        images, labels = sampler(batch_size, generator=generator, device=device)
        logits = _logits(model(images))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

    return loss_value, accuracy


def evaluate_model(
    model: nn.Module,
    sampler: Sampler,
    generator: torch.Generator,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        images, labels = sampler(batch_size, generator=generator, device=device)
        logits = _logits(model(images))
        loss = F.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def _logits(output: Tensor | dict[str, Tensor | list[Tensor]]) -> Tensor:
    if isinstance(output, dict):
        logits = output["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output['logits'] must be a tensor")
        return logits
    return output


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "model",
                "seed",
                "steps",
                "train_loss",
                "train_acc",
                "eval_loss",
                "eval_acc",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_summary(rows: list[ResultRow]) -> None:
    models = sorted({row.model for row in rows})
    print("summary_model,eval_acc_mean,eval_acc_std")
    for model in models:
        values = [row.eval_acc for row in rows if row.model == model]
        print(f"{model},{mean(values):.3f},{pstdev(values):.3f}")


if __name__ == "__main__":
    main()
