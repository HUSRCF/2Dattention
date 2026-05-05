"""Train briefly on a synthetic 2D alignment task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import (  # noqa: E402
    TinyPrefillLatticeAttnRes,
    get_best_device,
    sample_aligned_pair_batch,
    sample_distractor_aligned_pair_batch,
    sample_oriented_pair_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--task",
        choices=("oriented_pair", "aligned_pair", "distractor_aligned_pair"),
        default="oriented_pair",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    generator = torch.Generator().manual_seed(args.seed)

    model = TinyPrefillLatticeAttnRes(
        embed_dim=args.embed_dim,
        num_classes=2,
        prefill_rounds=2,
        read_blocks=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    print("device:", device)
    sampler = {
        "oriented_pair": sample_oriented_pair_batch,
        "aligned_pair": sample_aligned_pair_batch,
        "distractor_aligned_pair": sample_distractor_aligned_pair_batch,
    }[args.task]

    print("task:", args.task)
    print("steps:", args.steps)

    for step in range(1, args.steps + 1):
        images, labels = sampler(
            args.batch_size,
            generator=generator,
            device=device,
        )
        output = model(images)
        logits = output["logits"]
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step == args.steps or step % 10 == 0:
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            print(
                f"step={step:03d}",
                f"loss={loss.item():.4f}",
                f"train_acc={accuracy:.3f}",
            )

    model.eval()
    with torch.no_grad():
        images, labels = sampler(
            args.eval_batch_size,
            generator=generator,
            device=device,
        )
        logits = model(images)["logits"]
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        loss = F.cross_entropy(logits, labels).item()

    print("eval_loss:", f"{loss:.4f}")
    print("eval_acc:", f"{accuracy:.3f}")


if __name__ == "__main__":
    main()
