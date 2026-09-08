"""Paired CPU controls for shared, block-diagonal, and grouped lattice reads."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import TinyPrefillLatticeAttnRes, sample_aligned_pair_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("results/grouped_controls.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for repeat in range(args.seeds):
        seed = args.seed + repeat
        torch.manual_seed(seed)
        eval_generator = torch.Generator().manual_seed(seed + 2000)
        eval_images, eval_labels = sample_aligned_pair_batch(
            args.eval_size, generator=eval_generator, device="cpu"
        )
        for mode in ("shared", "blockdiag", "grouped", "grouped_globalnorm", "grouped_fullcontext"):
            torch.manual_seed(seed)
            groups = 1 if mode == "shared" else 2
            model = TinyPrefillLatticeAttnRes(
                embed_dim=args.embed_dim,
                num_classes=2,
                prefill_rounds=1,
                read_blocks=1,
                expert_groups=groups,
                expert_mode=mode,
            )
            train_generator = torch.Generator().manual_seed(seed + 1000)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
            for _ in range(args.steps):
                images, labels = sample_aligned_pair_batch(
                    args.batch_size, generator=train_generator, device="cpu"
                )
                loss = F.cross_entropy(model(images)["logits"], labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                output = model(eval_images)
                eval_loss = F.cross_entropy(output["logits"], eval_labels).item()
                eval_acc = (output["logits"].argmax(1) == eval_labels).float().mean().item()
            gate = model.read_blocks[0].read_gate
            set_uniform_routing(model, True)
            with torch.no_grad():
                uniform_acc = (model(eval_images)["logits"].argmax(1) == eval_labels).float().mean().item()
            set_uniform_routing(model, False)
            original_gate = gate.detach().clone()
            gate.data.zero_()
            with torch.no_grad():
                gate_zero_acc = (model(eval_images)["logits"].argmax(1) == eval_labels).float().mean().item()
            gate.data.copy_(original_gate)
            row = {
                "mode": mode,
                "seed": seed,
                "steps": args.steps,
                "params": sum(parameter.numel() for parameter in model.parameters()),
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "gate": float(original_gate),
                "gate_zero_acc": gate_zero_acc,
                "uniform_acc": uniform_acc,
            }
            rows.append(row)
            print(",".join(str(row[key]) for key in row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv: {args.out}")


def set_uniform_routing(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "uniform_routing"):
            module.uniform_routing = enabled


if __name__ == "__main__":
    main()
