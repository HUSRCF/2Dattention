"""Run a CPU-only shape and gradient smoke test for the tiny 2D model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import TinyPrefillLatticeAttnRes, get_best_device  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    device = get_best_device()

    batch_size = 2
    num_classes = 10
    images = torch.randn(batch_size, 3, 32, 32, device=device)
    targets = torch.tensor([1, 4], device=device)

    model = TinyPrefillLatticeAttnRes(num_classes=num_classes).to(device)
    output = model(images)
    logits = output["logits"]
    memories = output["memories"]
    routing_maps = output["routing_maps"]

    loss = F.cross_entropy(logits, targets)
    loss.backward()

    first_routing = routing_maps[0]
    routing_sum_error = (first_routing.sum(dim=1) - 1.0).abs().max().item()

    print("device:", device)
    print("mps_built:", torch.backends.mps.is_built())
    print("mps_available:", torch.backends.mps.is_available())
    print("input_shape:", tuple(images.shape))
    print("logits_shape:", tuple(logits.shape))
    print("num_memories:", len(memories))
    print("memory_shapes:", [tuple(memory.shape) for memory in memories])
    print("routing_shapes:", [tuple(routing.shape) for routing in routing_maps])
    print("routing_sum_max_error:", f"{routing_sum_error:.6f}")
    print("loss:", f"{loss.item():.6f}")
    print("backward_ok:", True)


if __name__ == "__main__":
    main()
