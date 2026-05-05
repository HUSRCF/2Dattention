from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import TinyPrefillLatticeAttnRes


def test_tiny_model_shapes_and_backward() -> None:
    torch.manual_seed(3)
    model = TinyPrefillLatticeAttnRes(
        embed_dim=32,
        num_classes=5,
        prefill_rounds=2,
        read_blocks=2,
    )
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 3])

    output = model(images)
    logits = output["logits"]
    memories = output["memories"]
    routing_maps = output["routing_maps"]

    assert logits.shape == (2, 5)
    assert len(memories) == 5
    assert all(memory.shape == (2, 32, 8, 8) for memory in memories)
    assert len(routing_maps) == 2
    assert routing_maps[0].shape == (2, 39, 8, 8)
    assert routing_maps[1].shape == (2, 52, 8, 8)
    assert torch.allclose(
        routing_maps[0].sum(dim=1),
        torch.ones(2, 8, 8),
        atol=1e-5,
    )

    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert model.head[-1].weight.grad is not None
