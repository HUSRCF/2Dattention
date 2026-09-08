from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import (  # noqa: E402
    BlockDiagonalLatticeMemoryRead,
    GroupedLatticeMemoryRead,
    TinyPrefillLatticeAttnRes,
    cross_group_decorrelation_loss,
    variance_floor_loss,
)


def test_grouped_read_has_private_routing_and_preserves_shape() -> None:
    reader = GroupedLatticeMemoryRead(dim=8, groups=2)
    memories = [torch.randn(3, 8, 4, 4), torch.randn(3, 8, 4, 4)]

    output, routing, private = reader(memories)

    assert output.shape == (3, 8, 4, 4)
    assert private.shape == (3, 2, 4, 4, 4)
    assert routing.shape[:2] == (3, 2)
    assert torch.allclose(routing.sum(dim=2), torch.ones(3, 2, 4, 4), atol=1e-5)


def test_block_diagonal_read_keeps_shared_routing() -> None:
    reader = BlockDiagonalLatticeMemoryRead(dim=8, groups=2)
    memories = [torch.randn(2, 8, 4, 4)]

    output, routing, private = reader(memories)

    assert output.shape == (2, 8, 4, 4)
    assert routing.shape == (2, 13, 4, 4)
    assert private.shape == (2, 2, 4, 4, 4)
    assert torch.allclose(routing.sum(dim=1), torch.ones(2, 4, 4), atol=1e-5)


def test_grouped_losses_are_finite_and_backpropagate() -> None:
    outputs = torch.randn(4, 2, 3, 4, 4, requires_grad=True)

    loss = cross_group_decorrelation_loss(outputs) + variance_floor_loss(outputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert outputs.grad is not None


def test_grouped_model_exposes_private_readouts() -> None:
    model = TinyPrefillLatticeAttnRes(
        embed_dim=8,
        num_classes=2,
        prefill_rounds=1,
        read_blocks=1,
        expert_groups=2,
    )
    result = model(torch.randn(2, 3, 32, 32))

    assert result["logits"].shape == (2, 2)
    assert len(result["expert_readouts"]) == 1
    assert result["expert_readouts"][0].shape[1] == 2


@pytest.mark.parametrize("dim, groups", [(7, 2), (8, 1)])
def test_grouped_read_rejects_invalid_partition(dim: int, groups: int) -> None:
    with pytest.raises(ValueError):
        GroupedLatticeMemoryRead(dim=dim, groups=groups)
