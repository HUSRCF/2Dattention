from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import (
    TinyPrefillLatticeAttnRes,
    sample_aligned_pair_batch,
    sample_distractor_aligned_pair_batch,
    sample_oriented_pair_batch,
)


def test_aligned_pair_batch_is_balanced_and_shaped() -> None:
    generator = torch.Generator().manual_seed(5)
    images, labels = sample_aligned_pair_batch(12, generator=generator)

    assert images.shape == (12, 3, 32, 32)
    assert labels.shape == (12,)
    assert labels.sum().item() == 6
    assert images[:, 0].amax().item() == 1.0
    assert images[:, 1].amax().item() == 1.0


def test_tiny_model_learns_one_toy_step() -> None:
    generator = torch.Generator().manual_seed(9)
    model = TinyPrefillLatticeAttnRes(
        embed_dim=24,
        num_classes=2,
        prefill_rounds=1,
        read_blocks=1,
    )
    images, labels = sample_aligned_pair_batch(8, generator=generator)
    logits = model(images)["logits"]

    assert logits.shape == (8, 2)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert model.head[-1].weight.grad is not None


def test_oriented_pair_batch_is_balanced_and_shaped() -> None:
    generator = torch.Generator().manual_seed(13)
    images, labels = sample_oriented_pair_batch(10, generator=generator)

    assert images.shape == (10, 3, 32, 32)
    assert labels.shape == (10,)
    assert labels.sum().item() == 5
    assert images[:, 0].amax().item() == 1.0
    assert images[:, 1].amax().item() == 1.0


def test_distractor_aligned_pair_batch_is_balanced_and_shaped() -> None:
    generator = torch.Generator().manual_seed(17)
    images, labels = sample_distractor_aligned_pair_batch(10, generator=generator)

    assert images.shape == (10, 3, 32, 32)
    assert labels.shape == (10,)
    assert labels.sum().item() == 5
    assert images[:, 0].amax().item() == 1.0
    assert images[:, 1].amax().item() == 1.0
    assert images[:, 2].amax().item() >= 0.45
