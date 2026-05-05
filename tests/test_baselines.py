from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import (
    ConvOnlyClassifier,
    MultiCLSViTClassifier,
    SequenceTransformerClassifier,
    TinyAnchorOnlyAttnResClassifier,
    TinyAnchorPrefillLatticeAttnRes,
    TinyGraphPrefillLatticeAttnRes,
    TinyPrefillLocalMixClassifier,
    TinyViTClassifier,
    TinyXAttnResClassifier,
)


def test_baselines_forward_and_backward() -> None:
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])

    for model in (
        ConvOnlyClassifier(embed_dim=16),
        SequenceTransformerClassifier(embed_dim=16),
        TinyViTClassifier(embed_dim=16),
        MultiCLSViTClassifier(embed_dim=16, num_classes=2, global_tokens=3),
        TinyPrefillLocalMixClassifier(
            embed_dim=16,
            num_classes=2,
            prefill_rounds=1,
            read_blocks=1,
        ),
        TinyAnchorOnlyAttnResClassifier(
            embed_dim=16,
            num_classes=2,
            prefill_rounds=1,
            read_blocks=1,
        ),
        TinyXAttnResClassifier(embed_dim=16, num_classes=2, prefill_rounds=1, read_blocks=1),
        TinyAnchorPrefillLatticeAttnRes(
            embed_dim=16,
            num_classes=2,
            prefill_rounds=1,
            read_blocks=1,
        ),
        TinyGraphPrefillLatticeAttnRes(
            embed_dim=16,
            num_classes=2,
            prefill_rounds=1,
            read_blocks=1,
            graph_k=2,
        ),
    ):
        output = model(images)
        logits = output["logits"] if isinstance(output, dict) else output
        assert isinstance(logits, Tensor)
        assert logits.shape == (2, 2)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
