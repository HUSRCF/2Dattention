from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d.detection import (
    DetectionCriterion,
    DetectionHead,
    HungarianMatcher,
    TinyAnchorRegionDETR,
)


def test_detection_head_shapes() -> None:
    head = DetectionHead(dim=16, num_classes=3)
    output = head(torch.randn(2, 5, 16))
    assert output["pred_logits"].shape == (2, 5, 4)
    assert output["pred_boxes"].shape == (2, 5, 4)
    assert float(output["pred_boxes"].detach().min()) >= 0.0
    assert float(output["pred_boxes"].detach().max()) <= 1.0


def test_hungarian_matcher_handles_variable_targets() -> None:
    matcher = HungarianMatcher()
    outputs = {
        "pred_logits": torch.randn(2, 4, 3),
        "pred_boxes": torch.rand(2, 4, 4),
    }
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]),
        },
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
        },
    ]
    matches = matcher(outputs, targets)
    assert len(matches) == 2
    assert matches[0][0].shape == (2,)
    assert matches[1][0].shape == (1,)


def test_detection_criterion_backward() -> None:
    head = DetectionHead(dim=16, num_classes=2)
    criterion = DetectionCriterion(num_classes=2)
    queries = torch.randn(2, 6, 16, requires_grad=True)
    outputs = head(queries)
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.3, 0.2]]),
        },
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.50, 0.50, 0.4, 0.4]]),
        },
    ]
    losses = criterion(outputs, targets)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert queries.grad is not None


def test_tiny_anchor_region_detr_backward() -> None:
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=2,
        num_queries=5,
        query_init="anchor",
    )
    criterion = DetectionCriterion(num_classes=2)
    images = torch.randn(2, 3, 32, 32)
    outputs = model(images)
    assert outputs["pred_logits"].shape == (2, 5, 3)
    assert outputs["pred_boxes"].shape == (2, 5, 4)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.40, 0.40, 0.25, 0.25]]),
        },
        {
            "labels": torch.tensor([1, 0]),
            "boxes": torch.tensor([[0.60, 0.60, 0.20, 0.20], [0.30, 0.70, 0.20, 0.30]]),
        },
    ]
    losses = criterion(outputs, targets)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
