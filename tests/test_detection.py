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
    box_cxcywh_to_xyxy,
    box_iou,
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


def test_box_iou_uses_standard_iou() -> None:
    boxes1 = box_cxcywh_to_xyxy(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    boxes2 = box_cxcywh_to_xyxy(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    assert torch.allclose(box_iou(boxes1, boxes2), torch.tensor([[1.0]]))


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
        feature_mode="anchor",
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


def test_tiny_anchor_region_detr_feature_query_modes() -> None:
    images = torch.randn(2, 3, 32, 32)
    for feature_mode in ("local", "anchor"):
        for query_init in ("learned", "anchor"):
            model = TinyAnchorRegionDETR(
                embed_dim=16,
                num_classes=1,
                num_queries=4,
                feature_mode=feature_mode,
                query_init=query_init,
            )
            outputs = model(images)
            assert outputs["pred_logits"].shape == (2, 4, 2)
            assert outputs["pred_boxes"].shape == (2, 4, 4)
            assert outputs["spatial_features"].ndim == 4
            routing_maps = outputs["routing_maps"]
            assert isinstance(routing_maps, list)
            assert len(routing_maps) == int(feature_mode == "anchor")
