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
from scripts.train_det_toy import cxcywh_iou, match_targets_by_iou, sample_square_detection_batch


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


def test_square_detection_batch_multi_modes() -> None:
    device = torch.device("cpu")
    for toy_mode in ("single", "multi", "distractor", "multi_distractor"):
        images, targets = sample_square_detection_batch(
            batch_size=4,
            image_size=32,
            device=device,
            toy_mode=toy_mode,
            max_objects=3,
            torch_seed=123,
        )
        assert images.shape == (4, 3, 32, 32)
        for target in targets:
            assert target["boxes"].ndim == 2
            assert target["boxes"].shape[1] == 4
            assert 1 <= target["boxes"].shape[0] <= 3
            assert target["labels"].shape[0] == target["boxes"].shape[0]
            boxes = target["boxes"].tolist()
            for idx, box in enumerate(boxes):
                for other in boxes[idx + 1 :]:
                    assert cxcywh_iou(box, other) <= 0.05


def test_match_targets_by_iou_uses_multiple_queries() -> None:
    pred_boxes = torch.tensor(
        [
            [0.20, 0.20, 0.20, 0.20],
            [0.80, 0.80, 0.20, 0.20],
            [0.50, 0.50, 0.20, 0.20],
        ]
    )
    target_boxes = torch.tensor(
        [
            [0.80, 0.80, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20],
        ]
    )
    matched = match_targets_by_iou(pred_boxes, target_boxes)
    assert matched.shape == (2,)
    assert torch.allclose(matched, torch.ones(2), atol=1e-5)


def test_match_targets_by_iou_rejects_more_targets_than_queries() -> None:
    pred_boxes = torch.rand(2, 4)
    target_boxes = torch.rand(3, 4)
    try:
        match_targets_by_iou(pred_boxes, target_boxes)
    except ValueError:
        return
    raise AssertionError("expected ValueError when targets exceed queries")
