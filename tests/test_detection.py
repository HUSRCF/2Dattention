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
from scripts.train_det_toy import (
    DenseMaskAuxHead,
    ap50_for_image,
    cxcywh_iou,
    dense_mask_aux_loss,
    dense_mask_targets_from_boxes,
    mask_gate_scale,
    match_targets_by_iou,
    query_mask_aux_loss,
    query_mask_aux_metrics,
    sample_square_detection_batch,
    summarize_stratified_iou,
    update_stratified_iou_lists,
)


def test_detection_head_shapes() -> None:
    head = DetectionHead(dim=16, num_classes=3)
    output = head(torch.randn(2, 5, 16))
    assert output["pred_logits"].shape == (2, 5, 4)
    assert output["pred_boxes"].shape == (2, 5, 4)
    assert output["pred_quality_logits"].shape == (2, 5)
    assert float(output["pred_boxes"].detach().min()) >= 0.0
    assert float(output["pred_boxes"].detach().max()) <= 1.0


def test_detection_head_box_quality_mode_shapes() -> None:
    head = DetectionHead(dim=16, num_classes=3, quality_mode="box")
    output = head(torch.randn(2, 5, 16))
    assert output["pred_logits"].shape == (2, 5, 4)
    assert output["pred_boxes"].shape == (2, 5, 4)
    assert output["pred_quality_logits"].shape == (2, 5)


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
    assert outputs["pred_quality_logits"].shape == (2, 5)
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
        for query_init in (
            "learned",
            "anchor",
            "anchor_detached",
            "anchor_residual",
            "anchor_residual_detached",
            "mask_proposal",
            "mask_proposal_nms",
        ):
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
            assert outputs["pred_quality_logits"].shape == (2, 4)
            assert outputs["spatial_features"].ndim == 4
            if query_init == "mask_proposal":
                assert outputs["query_mask_logits"].shape == outputs["spatial_features"].shape[0:1] + outputs[
                    "spatial_features"
                ].shape[2:4]
                assert outputs["query_proposal_indices"].shape == (2, 4)
            routing_maps = outputs["routing_maps"]
            assert isinstance(routing_maps, list)
            assert len(routing_maps) == int(feature_mode == "anchor")


def test_tiny_anchor_region_detr_mask_proposal_query_init() -> None:
    torch.manual_seed(13)
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=1,
        num_queries=4,
        feature_mode="local",
        query_init="mask_proposal",
    )
    criterion = DetectionCriterion(num_classes=1)
    images = torch.randn(2, 3, 32, 32)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.50, 0.50]]),
        },
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        },
    ]
    outputs = model(images)
    assert outputs["pred_logits"].shape == (2, 4, 2)
    assert outputs["query_proposal_indices"].shape == (2, 4)
    losses = dense_mask_aux_loss(outputs, targets, mask_head=None, dice_weight=1.0)
    det_losses = criterion(outputs, targets)
    total_loss = det_losses["loss"] + 0.5 * losses["loss_mask_aux"]
    total_loss.backward()
    assert model.query_mask_head.weight.grad is not None
    assert torch.isfinite(model.query_mask_head.weight.grad).all()
    assert float(model.query_mask_head.weight.grad.detach().abs().sum()) > 0.0


def test_tiny_anchor_region_detr_anchor_residual_query_init() -> None:
    torch.manual_seed(17)
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=1,
        num_queries=4,
        feature_mode="local",
        query_init="anchor_residual",
        query_mask_gate_init=0.01,
    )
    assert torch.allclose(model.anchor_query_gate.detach(), torch.tensor(0.01))
    criterion = DetectionCriterion(num_classes=1)
    images = torch.randn(2, 3, 32, 32)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.50, 0.50]]),
        },
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        },
    ]
    outputs = model(images)
    losses = criterion(outputs, targets)
    losses["loss"].backward()
    assert model.anchor_query_gate.grad is not None
    assert torch.isfinite(model.anchor_query_gate.grad).all()
    assert float(model.anchor_query_gate.grad.detach().abs().sum()) > 0.0


def test_tiny_anchor_region_detr_query_conditioned_mask_aux() -> None:
    torch.manual_seed(18)
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=1,
        num_queries=4,
        feature_mode="local",
        query_init="anchor_residual",
        query_refine="query_mask",
        query_mask_gate_init=0.01,
    )
    criterion = DetectionCriterion(num_classes=1)
    images = torch.randn(2, 3, 32, 32)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.50, 0.50]]),
        },
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        },
    ]
    outputs = model(images)
    assert outputs["query_mask_logits_per_query"].shape == (2, 4, 8, 8)
    mask_losses = query_mask_aux_loss(outputs, targets, criterion, dice_weight=1.0)
    det_losses = criterion(outputs, targets)
    total_loss = det_losses["loss"] + 0.5 * mask_losses["loss_mask_aux"]
    total_loss.backward()
    metrics = query_mask_aux_metrics(outputs, targets, criterion)

    assert torch.isfinite(mask_losses["loss_mask_aux"])
    assert torch.isfinite(metrics["mask_iou"])
    assert model.query_mask_query_proj.weight.grad is not None
    assert model.query_mask_feature_proj.weight.grad is not None
    assert float(model.query_mask_query_proj.weight.grad.detach().abs().sum()) > 0.0


def test_tiny_anchor_region_detr_oracle_mask_proposal_query_init() -> None:
    torch.manual_seed(19)
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=1,
        num_queries=4,
        feature_mode="local",
        query_init="mask_proposal_oracle_nms",
    )
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        feature_shape = model.coord_encoding(model.patch_embed(images)).shape[-2:]
    oracle_logits = torch.full((2, feature_shape[0], feature_shape[1]), -8.0)
    oracle_logits[:, 2:5, 2:5] = 8.0
    outputs = model(images, query_mask_logits_override=oracle_logits)

    assert outputs["pred_logits"].shape == (2, 4, 2)
    assert torch.allclose(outputs["query_mask_logits"], oracle_logits)
    assert outputs["query_proposal_indices"].shape == (2, 4)


def test_tiny_anchor_region_detr_oracle_requires_mask_override() -> None:
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=1,
        num_queries=4,
        feature_mode="local",
        query_init="mask_proposal_oracle_nms",
    )
    images = torch.randn(2, 3, 32, 32)

    try:
        model(images)
    except ValueError as exc:
        assert "query_mask_logits_override" in str(exc)
        return
    raise AssertionError("expected oracle query init to require mask override")


def test_tiny_anchor_region_detr_mask_proposal_nms_diversifies_indices() -> None:
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=1,
        num_queries=4,
        feature_mode="local",
        query_init="mask_proposal_nms",
    )
    images = torch.randn(2, 3, 32, 32)
    outputs = model(images)
    assert outputs["pred_logits"].shape == (2, 4, 2)
    indices = outputs["query_proposal_indices"]
    assert indices.shape == (2, 4)
    width = outputs["query_mask_logits"].shape[-1]
    y = indices // width
    x = indices % width
    for query_idx in range(indices.shape[1]):
        for other_idx in range(query_idx + 1, indices.shape[1]):
            distance = torch.maximum(
                (y[:, query_idx] - y[:, other_idx]).abs(),
                (x[:, query_idx] - x[:, other_idx]).abs(),
            )
            assert bool((distance > 2).all())


def test_tiny_anchor_region_detr_proposal_state_refine_modes() -> None:
    images = torch.randn(2, 3, 32, 32)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.50, 0.50]]),
        },
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        },
    ]
    for mode in (
        "proposal_decode2",
        "proposal_reinject",
        "proposal_persistent",
        "proposal_late_persistent",
        "proposal_persistent_stopgrad",
    ):
        model = TinyAnchorRegionDETR(
            embed_dim=16,
            num_classes=1,
            num_queries=4,
            feature_mode="local",
            query_init="mask_proposal_nms",
            query_refine=mode,
        )
        criterion = DetectionCriterion(num_classes=1)
        outputs = model(images)
        assert outputs["pred_logits"].shape == (2, 4, 2)
        assert outputs["query_proposal_indices"].shape == (2, 4)
        losses = criterion(outputs, targets)
        mask_losses = dense_mask_aux_loss(outputs, targets, mask_head=None, dice_weight=1.0)
        total_loss = losses["loss"] + 0.5 * mask_losses["loss_mask_aux"]
        total_loss.backward()
        if mode != "proposal_decode2":
            assert model.proposal_state_gate.grad is not None
        if mode in {"proposal_persistent", "proposal_late_persistent", "proposal_persistent_stopgrad"}:
            assert model.proposal_state_update[-1].weight.grad is not None


def test_tiny_anchor_region_detr_mask_pool_query_refine() -> None:
    torch.manual_seed(7)
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=1,
        num_queries=4,
        feature_mode="local",
        query_init="anchor",
        query_refine="mask_pool",
    )
    images = torch.randn(2, 3, 32, 32)
    outputs = model(images)
    assert outputs["pred_logits"].shape == (2, 4, 2)
    assert outputs["query_mask_logits"].shape == outputs["spatial_features"].shape[0:1] + outputs[
        "spatial_features"
    ].shape[2:4]
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.50, 0.50]]),
        },
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        },
    ]
    losses = dense_mask_aux_loss(outputs, targets, mask_head=None, dice_weight=1.0)
    losses["loss_mask_aux"].backward()
    assert torch.isfinite(losses["loss_mask_aux"])

    model.zero_grad(set_to_none=True)
    criterion = DetectionCriterion(num_classes=1)
    outputs = model(images)
    det_losses = criterion(outputs, targets)
    det_losses["loss"].backward()
    assert model.query_mask_gate.grad is not None
    assert model.query_mask_proj.weight.grad is not None
    assert torch.isfinite(model.query_mask_gate.grad).all()
    assert torch.isfinite(model.query_mask_proj.weight.grad).all()
    assert float(model.query_mask_gate.grad.detach().abs().sum()) > 0.0
    assert float(model.query_mask_proj.weight.grad.detach().abs().sum()) > 0.0


def test_tiny_anchor_region_detr_mask_biased_attention_refine() -> None:
    torch.manual_seed(11)
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=1,
        num_queries=4,
        feature_mode="local",
        query_init="anchor",
        query_refine="mask_bias",
        query_mask_gate_init=0.01,
    )
    assert torch.allclose(model.query_mask_gate.detach(), torch.tensor(0.01))
    model.set_query_mask_gate_scale(0.25)
    assert torch.allclose(model.query_mask_gate_scale, torch.tensor(0.25))
    criterion = DetectionCriterion(num_classes=1)
    images = torch.randn(2, 3, 32, 32)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.50, 0.50]]),
        },
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        },
    ]
    outputs = model(images)
    assert outputs["pred_logits"].shape == (2, 4, 2)
    assert outputs["query_mask_logits"].shape == outputs["spatial_features"].shape[0:1] + outputs[
        "spatial_features"
    ].shape[2:4]
    mask_losses = dense_mask_aux_loss(outputs, targets, mask_head=None, dice_weight=1.0)
    det_losses = criterion(outputs, targets)
    total_loss = det_losses["loss"] + 0.5 * mask_losses["loss_mask_aux"]
    total_loss.backward()
    assert model.query_mask_gate.grad is not None
    assert model.query_mask_head.weight.grad is not None
    assert torch.isfinite(model.query_mask_gate.grad).all()
    assert torch.isfinite(model.query_mask_head.weight.grad).all()
    assert float(model.query_mask_gate.grad.detach().abs().sum()) > 0.0
    assert float(model.query_mask_head.weight.grad.detach().abs().sum()) > 0.0


def test_mask_gate_scale_schedule() -> None:
    assert mask_gate_scale("none", step=5, total_steps=100) == 1.0
    assert mask_gate_scale("linear", step=0, total_steps=100) == 0.0
    assert mask_gate_scale("linear", step=25, total_steps=100) == 0.25
    assert mask_gate_scale("linear", step=200, total_steps=100) == 1.0


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


def test_stratified_iou_summary_groups_size_and_offset() -> None:
    strata = {
        "small": [],
        "medium": [],
        "large": [],
        "center": [],
        "offcenter": [],
    }
    ious = torch.tensor([0.2, 0.6, 0.8])
    boxes = torch.tensor(
        [
            [0.50, 0.50, 0.10, 0.10],
            [0.80, 0.50, 0.24, 0.24],
            [0.10, 0.10, 0.35, 0.35],
        ]
    )
    update_stratified_iou_lists(strata, ious, boxes)
    metrics = summarize_stratified_iou(strata)
    assert torch.isclose(torch.tensor(metrics["small_iou"]), torch.tensor(0.2))
    assert torch.isclose(torch.tensor(metrics["medium_iou"]), torch.tensor(0.6))
    assert torch.isclose(torch.tensor(metrics["large_iou"]), torch.tensor(0.8))
    assert torch.isclose(torch.tensor(metrics["center_iou"]), torch.tensor(0.2))
    assert torch.isclose(torch.tensor(metrics["offcenter_iou"]), torch.tensor(0.7))
    assert metrics["small_recall50"] == 0.0
    assert metrics["medium_recall50"] == 1.0
    assert metrics["large_recall50"] == 1.0


def test_ap50_for_image_penalizes_false_positive_ordering() -> None:
    target_boxes = torch.tensor([[0.20, 0.20, 0.20, 0.20]])
    pred_boxes = torch.tensor(
        [
            [0.80, 0.80, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20],
        ]
    )
    good_logits = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
    bad_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    assert ap50_for_image(good_logits, pred_boxes, target_boxes) > 0.99
    assert ap50_for_image(bad_logits, pred_boxes, target_boxes) < 0.51


def test_dense_mask_aux_targets_and_backward() -> None:
    model = TinyAnchorRegionDETR(
        embed_dim=16,
        num_classes=1,
        num_queries=4,
        feature_mode="local",
        query_init="anchor",
    )
    mask_head = DenseMaskAuxHead(dim=16)
    images = torch.randn(2, 3, 32, 32)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.50, 0.50]]),
        },
        {
            "labels": torch.tensor([0, 0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.25, 0.25]]),
        },
    ]
    masks = dense_mask_targets_from_boxes(targets, height=4, width=4, device=torch.device("cpu"))
    assert masks.shape == (2, 4, 4)
    assert masks[0].sum() > 0
    outputs = model(images)
    losses = dense_mask_aux_loss(outputs, targets, mask_head, dice_weight=1.0)
    losses["loss_mask_aux"].backward()
    assert torch.isfinite(losses["loss_mask_aux"])
    assert mask_head.proj.weight.grad is not None
