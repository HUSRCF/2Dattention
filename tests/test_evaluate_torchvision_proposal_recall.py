from __future__ import annotations

import torch

from scripts.evaluate_torchvision_proposal_recall import (
    best_iou_per_gt,
    proposal_slice_masks,
    set_rpn_top_n,
)


def test_best_iou_per_gt_returns_best_overlap_per_target() -> None:
    gt_boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    proposals = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 25.0, 25.0],
        ]
    )

    values = best_iou_per_gt(gt_boxes, proposals)

    assert torch.allclose(values, torch.tensor([1.0, 0.25]))


def test_best_iou_per_gt_handles_empty_proposals() -> None:
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

    values = best_iou_per_gt(gt_boxes, torch.empty((0, 4)))

    assert torch.equal(values, torch.tensor([0.0]))


def test_proposal_slice_masks_separate_center_and_size_bins() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [40.0, 40.0, 60.0, 60.0],
            [20.0, 20.0, 80.0, 80.0],
        ]
    )

    masks = proposal_slice_masks(
        boxes,
        resized_size=torch.tensor([100.0, 100.0]),
        center_radius=0.25,
        small_area_ratio=0.05,
        large_area_ratio=0.25,
    )

    assert masks["all"].tolist() == [True, True, True]
    assert masks["offcenter"].tolist() == [True, False, False]
    assert masks["center"].tolist() == [False, True, True]
    assert masks["small"].tolist() == [True, True, False]
    assert masks["medium"].tolist() == [False, False, False]
    assert masks["large"].tolist() == [False, False, True]


def test_set_rpn_top_n_updates_testing_budget_without_lowering_pre_nms() -> None:
    class DummyRPN:
        def __init__(self) -> None:
            self._post_nms_top_n = {"testing": 150}
            self._pre_nms_top_n = {"testing": 1000}

    class DummyModel:
        def __init__(self) -> None:
            self.rpn = DummyRPN()

    model = DummyModel()

    set_rpn_top_n(model, 300)

    assert model.rpn._post_nms_top_n["testing"] == 300
    assert model.rpn._pre_nms_top_n["testing"] == 1000
