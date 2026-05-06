"""Small Hungarian-style matcher for DETR sanity experiments."""

from __future__ import annotations

from itertools import combinations, permutations

import torch
from torch import Tensor, nn


class HungarianMatcher(nn.Module):
    """Match predicted queries to targets using class, box L1, and GIoU costs.

    This implementation intentionally avoids an external SciPy dependency. It is
    suitable for tiny research sanity checks with a small number of queries.
    """

    def __init__(
        self,
        class_cost: float = 1.0,
        bbox_cost: float = 5.0,
        giou_cost: float = 2.0,
    ) -> None:
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[Tensor, Tensor]]:
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        probabilities = logits.softmax(dim=-1)
        matches = []
        for batch_idx, target in enumerate(targets):
            labels = target["labels"]
            target_boxes = target["boxes"]
            if labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=logits.device)
                matches.append((empty, empty))
                continue
            class_cost = -probabilities[batch_idx][:, labels]
            bbox_cost = torch.cdist(boxes[batch_idx], target_boxes, p=1)
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(boxes[batch_idx]),
                box_cxcywh_to_xyxy(target_boxes),
            )
            total_cost = (
                self.class_cost * class_cost
                + self.bbox_cost * bbox_cost
                + self.giou_cost * giou_cost
            )
            matches.append(min_cost_assignment(total_cost))
        return matches


def min_cost_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Return a minimum-cost one-to-one assignment for small cost matrices."""

    num_queries, num_targets = cost.shape
    if num_targets > num_queries:
        raise ValueError("num_targets cannot exceed num_queries for this matcher")
    device = cost.device
    cpu_cost = cost.detach().cpu()
    best_cost = float("inf")
    best_queries: tuple[int, ...] | None = None
    best_targets: tuple[int, ...] | None = None
    target_ids = tuple(range(num_targets))
    for query_subset in combinations(range(num_queries), num_targets):
        for query_perm in permutations(query_subset):
            value = float(cpu_cost[list(query_perm), list(target_ids)].sum().item())
            if value < best_cost:
                best_cost = value
                best_queries = query_perm
                best_targets = target_ids
    if best_queries is None or best_targets is None:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return (
        torch.tensor(best_queries, dtype=torch.long, device=device),
        torch.tensor(best_targets, dtype=torch.long, device=device),
    )


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    half_width = width / 2
    half_height = height / 2
    return torch.stack(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        dim=-1,
    )


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_wh = (right_bottom - left_top).clamp(min=0)
    intersection = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    union = area1[:, None] + area2 - intersection
    iou = intersection / union.clamp_min(1e-8)

    enclosing_left_top = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    enclosing_right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    enclosing_wh = (enclosing_right_bottom - enclosing_left_top).clamp(min=0)
    enclosing_area = enclosing_wh[:, :, 0] * enclosing_wh[:, :, 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-8)
