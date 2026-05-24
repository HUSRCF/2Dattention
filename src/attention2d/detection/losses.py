"""Losses for tiny DETR-style detection experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou


class DetectionCriterion(nn.Module):
    """DETR-style set criterion for class and box prediction."""

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher | None = None,
        class_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        no_object_weight: float = 0.1,
        semantic_hard_negatives: Mapping[int, Sequence[tuple[int, float]]] | None = None,
        semantic_hard_negative_weight: float = 0.0,
        semantic_hard_negative_margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.semantic_hard_negatives = normalize_hard_negatives(semantic_hard_negatives)
        self.semantic_hard_negative_weight = semantic_hard_negative_weight
        self.semantic_hard_negative_margin = semantic_hard_negative_margin
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        matches = self.matcher(outputs, targets)
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        batch_size, num_queries, _ = logits.shape
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,
            dtype=torch.long,
            device=logits.device,
        )
        matched_pred_boxes = []
        matched_target_boxes = []
        for batch_idx, (src_idx, target_idx) in enumerate(matches):
            if src_idx.numel() == 0:
                continue
            labels = targets[batch_idx]["labels"][target_idx].to(logits.device)
            target_classes[batch_idx, src_idx] = labels
            matched_pred_boxes.append(boxes[batch_idx, src_idx])
            matched_target_boxes.append(targets[batch_idx]["boxes"][target_idx].to(boxes.device))

        loss_ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )
        if matched_pred_boxes:
            pred_boxes = torch.cat(matched_pred_boxes, dim=0)
            target_boxes = torch.cat(matched_target_boxes, dim=0)
            loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction="none").sum() / max(
                1,
                target_boxes.shape[0],
            )
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
            loss_giou = (1.0 - torch.diag(giou)).sum() / max(1, target_boxes.shape[0])
        else:
            loss_bbox = boxes.sum() * 0.0
            loss_giou = boxes.sum() * 0.0
        cardinality_error = cardinality(logits, targets, no_object_index=self.num_classes)
        loss_semantic_hard_negative = semantic_hard_negative_loss(
            logits,
            target_classes,
            hard_negatives=self.semantic_hard_negatives,
            no_object_index=self.num_classes,
            margin=self.semantic_hard_negative_margin,
        )
        total = (
            self.class_weight * loss_ce
            + self.bbox_weight * loss_bbox
            + self.giou_weight * loss_giou
            + self.semantic_hard_negative_weight * loss_semantic_hard_negative
        )
        return {
            "loss": total,
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_semantic_hard_negative": loss_semantic_hard_negative,
            "cardinality_error": cardinality_error,
        }


@torch.no_grad()
def cardinality(
    logits: Tensor,
    targets: list[dict[str, Tensor]],
    no_object_index: int,
) -> Tensor:
    predictions = logits.argmax(dim=-1)
    predicted_counts = (predictions != no_object_index).sum(dim=1)
    target_counts = torch.tensor(
        [len(target["labels"]) for target in targets],
        dtype=predicted_counts.dtype,
        device=predicted_counts.device,
    )
    return F.l1_loss(predicted_counts.float(), target_counts.float())


def normalize_hard_negatives(
    hard_negatives: Mapping[int, Sequence[tuple[int, float]]] | None,
) -> dict[int, list[tuple[int, float]]]:
    """Normalize semantic hard-negative mapping.

    Class ids in this mapping must already match the logits class indices used
    by ``DetectionCriterion``. RF-DETR/COCO category ids are often 1-based, so
    callers should adapt them before passing the mapping here.
    """

    if not hard_negatives:
        return {}
    normalized: dict[int, list[tuple[int, float]]] = {}
    for positive, negatives in hard_negatives.items():
        items = []
        for negative, weight in negatives:
            items.append((int(negative), float(weight)))
        if items:
            normalized[int(positive)] = items
    return normalized


def semantic_hard_negative_loss(
    logits: Tensor,
    target_classes: Tensor,
    *,
    hard_negatives: Mapping[int, Sequence[tuple[int, float]]] | None,
    no_object_index: int,
    margin: float = 0.5,
) -> Tensor:
    """Margin loss that pushes configured hard negatives below matched classes."""

    mapping = normalize_hard_negatives(hard_negatives)
    if not mapping:
        return logits.sum() * 0.0
    losses = []
    class_logits = logits[..., :no_object_index]
    for positive, negatives in mapping.items():
        if positive < 0 or positive >= no_object_index:
            continue
        mask = target_classes == positive
        if not torch.any(mask):
            continue
        positive_logits = class_logits[..., positive][mask]
        for negative, weight in negatives:
            if negative < 0 or negative >= no_object_index or negative == positive:
                continue
            negative_logits = class_logits[..., negative][mask]
            penalty = F.relu(margin + negative_logits - positive_logits)
            losses.append(penalty * float(weight))
    if not losses:
        return logits.sum() * 0.0
    return torch.cat([loss.reshape(-1) for loss in losses]).mean()
