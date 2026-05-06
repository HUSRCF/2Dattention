"""Tiny DETR-style detection components for anchor-region experiments."""

from .anchor_region_detr import TinyAnchorRegionDETR, anchor_queries_from_state
from .heads import DetectionHead, MLP
from .losses import DetectionCriterion, generalized_box_iou
from .matcher import HungarianMatcher, box_cxcywh_to_xyxy

__all__ = [
    "DetectionCriterion",
    "DetectionHead",
    "HungarianMatcher",
    "MLP",
    "TinyAnchorRegionDETR",
    "anchor_queries_from_state",
    "box_cxcywh_to_xyxy",
    "generalized_box_iou",
]
