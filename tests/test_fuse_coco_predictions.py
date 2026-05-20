from __future__ import annotations

from scripts.fuse_coco_predictions import fuse_coco_predictions, xywh_iou


def test_xywh_iou() -> None:
    assert xywh_iou([0, 0, 10, 10], [5, 0, 10, 10]) == 1 / 3


def test_fuse_coco_predictions_filters_and_nms_per_category() -> None:
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 3, "bbox": [1, 1, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 4, "bbox": [1, 1, 10, 10], "score": 0.7},
        {"image_id": 1, "category_id": 3, "bbox": [50, 50, 5, 5], "score": 0.1},
    ]

    fused = fuse_coco_predictions(predictions, score_threshold=0.2, iou_threshold=0.5)

    assert fused == [
        {"image_id": 1, "category_id": 3, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9},
        {"image_id": 1, "category_id": 4, "bbox": [1.0, 1.0, 10.0, 10.0], "score": 0.7},
    ]


def test_fuse_coco_predictions_can_be_class_agnostic() -> None:
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 4, "bbox": [1, 1, 10, 10], "score": 0.8},
    ]

    fused = fuse_coco_predictions(predictions, iou_threshold=0.5, class_agnostic=True)

    assert fused == [{"image_id": 1, "category_id": 3, "bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.9}]
