from __future__ import annotations

import numpy as np

from scripts.predict_rfdetr_coco import detections_to_coco_records, infer_model_num_classes, xyxy_to_xywh


class FakeDetections:
    def __init__(self) -> None:
        self.xyxy = np.array([[1, 2, 5, 8], [0, 0, 3, 0], [2, 2, 4, 4]], dtype=float)
        self.confidence = np.array([0.9, 0.8, 0.7], dtype=float)
        self.class_id = np.array([0, 1, 99], dtype=int)


def test_xyxy_to_xywh() -> None:
    assert xyxy_to_xywh(np.array([1, 2, 5, 8])) == [1.0, 2.0, 4.0, 6.0]


def test_detections_to_coco_records_maps_zero_based_classes_and_filters_invalid() -> None:
    records = detections_to_coco_records(FakeDetections(), image_id=42, category_ids=[10, 20])

    assert records == [
        {
            "image_id": 42,
            "category_id": 10,
            "bbox": [1.0, 2.0, 4.0, 6.0],
            "score": 0.9,
        }
    ]


def test_detections_to_coco_records_filters_model_background_even_with_larger_category_map() -> None:
    detections = FakeDetections()
    detections.class_id = np.array([90], dtype=int)
    detections.xyxy = np.array([[1, 1, 3, 3]], dtype=float)
    detections.confidence = np.array([0.8], dtype=float)

    records = detections_to_coco_records(detections, image_id=1, category_ids=list(range(1, 201)), model_num_classes=90)

    assert records == []


def test_infer_model_num_classes_reads_nested_model_args() -> None:
    class Args:
        num_classes = 123

    class ModelContext:
        args = Args()

    class Model:
        model = ModelContext()

    assert infer_model_num_classes(Model(), default=200) == 123
