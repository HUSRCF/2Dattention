from __future__ import annotations

from scripts.remap_crop_predictions_to_source_coco import remap_crop_predictions


def test_remap_crop_predictions_adds_crop_offset_and_source_image_id() -> None:
    crop_annotations = {
        "images": [{"id": 7, "source_image_id": 99, "crop_box": [10, 20, 50, 60]}],
    }
    crop_predictions = [
        {"image_id": 7, "category_id": 3, "bbox": [1, 2, 4, 5], "score": 0.9},
        {"image_id": 8, "category_id": 3, "bbox": [1, 2, 4, 5], "score": 0.1},
    ]

    remapped = remap_crop_predictions(crop_annotations, crop_predictions)

    assert remapped == [
        {
            "image_id": 99,
            "category_id": 3,
            "bbox": [11.0, 22.0, 4.0, 5.0],
            "score": 0.9,
        }
    ]
