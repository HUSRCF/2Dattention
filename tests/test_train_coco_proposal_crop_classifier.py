from __future__ import annotations

from PIL import Image

from scripts.train_coco_proposal_crop_classifier import (
    build_crop_samples,
    build_matched_prediction_crop_samples,
    crop_image,
)


def test_build_crop_samples_uses_annotation_boxes() -> None:
    data = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4]}],
    }

    samples = build_crop_samples(data, category_to_index={7: 0})

    assert len(samples) == 1
    assert samples[0].file_name == "a.jpg"
    assert samples[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert samples[0].label_index == 0


def test_crop_image_clamps_to_valid_region() -> None:
    image = Image.new("RGB", (10, 8), color=(255, 255, 255))

    crop = crop_image(image, (-5, -5, 20, 20))

    assert crop.size == (10, 8)


def test_crop_image_keeps_at_least_one_pixel() -> None:
    image = Image.new("RGB", (10, 8), color=(255, 255, 255))

    crop = crop_image(image, (9, 7, 0, 0))

    assert crop.size == (1, 1)


def test_build_matched_prediction_crop_samples_uses_nearest_gt_label() -> None:
    data = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10]},
            {"image_id": 1, "category_id": 8, "bbox": [50, 50, 10, 10]},
        ],
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [49, 49, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [20, 20, 5, 5], "score": 0.8},
    ]

    samples = build_matched_prediction_crop_samples(
        data,
        predictions=predictions,
        category_to_index={7: 0, 8: 1},
        iou_threshold=0.5,
    )

    assert len(samples) == 1
    assert samples[0].bbox == (49.0, 49.0, 10.0, 10.0)
    assert samples[0].label_index == 1
