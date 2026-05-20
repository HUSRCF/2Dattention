from __future__ import annotations

from scripts.evaluate_coco_category_id_transforms import (
    annotation_category_ids,
    annotation_image_ids,
    filter_predictions_to_annotation_images,
    prediction_category_coverage,
    shift_prediction_categories,
)


def test_annotation_category_ids_reads_coco_categories() -> None:
    assert annotation_category_ids({"categories": [{"id": "2"}, {"id": 5}]}) == {2, 5}


def test_filter_predictions_to_annotation_images_keeps_eval_images_only() -> None:
    predictions = [
        {"image_id": 1, "category_id": 2},
        {"image_id": 99, "category_id": 2},
    ]

    assert annotation_image_ids({"images": [{"id": "1"}]}) == {1}
    assert filter_predictions_to_annotation_images(predictions, {1}) == [{"image_id": 1, "category_id": 2}]


def test_shift_prediction_categories_can_drop_invalid_categories() -> None:
    predictions = [
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1], "score": 0.9},
        {"image_id": 1, "category_id": 9, "bbox": [0, 0, 1, 1], "score": 0.8},
    ]

    shifted = shift_prediction_categories(predictions, offset=-1, valid_categories={1, 2})

    assert shifted == [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.9}]


def test_prediction_category_coverage_counts_valid_prediction_categories() -> None:
    predictions = [
        {"category_id": 1},
        {"category_id": 2},
        {"category_id": 200},
    ]

    coverage = prediction_category_coverage(predictions, {1, 2, 3})

    assert coverage == {
        "predictions": 3,
        "in_gt_category_predictions": 2,
        "in_gt_category_fraction": 2 / 3,
    }
