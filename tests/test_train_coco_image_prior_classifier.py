from __future__ import annotations

from scripts.train_coco_image_prior_classifier import (
    build_samples,
    normalized_category_rows,
    relabel_predictions,
)


def test_build_samples_uses_category_index_and_file_name() -> None:
    data = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 7, "bbox": [0, 0, 5, 5], "area": 25}],
    }

    samples = build_samples(data, prior="largest", category_to_index={7: 3})

    assert len(samples) == 1
    assert samples[0].image_id == 1
    assert samples[0].file_name == "a.jpg"
    assert samples[0].label_index == 3


def test_relabel_predictions_uses_image_category_mapping() -> None:
    predictions = [
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1], "score": 0.5},
        {"image_id": 2, "category_id": 3, "bbox": [0, 0, 1, 1], "score": 0.4},
    ]

    relabeled = relabel_predictions(predictions, {1: 9})

    assert relabeled == [
        {"image_id": 1, "category_id": 9, "bbox": [0, 0, 1, 1], "score": 0.5},
        {"image_id": 2, "category_id": 3, "bbox": [0, 0, 1, 1], "score": 0.4},
    ]


def test_relabel_predictions_can_expand_topk_with_score_multiply() -> None:
    predictions = [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1], "score": 0.5}]

    relabeled = relabel_predictions(
        predictions,
        {1: [(9, 0.2), (10, 0.1)]},
        top_k=2,
        score_mode="multiply",
    )

    assert relabeled == [
        {"image_id": 1, "category_id": 9, "bbox": [0, 0, 1, 1], "score": 0.1},
        {"image_id": 1, "category_id": 10, "bbox": [0, 0, 1, 1], "score": 0.05},
    ]


def test_normalized_category_rows_accepts_legacy_int_mapping() -> None:
    assert normalized_category_rows(7) == [(7, 1.0)]
