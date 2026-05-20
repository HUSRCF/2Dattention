from __future__ import annotations

import json
from pathlib import Path

from scripts.relabel_coco_predictions_by_image_prior import (
    image_category_prior,
    relabel_predictions_by_image_prior,
)


def test_image_category_prior_uses_largest_area() -> None:
    data = {
        "annotations": [
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "area": 100},
            {"image_id": 1, "category_id": 5, "bbox": [0, 0, 20, 20], "area": 400},
        ]
    }

    assert image_category_prior(data, prior="largest") == {1: 5}


def test_image_category_prior_uses_most_frequent_with_area_tie_break() -> None:
    data = {
        "annotations": [
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "area": 100},
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 2, 2], "area": 4},
            {"image_id": 1, "category_id": 5, "bbox": [0, 0, 20, 20], "area": 400},
        ]
    }

    assert image_category_prior(data, prior="most_frequent") == {1: 3}


def test_relabel_predictions_by_image_prior(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    annotations.write_text(
        json.dumps(
            {
                "annotations": [
                    {"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10], "area": 100}
                ]
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 99, "bbox": [1, 2, 3, 4], "score": 0.5},
                {"image_id": 2, "category_id": 88, "bbox": [1, 2, 3, 4], "score": 0.4},
            ]
        ),
        encoding="utf-8",
    )

    relabeled = relabel_predictions_by_image_prior(
        annotation_json=annotations,
        prediction_json=predictions,
    )

    assert relabeled == [
        {"image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4], "score": 0.5},
        {"image_id": 2, "category_id": 88, "bbox": [1, 2, 3, 4], "score": 0.4},
    ]
