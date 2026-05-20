from __future__ import annotations

import json
from pathlib import Path

from scripts.calibrate_coco_prediction_scores import calibrate_prediction_scores, prediction_rank_features


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_prediction_rank_features_are_fractional_within_image_and_category() -> None:
    predictions = [
        {"image_id": 1, "category_id": 2, "score": 0.9},
        {"image_id": 1, "category_id": 2, "score": 0.3},
        {"image_id": 1, "category_id": 3, "score": 0.6},
    ]

    ranks = prediction_rank_features(predictions)

    assert ranks == [(0.0, 0.0), (1.0, 1.0), (0.5, 0.0)]


def test_calibrate_prediction_scores_filters_apply_images_and_preserves_original_score(tmp_path: Path) -> None:
    train_annotations = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20]},
        ],
        "categories": [{"id": 2, "name": "object"}],
    }
    apply_annotations = {
        "images": [{"id": 2, "width": 100, "height": 100}],
        "annotations": [
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [0, 0, 20, 20]},
        ],
        "categories": [{"id": 2, "name": "object"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20], "score": 0.2},
        {"image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.9},
        {"image_id": 2, "category_id": 2, "bbox": [0, 0, 20, 20], "score": 0.2},
        {"image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.9},
    ]
    train_path = tmp_path / "train.json"
    apply_path = tmp_path / "apply.json"
    prediction_path = tmp_path / "predictions.json"
    write_json(train_path, train_annotations)
    write_json(apply_path, apply_annotations)
    write_json(prediction_path, predictions)

    rows, summary = calibrate_prediction_scores(
        train_path,
        apply_path,
        prediction_path,
        class_aware=True,
        score_mode="replace",
    )

    assert [row["image_id"] for row in rows] == [2, 2]
    assert all("original_score" in row for row in rows)
    assert rows[0]["score"] > rows[1]["score"]
    assert summary["train_predictions"] == 2
    assert summary["apply_predictions"] == 2


def test_class_aware_calibration_treats_wrong_category_as_zero_target(tmp_path: Path) -> None:
    annotations = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20]},
        ],
        "categories": [{"id": 2, "name": "object"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20], "score": 0.5},
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 20, 20], "score": 0.5},
    ]
    annotation_path = tmp_path / "annotations.json"
    prediction_path = tmp_path / "predictions.json"
    write_json(annotation_path, annotations)
    write_json(prediction_path, predictions)

    rows, _ = calibrate_prediction_scores(
        annotation_path,
        annotation_path,
        prediction_path,
        class_aware=True,
        score_mode="replace",
        category_weight=1.0,
        category_smoothing=0.0,
    )

    assert rows[0]["score"] > 0.9
    assert rows[1]["score"] < 0.1
