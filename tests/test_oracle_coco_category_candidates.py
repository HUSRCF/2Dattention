from __future__ import annotations

import json
from pathlib import Path

from scripts.oracle_coco_category_candidates import oracle_category_candidates


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_candidate_oracle_emits_only_when_nearest_gt_category_is_candidate(tmp_path: Path) -> None:
    annotations = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10]}],
        "categories": [{"id": 7, "name": "target"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10], "score": 0.2},
        {"image_id": 1, "category_id": 4, "bbox": [50, 50, 10, 10], "score": 0.8},
    ]
    annotation_path = tmp_path / "annotations.json"
    prediction_path = tmp_path / "predictions.json"
    write_json(annotation_path, annotations)
    write_json(prediction_path, predictions)

    rows, summary = oracle_category_candidates(
        annotation_json=annotation_path,
        prediction_json=prediction_path,
        score_mode="group_max",
    )

    assert rows == [{"image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10], "score": 0.9}]
    assert summary["groups"] == 2
    assert summary["candidate_hits"] == 1
    assert summary["candidate_hit_rate"] == 0.5


def test_candidate_oracle_can_keep_missing_groups(tmp_path: Path) -> None:
    annotations = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 10, 10]}],
        "categories": [{"id": 7, "name": "target"}],
    }
    predictions = [{"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9}]
    annotation_path = tmp_path / "annotations.json"
    prediction_path = tmp_path / "predictions.json"
    write_json(annotation_path, annotations)
    write_json(prediction_path, predictions)

    rows, summary = oracle_category_candidates(
        annotation_json=annotation_path,
        prediction_json=prediction_path,
        keep_missing=True,
    )

    assert rows == predictions
    assert summary["candidate_hits"] == 0
