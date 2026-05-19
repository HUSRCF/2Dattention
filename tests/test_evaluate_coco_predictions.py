from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_coco_predictions import evaluate_coco_predictions


pytest.importorskip("pycocotools")


def write_tiny_coco(path: Path) -> None:
    data = {
        "info": {"description": "tiny test"},
        "licenses": [],
        "images": [{"id": 1, "file_name": "sample.JPEG", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10.0, 12.0, 20.0, 16.0],
                "area": 320.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 2, "name": "class_a"}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_evaluate_coco_predictions_scores_perfect_prediction(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    write_tiny_coco(annotations)
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [10.0, 12.0, 20.0, 16.0],
                    "score": 0.99,
                }
            ]
        ),
        encoding="utf-8",
    )

    metrics = evaluate_coco_predictions(annotations, predictions)

    assert metrics["ap50"] > 0.99
    assert metrics["ap"] > 0.99


def test_evaluate_coco_predictions_handles_empty_predictions(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    write_tiny_coco(annotations)
    predictions.write_text("[]", encoding="utf-8")

    metrics = evaluate_coco_predictions(annotations, predictions)

    assert all(value == 0.0 for value in metrics.values())


def test_evaluate_coco_predictions_accepts_exporter_annotations_without_info(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    data = {
        "images": [{"id": 1, "file_name": "sample.JPEG", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10.0, 12.0, 20.0, 16.0],
                "area": 320.0,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 2, "name": "class_a"}],
    }
    annotations.write_text(json.dumps(data), encoding="utf-8")
    predictions.write_text(
        json.dumps([{"image_id": 1, "category_id": 2, "bbox": [10.0, 12.0, 20.0, 16.0], "score": 0.99}]),
        encoding="utf-8",
    )

    metrics = evaluate_coco_predictions(annotations, predictions)

    assert metrics["ap50"] > 0.99
