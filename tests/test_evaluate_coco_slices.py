from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_coco_slices import evaluate_coco_slices


pytest.importorskip("pycocotools")


def test_evaluate_coco_slices_reports_slice_counts_and_metrics(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    data = {
        "images": [
            {"id": 1, "file_name": "center.JPEG", "width": 100, "height": 100},
            {"id": 2, "file_name": "offcenter.JPEG", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 40, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    annotations.write_text(json.dumps(data), encoding="utf-8")
    predictions.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 1, "bbox": [40, 40, 20, 20], "score": 0.99},
                {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.98},
            ]
        ),
        encoding="utf-8",
    )

    rows = evaluate_coco_slices(
        annotations,
        predictions,
        slices=("all", "offcenter", "center"),
        center_radius=0.25,
    )

    by_slice = {row["slice"]: row for row in rows}
    assert by_slice["all"]["annotations"] == 2
    assert by_slice["offcenter"]["annotations"] == 1
    assert by_slice["center"]["annotations"] == 1
    assert by_slice["all"]["ap50"] > 0.99
    assert by_slice["offcenter"]["ap50"] > 0.99
    assert by_slice["center"]["ap50"] > 0.99


def test_evaluate_coco_slices_handles_empty_slice(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    data = {
        "images": [{"id": 1, "file_name": "center.JPEG", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 40, 20, 20], "area": 400, "iscrowd": 0}
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    annotations.write_text(json.dumps(data), encoding="utf-8")
    predictions.write_text(
        json.dumps([{"image_id": 1, "category_id": 1, "bbox": [40, 40, 20, 20], "score": 0.99}]),
        encoding="utf-8",
    )

    rows = evaluate_coco_slices(annotations, predictions, slices=("offcenter",), center_radius=0.25)

    assert rows == [
        {
            "slice": "offcenter",
            "images": 0,
            "annotations": 0,
            "ap": 0.0,
            "ap50": 0.0,
            "ap75": 0.0,
            "ap_small": 0.0,
            "ap_medium": 0.0,
            "ap_large": 0.0,
            "ar1": 0.0,
            "ar10": 0.0,
            "ar100": 0.0,
        }
    ]


def test_evaluate_coco_slices_keeps_image_level_predictions_as_slice_false_positives(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    data = {
        "images": [{"id": 1, "file_name": "mixed.JPEG", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 40, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    annotations.write_text(json.dumps(data), encoding="utf-8")
    predictions.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 1, "bbox": [40, 40, 20, 20], "score": 0.99},
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.98},
            ]
        ),
        encoding="utf-8",
    )

    rows = evaluate_coco_slices(annotations, predictions, slices=("offcenter",), center_radius=0.25)

    assert rows[0]["annotations"] == 1
    assert 0.0 < rows[0]["ap50"] < 1.0
