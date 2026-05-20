from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_coco_prediction_recall import evaluate_prediction_recall


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_prediction_recall_can_run_class_agnostic_and_class_aware(tmp_path: Path) -> None:
    annotations = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20]},
        ],
        "categories": [{"id": 2, "name": "object"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.8},
    ]
    annotation_path = tmp_path / "annotations.json"
    prediction_path = tmp_path / "predictions.json"
    write_json(annotation_path, annotations)
    write_json(prediction_path, predictions)

    class_agnostic = evaluate_prediction_recall(
        annotation_path,
        prediction_path,
        top_ks=(1,),
        iou_thresholds=(0.5,),
        slices=("all",),
        class_aware=False,
    )
    class_aware = evaluate_prediction_recall(
        annotation_path,
        prediction_path,
        top_ks=(1,),
        iou_thresholds=(0.5,),
        slices=("all",),
        class_aware=True,
    )

    assert class_agnostic[0]["recall"] == 1.0
    assert class_aware[0]["recall"] == 0.0


def test_prediction_recall_respects_top_k_order(tmp_path: Path) -> None:
    annotations = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20]},
        ],
        "categories": [{"id": 2, "name": "object"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20], "score": 0.8},
    ]
    annotation_path = tmp_path / "annotations.json"
    prediction_path = tmp_path / "predictions.json"
    write_json(annotation_path, annotations)
    write_json(prediction_path, predictions)

    rows = evaluate_prediction_recall(
        annotation_path,
        prediction_path,
        top_ks=(1, 2),
        iou_thresholds=(0.5,),
        slices=("all",),
        class_aware=True,
    )

    assert rows[0]["top_k"] == 1
    assert rows[0]["recall"] == 0.0
    assert rows[1]["top_k"] == 2
    assert rows[1]["recall"] == 1.0
