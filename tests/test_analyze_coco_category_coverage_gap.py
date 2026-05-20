from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_coco_category_coverage_gap import analyze_category_coverage_gap


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_category_coverage_gap_separates_global_and_per_category_topk(tmp_path: Path) -> None:
    annotations = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20]}],
        "categories": [{"id": 2, "name": "object"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 20, 20], "score": 0.8},
    ]
    annotation_path = tmp_path / "annotations.json"
    prediction_path = tmp_path / "predictions.json"
    write_json(annotation_path, annotations)
    write_json(prediction_path, predictions)

    rows = analyze_category_coverage_gap(
        annotation_json=annotation_path,
        prediction_json=prediction_path,
        top_ks=(1,),
        iou_thresholds=(0.5,),
        slices=("all",),
    )

    assert rows[0]["global_topk_loc_recall"] == 1.0
    assert rows[0]["global_topk_class_recall"] == 0.0
    assert rows[0]["per_category_topk_class_recall"] == 1.0
    assert rows[0]["ranking_gap"] == 1.0
