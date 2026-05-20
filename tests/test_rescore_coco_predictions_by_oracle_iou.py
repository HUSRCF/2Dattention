from __future__ import annotations

import json
from pathlib import Path

from scripts.rescore_coco_predictions_by_oracle_iou import rescore_predictions_by_oracle_iou


def test_rescore_predictions_by_oracle_iou_uses_nearest_gt(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.JPEG", "width": 100, "height": 100}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "area": 100},
                    {"id": 2, "image_id": 1, "category_id": 3, "bbox": [20, 20, 10, 10], "area": 100},
                ],
                "categories": [{"id": 2, "name": "a"}, {"id": 3, "name": "b"}],
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps([{"image_id": 1, "category_id": 99, "bbox": [0, 0, 10, 10], "score": 0.01}]),
        encoding="utf-8",
    )

    records = rescore_predictions_by_oracle_iou(annotations, predictions)

    assert records[0]["score"] == 1.0


def test_class_aware_oracle_rescoring_respects_category(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.JPEG", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10]}],
                "categories": [{"id": 2, "name": "a"}],
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps([{"image_id": 1, "category_id": 99, "bbox": [0, 0, 10, 10], "score": 0.01}]),
        encoding="utf-8",
    )

    records = rescore_predictions_by_oracle_iou(annotations, predictions, class_aware=True)

    assert records[0]["score"] == 0.0
