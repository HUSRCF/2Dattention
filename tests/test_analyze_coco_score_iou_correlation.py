from __future__ import annotations

import json

from pathlib import Path

from scripts.analyze_coco_score_iou_correlation import (
    analyze_score_iou,
    pearson,
    ranks,
    spearman,
    xyxy_iou,
)


def test_xyxy_iou() -> None:
    assert xyxy_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert xyxy_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0


def test_ranks_average_ties() -> None:
    assert ranks([2.0, 1.0, 1.0, 3.0]) == [2.0, 0.5, 0.5, 3.0]


def test_correlations_are_positive_for_aligned_scores() -> None:
    xs = [0.1, 0.2, 0.3]
    ys = [0.2, 0.3, 0.4]

    assert pearson(xs, ys) > 0.99
    assert spearman(xs, ys) > 0.99


def test_analyze_score_iou(tmp_path: Path) -> None:
    annotations = {
        "images": [{"id": 1}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10]},
        ],
        "categories": [{"id": 2, "name": "object"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [20, 20, 5, 5], "score": 0.1},
    ]
    annotation_path = tmp_path / "annotations.json"
    prediction_path = tmp_path / "predictions.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")

    row = analyze_score_iou(annotation_path, prediction_path)

    assert row["predictions"] == 2
    assert row["images"] == 1
    assert row["mean_nearest_iou"] == 0.5
    assert row["top10_mean_iou"] == 0.5
    assert row["spearman_score_iou"] > 0.99
