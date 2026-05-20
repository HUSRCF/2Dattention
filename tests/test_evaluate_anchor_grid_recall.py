from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.evaluate_anchor_grid_recall import (
    evaluate_anchor_grid_recall,
    generate_anchor_grid,
    load_scaled_boxes,
)


def test_generate_anchor_grid_produces_clipped_xyxy_boxes() -> None:
    anchors = generate_anchor_grid(image_size=16, strides=(8,), scales=(16,), aspect_ratios=(1.0,))

    assert anchors.shape == (4, 4)
    assert torch.all(anchors[:, :2] >= 0)
    assert torch.all(anchors[:, 2:] <= 16)
    assert torch.all(anchors[:, 2] > anchors[:, 0])
    assert torch.all(anchors[:, 3] > anchors[:, 1])


def test_load_scaled_boxes_converts_coco_xywh_to_resized_xyxy(tmp_path: Path) -> None:
    annotation_json = tmp_path / "ann.json"
    annotation_json.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.JPEG", "width": 200, "height": 100}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 25, 100, 50]}
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
        ),
        encoding="utf-8",
    )

    gt_by_image, image_sizes = load_scaled_boxes(annotation_json, image_size=100)

    assert torch.allclose(gt_by_image[1], torch.tensor([[25.0, 25.0, 75.0, 75.0]]))
    assert torch.equal(image_sizes[1], torch.tensor([100.0, 100.0]))


def test_evaluate_anchor_grid_recall_reports_slice_recall() -> None:
    gt_by_image = {
        1: torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [40.0, 40.0, 60.0, 60.0],
            ]
        )
    }
    anchors = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [40.0, 40.0, 60.0, 60.0],
        ]
    )

    rows = evaluate_anchor_grid_recall(
        gt_by_image=gt_by_image,
        image_sizes={1: torch.tensor([100.0, 100.0])},
        anchors=anchors,
        iou_thresholds=(0.5,),
        center_radius=0.25,
        small_area_ratio=0.05,
        large_area_ratio=0.25,
    )

    by_slice = {row["slice"]: row for row in rows}
    assert by_slice["all"]["gt_count"] == 2
    assert by_slice["all"]["recall"] == 1.0
    assert by_slice["offcenter"]["gt_count"] == 1
    assert by_slice["center"]["gt_count"] == 1
