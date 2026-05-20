from __future__ import annotations

from scripts.summarize_coco_bbox_stats import summarize_bbox_stats


def test_summarize_bbox_stats_scales_boxes_to_resolution() -> None:
    data = {
        "images": [{"id": 1, "width": 100, "height": 200}],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [0, 0, 10, 20]},
            {"id": 2, "image_id": 1, "bbox": [0, 0, 50, 100]},
        ],
    }

    rows = summarize_bbox_stats(data, resolutions=[100], small_area_ratio=0.05)

    assert rows[0]["annotations"] == 2
    assert rows[0]["small_fraction"] == 0.5
    assert rows[0]["median_short_px"] == 30.0
    assert rows[0]["median_area_px"] == 1300.0


def test_summarize_bbox_stats_handles_empty_annotations() -> None:
    rows = summarize_bbox_stats({"images": [], "annotations": []}, resolutions=[128])

    assert rows[0]["annotations"] == 0
    assert rows[0]["median_short_px"] == 0.0
