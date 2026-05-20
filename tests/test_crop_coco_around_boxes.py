from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.crop_coco_around_boxes import clip_bbox_to_crop, compute_context_crop, crop_coco_around_boxes


def test_compute_context_crop_keeps_square_inside_image() -> None:
    crop = compute_context_crop([90, 90, 5, 5], image_width=100, image_height=100, crop_scale=6, min_crop_size=40)

    assert crop == (60.0, 60.0, 100.0, 100.0)


def test_clip_bbox_to_crop_offsets_and_clips() -> None:
    clipped = clip_bbox_to_crop([10, 10, 20, 20], (20, 20, 50, 50))

    assert clipped == [0.0, 0.0, 10.0, 10.0]


def test_crop_coco_around_boxes_writes_crops_and_annotations(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (100, 100), "white").save(image_root / "sample.jpg")
    annotations = tmp_path / "ann.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [45, 45, 5, 5], "area": 25}
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
        ),
        encoding="utf-8",
    )

    cropped = crop_coco_around_boxes(
        annotations=annotations,
        image_root=image_root,
        out_image_dir=tmp_path / "crops",
        slice_name="small",
        crop_scale=4,
        min_crop_size=40,
        file_prefix="traincrop",
    )

    assert len(cropped["images"]) == 1
    assert len(cropped["annotations"]) == 1
    assert cropped["annotations"][0]["bbox"] == [17.5, 17.5, 5.0, 5.0]
    assert cropped["images"][0]["file_name"].startswith("traincrop_")
    assert cropped["images"][0]["source_image_id"] == 1
    assert cropped["images"][0]["crop_box"] == [27.5, 27.5, 67.5, 67.5]
    assert (tmp_path / "crops" / cropped["images"][0]["file_name"]).exists()


def test_crop_coco_around_boxes_limits_crops_per_image(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (100, 100), "white").save(image_root / "sample.jpg")
    annotations = tmp_path / "ann.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 5, 5], "area": 25},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 5, 5], "area": 25},
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
        ),
        encoding="utf-8",
    )

    cropped = crop_coco_around_boxes(
        annotations=annotations,
        image_root=image_root,
        out_image_dir=tmp_path / "crops",
        slice_name="small",
        max_crops_per_image=1,
    )

    assert len(cropped["images"]) == 1
