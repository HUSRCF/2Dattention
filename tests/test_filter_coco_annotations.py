from __future__ import annotations

import json
from pathlib import Path

from scripts.filter_coco_annotations import (
    bbox_area_ratio,
    bbox_center_distance,
    filter_coco_annotations,
)


def write_slice_fixture(path: Path) -> None:
    data = {
        "images": [
            {"id": 1, "file_name": "center.JPEG", "width": 100, "height": 100},
            {"id": 2, "file_name": "offcenter.JPEG", "width": 100, "height": 100},
            {"id": 3, "file_name": "large.JPEG", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 40, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [10, 10, 60, 60], "area": 3600, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_bbox_slice_helpers_compute_normalized_geometry() -> None:
    image = {"width": 100, "height": 100}
    annotation = {"bbox": [0, 0, 10, 10]}

    assert bbox_area_ratio(annotation, image) == 0.01
    assert round(bbox_center_distance(annotation, image), 3) == 0.636


def test_filter_coco_annotations_offcenter_slice(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    write_slice_fixture(path)

    filtered = filter_coco_annotations(path, slice_name="offcenter", center_radius=0.25)

    assert [annotation["id"] for annotation in filtered["annotations"]] == [2]
    assert [image["id"] for image in filtered["images"]] == [2]


def test_filter_coco_annotations_area_slices(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    write_slice_fixture(path)

    small = filter_coco_annotations(path, slice_name="small", small_area_ratio=0.05)
    medium = filter_coco_annotations(path, slice_name="medium", small_area_ratio=0.05, large_area_ratio=0.25)
    large = filter_coco_annotations(path, slice_name="large", large_area_ratio=0.25)

    assert [annotation["id"] for annotation in small["annotations"]] == [1, 2]
    assert [annotation["id"] for annotation in medium["annotations"]] == []
    assert [annotation["id"] for annotation in large["annotations"]] == [3]
