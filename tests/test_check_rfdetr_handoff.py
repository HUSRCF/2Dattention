from __future__ import annotations

import json
from pathlib import Path

from scripts.check_rfdetr_handoff import coco_split_report, dataset_report


def test_coco_split_report_flags_valid_category_and_files(tmp_path: Path) -> None:
    (tmp_path / "sample.jpg").write_text("x", encoding="utf-8")
    report = coco_split_report(
        {
            "images": [{"id": 1, "file_name": "sample.jpg", "width": 1, "height": 1}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [0, 0, 1, 1]}],
            "categories": [{"id": 3, "name": "object"}],
        },
        tmp_path,
    )

    assert report["ok"] is True
    assert report["images"] == 1
    assert report["categories"] == 1


def test_dataset_report_reads_rfdetr_splits(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        (split_dir / "sample.jpg").write_text("x", encoding="utf-8")
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "sample.jpg", "width": 1, "height": 1}],
                    "annotations": [],
                    "categories": [{"id": 1, "name": "object"}],
                }
            ),
            encoding="utf-8",
        )

    report = dataset_report(tmp_path)

    assert report["splits"]["train"]["exists"] is True
    assert report["splits"]["valid"]["ok"] is True
    assert report["splits"]["test"]["images"] == 1
