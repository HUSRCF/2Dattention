from __future__ import annotations

import json
from pathlib import Path

from scripts.check_rfdetr_handoff import (
    category_table_sha256,
    coco_split_report,
    dataset_report,
    package_report,
    split_consistency_report,
)


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
    assert report["checked_files"] == 1
    assert report["missing_files_count"] == 0


def test_coco_split_report_checks_all_files_not_only_first_twenty(tmp_path: Path) -> None:
    images = []
    for image_id in range(1, 22):
        file_name = f"sample_{image_id}.jpg"
        images.append({"id": image_id, "file_name": file_name, "width": 10, "height": 10})
        if image_id < 21:
            (tmp_path / file_name).write_text("x", encoding="utf-8")

    report = coco_split_report(
        {
            "images": images,
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}],
        },
        tmp_path,
    )

    assert report["ok"] is False
    assert report["checked_files"] == 21
    assert report["missing_files_count"] == 1
    assert report["missing_files"] == ["sample_21.jpg"]


def test_coco_split_report_flags_orphan_and_invalid_bboxes(tmp_path: Path) -> None:
    (tmp_path / "sample.jpg").write_text("x", encoding="utf-8")
    report = coco_split_report(
        {
            "images": [{"id": 1, "file_name": "sample.jpg", "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1]},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 1]},
                {"id": 3, "image_id": 1, "category_id": 1, "bbox": [9, 9, 2, 2]},
            ],
            "categories": [{"id": 1, "name": "object"}],
        },
        tmp_path,
    )

    assert report["ok"] is False
    assert report["orphan_annotation_ids"] == [1]
    assert report["invalid_bbox_ids"] == [2]
    assert report["out_of_bounds_bbox_ids"] == [3]


def test_coco_split_report_tolerates_tiny_float_bbox_overflow(tmp_path: Path) -> None:
    (tmp_path / "sample.jpg").write_text("x", encoding="utf-8")
    report = coco_split_report(
        {
            "images": [{"id": 1, "file_name": "sample.jpg", "width": 10, "height": 10}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10.0005, 10]}],
            "categories": [{"id": 1, "name": "object"}],
        },
        tmp_path,
    )

    assert report["ok"] is True
    assert report["out_of_bounds_bbox_ids"] == []


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


def test_split_consistency_checks_full_category_ids_and_table() -> None:
    train_categories = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    reordered_categories = [{"id": 2, "name": "b"}, {"id": 1, "name": "a"}]
    renamed_categories = [{"id": 1, "name": "a"}, {"id": 2, "name": "renamed"}]
    train_sha = category_table_sha256(train_categories)

    report = split_consistency_report(
        {
            "train": {
                "exists": True,
                "min_category_id": 1,
                "max_category_id": 2,
                "categories": 2,
                "category_ids": [1, 2],
                "category_table_sha256": train_sha,
                "annotation_sha256": "train",
            },
            "valid": {
                "exists": True,
                "min_category_id": 1,
                "max_category_id": 3,
                "categories": 2,
                "category_ids": [1, 3],
                "category_table_sha256": category_table_sha256(reordered_categories),
                "annotation_sha256": "valid",
            },
            "test": {
                "exists": True,
                "min_category_id": 1,
                "max_category_id": 2,
                "categories": 2,
                "category_ids": [1, 2],
                "category_table_sha256": category_table_sha256(renamed_categories),
                "annotation_sha256": "valid",
            },
        },
        {
            "train": {1, 2},
            "valid": {3},
            "test": {3, 4},
        },
    )

    assert report["category_range_matches_train"]["valid"] is False
    assert report["category_ids_match_train"]["valid"] is False
    assert report["category_ids_match_train"]["test"] is True
    assert report["category_table_matches_train"]["test"] is False
    assert report["valid_test_annotations_identical"] is True
    assert report["image_id_overlap_counts"] == {
        "train_valid": 0,
        "train_test": 0,
        "valid_test": 1,
    }
    assert report["image_ids_disjoint"] is False


def test_split_consistency_reports_file_name_overlap_separately_from_numeric_ids() -> None:
    report = split_consistency_report(
        {
            "train": {
                "exists": True,
                "min_category_id": 1,
                "max_category_id": 1,
                "categories": 1,
                "category_ids": [1],
                "category_table_sha256": "sha",
                "annotation_sha256": "train",
            },
            "valid": {
                "exists": True,
                "min_category_id": 1,
                "max_category_id": 1,
                "categories": 1,
                "category_ids": [1],
                "category_table_sha256": "sha",
                "annotation_sha256": "valid",
            },
            "test": {
                "exists": True,
                "min_category_id": 1,
                "max_category_id": 1,
                "categories": 1,
                "category_ids": [1],
                "category_table_sha256": "sha",
                "annotation_sha256": "test",
            },
        },
        {"train": {1}, "valid": {1}, "test": {3}},
        {"train": {"a.jpg"}, "valid": {"b.jpg"}, "test": {"a.jpg"}},
    )

    assert report["image_id_overlap_counts"]["train_valid"] == 1
    assert report["file_name_overlap_counts"]["train_valid"] == 0
    assert report["file_name_overlap_counts"]["train_test"] == 1
    assert report["image_ids_disjoint"] is False
    assert report["file_names_disjoint"] is False


def test_package_report_exposes_import_failure(monkeypatch) -> None:
    import importlib.machinery

    monkeypatch.setattr(
        "scripts.check_rfdetr_handoff.importlib.util.find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None),
    )

    def fake_import_module(name: str):
        if name == "rfdetr":
            raise ImportError("broken transitive dependency")
        return object()

    monkeypatch.setattr("scripts.check_rfdetr_handoff.importlib.import_module", fake_import_module)

    report = package_report()

    assert report["rfdetr"]["installed"] is True
    assert report["rfdetr"]["import_ok"] is False
    assert "broken transitive dependency" in report["rfdetr"]["import_error"]
