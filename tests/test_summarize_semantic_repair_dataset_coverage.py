from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_semantic_repair_dataset_coverage import summarize_repair_coverage


def write_split(root: Path, split: str, annotations: list[dict[str, int]]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    data = {
        "images": [{"id": 1}, {"id": 2}, {"id": 3}],
        "categories": [
            {"id": 1, "name": "target"},
            {"id": 2, "name": "wrong"},
            {"id": 3, "name": "unused"},
        ],
        "annotations": annotations,
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")


def test_summarize_repair_coverage_counts_positive_and_negative_support(tmp_path: Path) -> None:
    write_split(
        tmp_path,
        "train",
        [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 2, "category_id": 1},
            {"id": 3, "image_id": 2, "category_id": 1},
            {"id": 4, "image_id": 3, "category_id": 2},
        ],
    )
    write_split(tmp_path, "valid", [{"id": 5, "image_id": 1, "category_id": 2}])
    write_split(tmp_path, "test", [{"id": 6, "image_id": 1, "category_id": 1}])
    config = {
        "targets": [
            {
                "category_id": 1,
                "category_name": "target",
                "total_weight": 2.0,
                "hard_negatives": [
                    {
                        "negative_category_id": 2,
                        "negative_category_name": "wrong",
                        "samples": 4,
                        "weight": 1.5,
                    }
                ],
            }
        ]
    }

    rows = summarize_repair_coverage(dataset_dir=tmp_path, repair_config=config)

    assert rows[0]["role"] == "positive"
    assert rows[0]["category_name"] == "target"
    assert rows[0]["train_images"] == "2"
    assert rows[0]["train_boxes"] == "3"
    assert rows[0]["test_boxes"] == "1"
    assert rows[1]["role"] == "negative"
    assert rows[1]["positive_category_name"] == "target"
    assert rows[1]["category_name"] == "wrong"
    assert rows[1]["configured_samples"] == "4"
    assert rows[1]["configured_weight"] == "1.5"
    assert rows[1]["train_boxes"] == "1"
    assert rows[1]["valid_boxes"] == "1"


def test_summarize_repair_coverage_marks_missing_category_names(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        write_split(tmp_path, split, [])
    config = {
        "targets": [
            {
                "category_id": 99,
                "category_name": "missing",
                "hard_negatives": [],
            }
        ]
    }

    rows = summarize_repair_coverage(dataset_dir=tmp_path, repair_config=config)

    assert rows[0]["present_in_dataset_categories"] == "false"
    assert rows[0]["train_boxes"] == "0"
