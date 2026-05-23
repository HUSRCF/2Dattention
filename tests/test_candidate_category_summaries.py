from __future__ import annotations

import csv
import sys
from pathlib import Path

from scripts import summarize_candidate_category_transitions
from scripts import summarize_candidate_failure_categories
from scripts.select_coco_category_samples import select_category_samples


def test_candidate_category_transitions_label_zero_hit_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base = tmp_path / "base.csv"
    latest = tmp_path / "latest.csv"
    base.write_text(
        "\n".join(
            [
                "category_id,category_name,groups,candidate_hits,candidate_hit_rate,mean_nearest_iou,mean_hit_iou",
                "1,cat_one,20,0,0.0,0.4,0.0",
                "2,cat_two,20,2,0.1,0.5,0.5",
                "3,cat_three,20,0,0.0,0.6,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    latest.write_text(
        "\n".join(
            [
                "category_id,category_name,groups,candidate_hits,candidate_hit_rate,mean_nearest_iou,mean_hit_iou",
                "1,cat_one,20,3,0.15,0.45,0.45",
                "2,cat_two,20,0,0.0,0.55,0.0",
                "3,cat_three,20,0,0.0,0.62,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "transitions.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_candidate_category_transitions.py",
            "--entry",
            f"base={base}",
            "--entry",
            f"latest={latest}",
            "--out",
            str(out),
        ],
    )

    summarize_candidate_category_transitions.main()
    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    by_id = {row["category_id"]: row for row in rows}

    assert by_id["1"]["transition"] == "rescued_from_zero_hit"
    assert by_id["2"]["transition"] == "regressed_to_zero_hit"
    assert by_id["3"]["transition"] == "persistent_zero_hit"
    assert by_id["1"]["hit_rate_delta_latest_vs_base"] == "0.15"


def test_candidate_failure_categories_join_counts_and_top_confusion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    transitions = tmp_path / "transitions.csv"
    transitions.write_text(
        "\n".join(
            [
                "category_id,category_name,max_groups,transition,hit_rate_delta_latest_vs_base,nearest_iou_delta_latest_vs_base,base_groups,base_hits,base_hit_rate,base_mean_nearest_iou,latest_groups,latest_hits,latest_hit_rate,latest_mean_nearest_iou",
                "1,cat_one,25,persistent_zero_hit,0,0.1,20,0,0,0.3,25,0,0,0.4",
                "2,cat_two,22,rescued_from_zero_hit,0.2,0.0,20,0,0,0.4,22,4,0.2,0.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split_counts = tmp_path / "split_counts.csv"
    split_counts.write_text(
        "\n".join(
            [
                "category_id,category_name,groups,candidate_hits,candidate_hit_rate,mean_nearest_iou,mean_hit_iou,train_gt_count,valid_gt_count,test_gt_count",
                "1,cat_one,20,0,0,0.3,0,7,1,3",
                "2,cat_two,20,0,0,0.4,0,8,2,4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    confusions = tmp_path / "confusions.csv"
    confusions.write_text(
        "\n".join(
            [
                "gt_category_id,gt_category_name,pred_category_id,pred_category_name,top_k,iou_threshold,gt_count,loc_matched_count,pair_count,pair_fraction_of_gt,pair_fraction_of_loc_matched,mean_iou,mean_score,is_correct_category",
                "1,cat_one,9,wrong_low,100,0.5,5,5,1,0.2,0.2,0.95,0.9,False",
                "1,cat_one,8,wrong_high,100,0.5,5,5,3,0.6,0.6,0.80,0.2,False",
                "2,cat_two,7,wrong_two,100,0.5,5,5,4,0.8,0.8,0.90,0.3,False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "failure.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_candidate_failure_categories.py",
            "--transitions",
            str(transitions),
            "--split-counts",
            str(split_counts),
            "--confusions",
            str(confusions),
            "--out",
            str(out),
        ],
    )

    summarize_candidate_failure_categories.main()
    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))

    assert len(rows) == 1
    assert rows[0]["category_id"] == "1"
    assert rows[0]["train_gt_count"] == "7"
    assert rows[0]["top_wrong_pred_category_id"] == "8"
    assert rows[0]["top_wrong_pair_count"] == "3"


def test_select_coco_category_samples_filters_transition_and_limits() -> None:
    annotations = {
        "images": [
            {"id": 1, "file_name": "one.jpg", "width": 100, "height": 50},
            {"id": 2, "file_name": "two.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]},
            {"id": 11, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 12, "image_id": 2, "category_id": 2, "bbox": [5, 5, 20, 20]},
        ],
    }
    category_report = [
        {"category_id": "1", "category_name": "cat_one", "transition": "persistent_zero_hit"},
        {"category_id": "2", "category_name": "cat_two", "transition": "rescued_from_zero_hit"},
    ]

    rows = select_category_samples(
        annotations=annotations,
        category_report=category_report,
        transitions={"persistent_zero_hit"},
        max_per_category=1,
    )

    assert len(rows) == 1
    assert rows[0]["image_id"] == "1"
    assert rows[0]["category_name"] == "cat_one"
    assert rows[0]["area"] == "200"
    assert rows[0]["area_ratio"] == "0.04"
