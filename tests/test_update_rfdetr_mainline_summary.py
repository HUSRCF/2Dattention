from __future__ import annotations

import csv
from pathlib import Path

from scripts.rank_rfdetr_stage_gate_rows import read_rows
from scripts.update_rfdetr_mainline_summary import build_standard_entry
from scripts.update_rfdetr_mainline_summary import summarize_standard_artifacts
from scripts.update_rfdetr_mainline_summary import update_summary_rows
from scripts.update_rfdetr_mainline_summary import write_summary


def write_one(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_standard_artifacts(prefix: Path) -> None:
    coco_fields = ["ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar1", "ar10", "ar100"]
    write_one(prefix.with_name(prefix.name + "_test_cocoeval.csv"), coco_fields, {"ap": 0.1, "ap50": 0.2, "ap75": 0.3})
    write_one(prefix.with_name(prefix.name + "_test_loc_cocoeval.csv"), coco_fields, {"ap": 0.4, "ap50": 0.5, "ap75": 0.6})
    write_rows(
        prefix.with_name(prefix.name + "_test_slices.csv"),
        ["slice", "images", "annotations", "ap", "ap50", "ap75"],
        [
            {"slice": "offcenter", "images": 1, "annotations": 1, "ap50": 0.21},
            {"slice": "center", "images": 1, "annotations": 1, "ap50": 0.22},
            {"slice": "small", "images": 1, "annotations": 1, "ap50": 0.23},
            {"slice": "medium", "images": 1, "annotations": 1, "ap50": 0.24},
            {"slice": "large", "images": 1, "annotations": 1, "ap50": 0.25},
        ],
    )
    write_one(
        prefix.with_name(prefix.name + "_candidate_oracle_summary.csv"),
        ["groups", "emitted_predictions", "candidate_hits", "candidate_hit_rate", "mean_nearest_iou", "mean_hit_iou", "score_mode"],
        {
            "groups": 10,
            "emitted_predictions": 5,
            "candidate_hits": 5,
            "candidate_hit_rate": 0.5,
            "mean_nearest_iou": 0.7,
            "mean_hit_iou": 0.8,
            "score_mode": "group_max",
        },
    )
    write_rows(
        prefix.with_name(prefix.name + "_candidate_oracle_per_category.csv"),
        ["category_id", "category_name", "groups", "candidate_hits", "candidate_hit_rate", "mean_nearest_iou", "mean_hit_iou"],
        [
            {
                "category_id": 1,
                "category_name": "n00000001",
                "groups": 20,
                "candidate_hits": 10,
                "candidate_hit_rate": 0.5,
                "mean_nearest_iou": 0.7,
                "mean_hit_iou": 0.8,
            }
        ],
    )
    write_one(prefix.with_name(prefix.name + "_candidate_oracle_groupmax_cocoeval.csv"), coco_fields, {"ap": 0.11, "ap50": 0.12, "ap75": 0.13})
    write_rows(
        prefix.with_name(prefix.name + "_category_coverage_gap.csv"),
        [
            "slice",
            "top_k",
            "iou_threshold",
            "gt_count",
            "global_topk_loc_recall",
            "global_topk_class_recall",
            "per_category_topk_class_recall",
            "global_category_retention",
            "ranking_gap",
            "mean_global_loc_best_iou",
            "mean_global_class_best_iou",
            "mean_per_category_class_best_iou",
        ],
        [
            {
                "slice": "all",
                "top_k": 100,
                "iou_threshold": 0.5,
                "gt_count": 1,
                "global_topk_loc_recall": 0.9,
                "global_topk_class_recall": 0.8,
                "per_category_topk_class_recall": 0.85,
            }
        ],
    )
    score_fields = [
        "predictions",
        "images",
        "mean_score",
        "mean_nearest_iou",
        "pearson_score_iou",
        "spearman_score_iou",
        "top10_mean_iou",
        "top50_mean_iou",
        "top100_mean_iou",
    ]
    write_one(prefix.with_name(prefix.name + "_score_iou_loc.csv"), score_fields, {"spearman_score_iou": 0.31})
    write_one(prefix.with_name(prefix.name + "_score_iou_classaware.csv"), score_fields, {"spearman_score_iou": 0.41})


def test_build_standard_entry_uses_artifact_prefix() -> None:
    entry = build_standard_entry(
        setting="run",
        protocol="proto",
        artifact_prefix=Path("results/run"),
        notes="note",
    )

    assert entry.startswith("run,protocol=proto,class=results/run_test_cocoeval.csv")
    assert "candidate_oracle=results/run_candidate_oracle_groupmax_cocoeval.csv" in entry
    assert entry.endswith(",notes=note")


def test_build_standard_entry_uses_candidate_score_mode_suffix() -> None:
    entry = build_standard_entry(
        setting="run",
        protocol="proto",
        artifact_prefix=Path("results/run"),
        candidate_score_mode="oracle_iou",
    )

    assert "candidate_oracle=results/run_candidate_oracle_oracleiou_cocoeval.csv" in entry


def test_summarize_standard_artifacts_reads_bundle(tmp_path: Path) -> None:
    prefix = tmp_path / "run"
    write_standard_artifacts(prefix)

    row = summarize_standard_artifacts(
        setting="run",
        protocol="proto",
        artifact_prefix=prefix,
        notes="note",
    )

    assert row["setting"] == "run"
    assert row["protocol"] == "proto"
    assert row["class_ap50"] == "0.2"
    assert row["loc_ap50"] == "0.5"
    assert row["offcenter_ap50"] == "0.21"
    assert row["candidate_hit_rate"] == "0.5"
    assert row["candidate_oracle_ap50"] == "0.12"
    assert row["top100_class_percat_recall"] == "0.85"
    assert row["score_iou_class_spearman"] == "0.41"
    assert row["high_support_weighted_hit"] == "0.5"


def test_update_summary_rows_replaces_by_setting() -> None:
    existing = [{"setting": "old", "class_ap50": "0.1"}, {"setting": "keep", "class_ap50": "0.2"}]
    new = {"setting": "old", "class_ap50": "0.3"}

    rows = update_summary_rows(existing, new, replace=True)

    assert rows == [{"setting": "old", "class_ap50": "0.3"}, {"setting": "keep", "class_ap50": "0.2"}]


def test_write_summary_uses_mainline_field_order(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"

    write_summary(path, [{"setting": "run", "class_ap50": "0.2", "unknown": "ignored"}])
    rows = read_rows(path)

    assert rows[0]["setting"] == "run"
    assert rows[0]["class_ap50"] == "0.2"
    assert "unknown" not in rows[0]
