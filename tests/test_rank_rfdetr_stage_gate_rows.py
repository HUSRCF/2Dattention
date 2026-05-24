from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.rank_rfdetr_stage_gate_rows import rank_rows


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_rank_rows_computes_mean_rank_across_metrics(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_summary(
        first,
        [
            {"setting": "class_best", "class_ap50": "0.40", "offcenter_ap50": "0.10"},
            {"setting": "balanced", "class_ap50": "0.30", "offcenter_ap50": "0.30"},
        ],
    )
    write_summary(
        second,
        [
            {"setting": "offcenter_best", "class_ap50": "0.10", "offcenter_ap50": "0.40"},
            {"setting": "low", "class_ap50": "0.20", "offcenter_ap50": "0.20"},
        ],
    )

    rows = rank_rows(summary_paths=[first, second], metrics=["class_ap50", "offcenter_ap50"])

    assert [row["setting"] for row in rows] == ["balanced", "class_best", "offcenter_best", "low"]
    assert rows[0]["class_ap50_rank"] == "2"
    assert rows[0]["offcenter_ap50_rank"] == "2"
    assert rows[0]["mean_rank"] == "2"
    assert rows[0]["ranked_metric_count"] == "2"
    assert rows[0]["source_summary"] == str(first)


def test_rank_rows_skips_blank_metric_values(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    write_summary(
        summary,
        [
            {"setting": "complete", "class_ap50": "0.3", "small_ap50": "0.2"},
            {"setting": "missing_small", "class_ap50": "0.4", "small_ap50": ""},
        ],
    )

    rows = rank_rows(summary_paths=[summary], metrics=["class_ap50", "small_ap50"])
    missing_small = next(row for row in rows if row["setting"] == "missing_small")

    assert [row["setting"] for row in rows] == ["complete", "missing_small"]
    assert missing_small["class_ap50_rank"] == "1"
    assert missing_small["small_ap50_rank"] == "3"
    assert missing_small["mean_rank"] == "2"
    assert missing_small["ranked_metric_count"] == "1"


def test_rank_rows_sorts_rows_without_ranked_metrics_last(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    write_summary(
        summary,
        [
            {"setting": "ranked", "class_ap50": "0.3", "small_ap50": "0.2"},
            {"setting": "unranked", "class_ap50": "", "small_ap50": ""},
        ],
    )

    rows = rank_rows(summary_paths=[summary], metrics=["class_ap50", "small_ap50"])

    assert [row["setting"] for row in rows] == ["ranked", "unranked"]
    assert rows[1]["class_ap50_rank"] == "3"
    assert rows[1]["small_ap50_rank"] == "3"
    assert rows[1]["mean_rank"] == "3"
    assert rows[1]["ranked_metric_count"] == "0"


def test_rank_rows_uses_dense_tie_ranks(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    write_summary(
        summary,
        [
            {"setting": "a", "class_ap50": "0.3", "small_ap50": "0.2"},
            {"setting": "b", "class_ap50": "0.3", "small_ap50": "0.1"},
            {"setting": "c", "class_ap50": "0.2", "small_ap50": "0.3"},
        ],
    )

    rows = rank_rows(summary_paths=[summary], metrics=["class_ap50", "small_ap50"])
    by_setting = {row["setting"]: row for row in rows}

    assert by_setting["a"]["class_ap50_rank"] == "1"
    assert by_setting["b"]["class_ap50_rank"] == "1"
    assert by_setting["c"]["class_ap50_rank"] == "2"


def test_rank_rows_rejects_metric_absent_from_all_inputs(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    write_summary(summary, [{"setting": "a", "class_ap50": "0.1"}])

    with pytest.raises(ValueError, match="not found"):
        rank_rows(summary_paths=[summary], metrics=["missing_metric"])


def test_rank_rows_rejects_empty_metric_list(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    write_summary(summary, [{"setting": "a", "class_ap50": "0.1"}])

    with pytest.raises(ValueError, match="at least one metric"):
        rank_rows(summary_paths=[summary], metrics=[])
