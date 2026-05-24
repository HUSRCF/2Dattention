from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.select_rfdetr_stage_gate_rows import select_rows


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["setting", "class_ap50", "loc_ap50"])
        writer.writeheader()
        writer.writerows(rows)


def test_select_rows_ranks_across_multiple_summaries(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_summary(
        first,
        [
            {"setting": "a", "class_ap50": "0.10", "loc_ap50": "0.50"},
            {"setting": "b", "class_ap50": "0.30", "loc_ap50": "0.40"},
        ],
    )
    write_summary(second, [{"setting": "c", "class_ap50": "0.20", "loc_ap50": "0.60"}])

    rows = select_rows(summary_paths=[first, second], metric="class_ap50", top_k=2)

    assert [row["setting"] for row in rows] == ["b", "c"]
    assert rows[0]["source_summary"] == str(first)
    assert rows[1]["source_summary"] == str(second)


def test_select_rows_rejects_missing_metric(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    write_summary(summary, [{"setting": "a", "class_ap50": "0.1", "loc_ap50": "0.2"}])

    with pytest.raises(ValueError, match="not found"):
        select_rows(summary_paths=[summary], metric="missing", top_k=1)
