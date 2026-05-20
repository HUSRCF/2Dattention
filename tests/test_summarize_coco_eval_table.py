from __future__ import annotations

import pytest

from pathlib import Path

from scripts.summarize_coco_eval_table import parse_entry, read_cocoeval_csv, summarize_entry


def write_cocoeval(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ap,ap50,ap75,ap_small,ap_medium,ap_large,ar1,ar10,ar100",
                "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_entry_requires_named_path() -> None:
    name, path = parse_entry("baseline=/tmp/baseline.csv")

    assert name == "baseline"
    assert str(path) == "/tmp/baseline.csv"


def test_parse_entry_rejects_unnamed_path() -> None:
    with pytest.raises(ValueError):
        parse_entry("/tmp/baseline.csv")


def test_read_cocoeval_csv(tmp_path: Path) -> None:
    path = tmp_path / "eval.csv"
    write_cocoeval(path)

    metrics = read_cocoeval_csv(path)

    assert metrics["ap"] == 0.1
    assert metrics["ap50"] == 0.2
    assert metrics["ar100"] == 0.9


def test_summarize_entry(tmp_path: Path) -> None:
    path = tmp_path / "eval.csv"
    write_cocoeval(path)

    row = summarize_entry(f"model={path}")

    assert row["name"] == "model"
    assert row["source"] == str(path)
    assert row["ap75"] == 0.3
