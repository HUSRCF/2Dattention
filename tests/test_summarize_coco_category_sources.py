from __future__ import annotations

from pathlib import Path

from scripts.summarize_coco_category_sources import FIELDNAMES
from scripts.summarize_coco_eval_table import parse_entry


def test_category_source_summary_has_stable_core_columns() -> None:
    assert "ap50" in FIELDNAMES
    assert "global_topk_class_recall50" in FIELDNAMES
    assert "global_category_retention50" in FIELDNAMES


def test_category_source_summary_reuses_named_entry_parser() -> None:
    name, path = parse_entry("source=/tmp/predictions.json")

    assert name == "source"
    assert path == Path("/tmp/predictions.json")
