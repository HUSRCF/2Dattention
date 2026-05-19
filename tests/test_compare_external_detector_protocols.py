from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.compare_external_detector_protocols import compare_protocol_summaries


def write_summary(path: Path, name: str, ap50: float, offcenter_ap50: float) -> None:
    fieldnames = [
        "name",
        "coco_ap50",
        "coco_ap75",
        "loc_ap50",
        "loc_ap75",
        "offcenter_ap50",
        "center_ap50",
        "small_ap50",
        "medium_ap50",
        "large_ap50",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "name": name,
                "coco_ap50": ap50,
                "coco_ap75": ap50 / 2.0,
                "loc_ap50": ap50 + 0.10,
                "loc_ap75": ap50 / 2.0 + 0.05,
                "offcenter_ap50": offcenter_ap50,
                "center_ap50": 0.0,
                "small_ap50": 0.0,
                "medium_ap50": 0.0,
                "large_ap50": 0.0,
            }
        )


def test_compare_protocol_summaries_adds_reference_deltas(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    model = tmp_path / "model.csv"
    write_summary(baseline, "baseline", ap50=0.10, offcenter_ap50=0.05)
    write_summary(model, "model", ap50=0.13, offcenter_ap50=0.08)

    rows = compare_protocol_summaries([baseline, model], reference="baseline")
    by_name = {row["name"]: row for row in rows}

    assert by_name["baseline"]["delta_coco_ap50"] == 0.0
    assert round(by_name["model"]["delta_coco_ap50"], 3) == 0.03
    assert round(by_name["model"]["delta_loc_ap50"], 3) == 0.03
    assert round(by_name["model"]["delta_offcenter_ap50"], 3) == 0.03


def test_compare_protocol_summaries_rejects_missing_reference(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    write_summary(baseline, "baseline", ap50=0.10, offcenter_ap50=0.05)

    with pytest.raises(ValueError, match="expected exactly one reference"):
        compare_protocol_summaries([baseline], reference="missing")
