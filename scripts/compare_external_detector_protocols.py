"""Compare one-line external-detector protocol summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DELTA_FIELDS = (
    "coco_ap50",
    "coco_ap75",
    "loc_ap50",
    "loc_ap75",
    "offcenter_ap50",
    "center_ap50",
    "small_ap50",
    "medium_ap50",
    "large_ap50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csvs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=str, default=None, help="Reference row name for delta columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_protocol_summaries(args.summary_csvs, reference=args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_protocol_comparison: {args.out}")


def compare_protocol_summaries(summary_csvs: list[Path], reference: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for path in summary_csvs:
        rows.extend(read_single_row_summary(path))
    if reference is None:
        return rows
    refs = [row for row in rows if row.get("name") == reference]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one reference row named {reference!r}, found {len(refs)}")
    ref = refs[0]
    for row in rows:
        for field in DELTA_FIELDS:
            row[f"delta_{field}"] = parse_float(row.get(field, "")) - parse_float(ref.get(field, ""))
    return rows


def read_single_row_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one summary row in {path}, found {len(rows)}")
    return rows


def parse_float(value: object) -> float:
    if value == "" or value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
