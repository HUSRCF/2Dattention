"""Select top RF-DETR protocol rows from one or more summary CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--metric", default="class_ap50")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = select_rows(summary_paths=args.summary, metric=args.metric, top_k=args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.out, rows)
    print(f"saved_rfdetr_stage_gate_rows: {args.out}")
    for row in rows:
        print(f"{row.get('setting', '')}: {args.metric}={row.get(args.metric, '')}")


def select_rows(*, summary_paths: list[Path], metric: str, top_k: int) -> list[dict[str, str]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    rows: list[dict[str, str]] = []
    for path in summary_paths:
        for row in read_rows(path):
            if metric not in row:
                raise ValueError(f"metric {metric!r} not found in {path}")
            out = dict(row)
            out["source_summary"] = str(path)
            rows.append(out)
    rows.sort(key=lambda row: (-parse_float(row.get(metric, "")), row.get("setting", "")))
    return rows[:top_k]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "source_summary",
        "setting",
        "split_protocol",
        "protocol",
        "class_ap",
        "class_ap50",
        "class_ap75",
        "loc_ap",
        "loc_ap50",
        "loc_ap75",
        "offcenter_ap50",
        "center_ap50",
        "small_ap50",
        "medium_ap50",
        "large_ap50",
        "notes",
    ]
    ordered = [field for field in preferred if field in fieldnames]
    ordered.extend(field for field in fieldnames if field not in ordered)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str) -> float:
    if value is None or value == "":
        return float("-inf")
    return float(value)


if __name__ == "__main__":
    main()
