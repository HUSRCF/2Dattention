"""Rank RF-DETR protocol rows by multiple stage-gate metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument(
        "--metrics",
        default="class_ap50,offcenter_ap50,small_ap50,loc_ap50",
        help="Comma-separated metrics to rank descending.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    rows = rank_rows(summary_paths=args.summary, metrics=metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.out, rows, metrics)
    print(f"saved_rfdetr_stage_gate_rank: {args.out}")
    for row in rows[:10]:
        print(
            f"{row.get('setting', '')}: mean_rank={row['mean_rank']} "
            + " ".join(f"{metric}={row.get(metric, '')}" for metric in metrics)
        )


def rank_rows(*, summary_paths: list[Path], metrics: list[str]) -> list[dict[str, str]]:
    if not metrics:
        raise ValueError("at least one metric is required")
    rows: list[dict[str, str]] = []
    for path in summary_paths:
        for row in read_rows(path):
            enriched = dict(row)
            enriched["source_summary"] = str(path)
            rows.append(enriched)
    if not rows:
        return []
    for metric in metrics:
        if not any(metric in row and row.get(metric, "") != "" for row in rows):
            raise ValueError(f"metric {metric!r} not found in any input row")
        missing_rank = len(rows) + 1
        scored = [
            (index, parse_float(row.get(metric, "")), row.get("setting", ""))
            for index, row in enumerate(rows)
        ]
        scored.sort(key=lambda item: (-item[1], item[2]))
        current_rank = 0
        previous_value: float | None = None
        for index, value, _setting in scored:
            if value == float("-inf"):
                rows[index][f"{metric}_rank"] = str(missing_rank)
                continue
            if previous_value is None or value != previous_value:
                current_rank += 1
                previous_value = value
            rows[index][f"{metric}_rank"] = str(current_rank)
    for row in rows:
        ranks = [parse_int(row.get(f"{metric}_rank", "")) for metric in metrics]
        valid_count = sum(1 for metric in metrics if row.get(metric, "") != "")
        row["mean_rank"] = format_float(sum(ranks) / len(ranks)) if ranks else ""
        row["ranked_metric_count"] = str(valid_count)
    rows.sort(
        key=lambda row: (
            parse_mean_rank(row.get("mean_rank", "")),
            -parse_float(row.get("class_ap50", "")),
            row.get("setting", ""),
        )
    )
    return rows


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], metrics: list[str]) -> None:
    preferred = [
        "source_summary",
        "setting",
        "split_protocol",
        "protocol",
        "mean_rank",
        "ranked_metric_count",
    ]
    for metric in metrics:
        preferred.extend([metric, f"{metric}_rank"])
    preferred.extend(
        [
            "class_ap",
            "class_ap75",
            "loc_ap",
            "loc_ap75",
            "center_ap50",
            "medium_ap50",
            "large_ap50",
            "notes",
        ]
    )
    fieldnames = sorted({key for row in rows for key in row})
    ordered = [field for field in preferred if field in fieldnames]
    ordered.extend(field for field in fieldnames if field not in ordered)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("-inf")
    return float(value)


def parse_mean_rank(value: str | None) -> float:
    if value is None or value == "":
        return float("inf")
    return float(value)


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        raise ValueError("empty integer")
    return int(float(value))


def format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    main()
