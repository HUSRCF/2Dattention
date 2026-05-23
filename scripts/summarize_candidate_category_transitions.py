"""Compare per-category candidate hit transitions across detector runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_COLUMNS = (
    "category_id",
    "category_name",
    "max_groups",
    "high_support_runs",
    "zero_hit_runs",
    "transition",
    "hit_rate_delta_latest_vs_base",
    "nearest_iou_delta_latest_vs_base",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Named per-category candidate CSV as name=/path/to/file.csv.",
    )
    parser.add_argument(
        "--min-groups",
        type=int,
        default=20,
        help="Minimum grouped boxes for high-support transition labels.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = [parse_entry(entry) for entry in args.entry]
    names = [name for name, _ in entries]
    per_run = [(name, read_per_category(path)) for name, path in entries]
    rows = build_rows(per_run, min_groups=args.min_groups)
    fieldnames = list(BASE_COLUMNS)
    for name in names:
        fieldnames.extend(
            [
                f"{name}_groups",
                f"{name}_hits",
                f"{name}_hit_rate",
                f"{name}_mean_nearest_iou",
                f"{name}_zero_hit",
            ]
        )
    rows.sort(
        key=lambda row: (
            transition_rank(row["transition"]),
            -parse_float(row["max_groups"]),
            row["category_id"],
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_candidate_category_transitions: {args.out}")
    print_summary(rows)


def build_rows(
    per_run: list[tuple[str, dict[int, dict[str, str]]]],
    min_groups: int,
) -> list[dict[str, str]]:
    all_category_ids = sorted({category_id for _, rows in per_run for category_id in rows})
    base_name, base_rows = per_run[0]
    latest_name, latest_rows = per_run[-1]
    out = []
    for category_id in all_category_ids:
        names = [
            rows[category_id].get("category_name", "")
            for _, rows in per_run
            if category_id in rows
        ]
        category_name = next((name for name in names if name), str(category_id))
        run_stats = []
        row = {
            "category_id": str(category_id),
            "category_name": category_name,
        }
        for name, rows in per_run:
            stats = stats_for(rows.get(category_id, {}), min_groups=min_groups)
            run_stats.append(stats)
            row[f"{name}_groups"] = str(stats.groups)
            row[f"{name}_hits"] = str(stats.hits)
            row[f"{name}_hit_rate"] = format_float(stats.hit_rate)
            row[f"{name}_mean_nearest_iou"] = format_float(stats.mean_nearest_iou)
            row[f"{name}_zero_hit"] = str(int(stats.high_support_zero_hit))
        high_support = [stats for stats in run_stats if stats.high_support]
        row["max_groups"] = str(max((stats.groups for stats in run_stats), default=0))
        row["high_support_runs"] = str(len(high_support))
        row["zero_hit_runs"] = str(sum(1 for stats in high_support if stats.high_support_zero_hit))
        row["transition"] = transition_label(
            base_rows.get(category_id, {}),
            latest_rows.get(category_id, {}),
            min_groups=min_groups,
            base_name=base_name,
            latest_name=latest_name,
        )
        row["hit_rate_delta_latest_vs_base"] = format_float(
            stats_for(latest_rows.get(category_id, {}), min_groups=min_groups).hit_rate
            - stats_for(base_rows.get(category_id, {}), min_groups=min_groups).hit_rate
        )
        row["nearest_iou_delta_latest_vs_base"] = format_float(
            stats_for(latest_rows.get(category_id, {}), min_groups=min_groups).mean_nearest_iou
            - stats_for(base_rows.get(category_id, {}), min_groups=min_groups).mean_nearest_iou
        )
        out.append(row)
    return out


def transition_label(
    base_row: dict[str, str],
    latest_row: dict[str, str],
    min_groups: int,
    base_name: str,
    latest_name: str,
) -> str:
    base = stats_for(base_row, min_groups=min_groups)
    latest = stats_for(latest_row, min_groups=min_groups)
    if not base.high_support and not latest.high_support:
        return "no_high_support"
    if not base.high_support and latest.high_support:
        return f"new_high_support_in_{latest_name}"
    if base.high_support and not latest.high_support:
        return f"lost_high_support_after_{base_name}"
    if base.high_support_zero_hit and latest.high_support_zero_hit:
        return "persistent_zero_hit"
    if base.high_support_zero_hit and not latest.high_support_zero_hit:
        return "rescued_from_zero_hit"
    if not base.high_support_zero_hit and latest.high_support_zero_hit:
        return "regressed_to_zero_hit"
    return "persistent_candidate_hit"


def parse_entry(entry: str) -> tuple[str, Path]:
    if "=" not in entry:
        raise ValueError(f"expected name=path entry, got {entry!r}")
    name, path_text = entry.split("=", 1)
    if not name:
        raise ValueError(f"empty entry name in {entry!r}")
    return name, Path(path_text)


def read_per_category(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {parse_int(row["category_id"]): row for row in rows}


class CategoryStats:
    def __init__(self, row: dict[str, str], min_groups: int) -> None:
        self.groups = parse_int(row.get("groups", "0"))
        self.hits = parse_int(row.get("candidate_hits", "0"))
        self.hit_rate = parse_float(row.get("candidate_hit_rate", "0"))
        self.mean_nearest_iou = parse_float(row.get("mean_nearest_iou", "0"))
        self.high_support = self.groups >= min_groups
        self.high_support_zero_hit = self.high_support and self.hits == 0


def stats_for(row: dict[str, str], min_groups: int) -> CategoryStats:
    return CategoryStats(row, min_groups=min_groups)


def transition_rank(label: str) -> int:
    order = {
        "persistent_zero_hit": 0,
        "regressed_to_zero_hit": 1,
        "rescued_from_zero_hit": 2,
        "persistent_candidate_hit": 3,
        "new_high_support": 4,
        "lost_high_support": 5,
        "no_high_support": 6,
    }
    for prefix, rank in order.items():
        if label.startswith(prefix):
            return rank
    return 99


def print_summary(rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["transition"]] = counts.get(row["transition"], 0) + 1
    for label, count in sorted(counts.items(), key=lambda item: (transition_rank(item[0]), item[0])):
        print(f"{label}: {count}")


def parse_int(value: str) -> int:
    return int(float(value or "0"))


def parse_float(value: str) -> float:
    return float(value or "0")


def format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    main()
