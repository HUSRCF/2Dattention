"""Summarize RF-DETR mainline diagnostics from standard CSV artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = (
    "setting",
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
    "candidate_hit_rate",
    "candidate_mean_nearest_iou",
    "candidate_mean_hit_iou",
    "candidate_oracle_ap",
    "candidate_oracle_ap50",
    "candidate_oracle_ap75",
    "top100_loc_recall",
    "top100_class_global_recall",
    "top100_class_percat_recall",
    "score_iou_loc_spearman",
    "score_iou_class_spearman",
    "high_support_min_groups",
    "high_support_categories",
    "high_support_zero_hit",
    "high_support_weighted_hit",
    "high_support_mean_nearest_iou",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help=(
            "Run entry as comma-separated fields: "
            "name,protocol=...,class=...,loc=...,slices=...,candidate=...,"
            "candidate_per_category=...,candidate_oracle=...,coverage=...,"
            "score_loc=...,score_class=...,notes=..."
        ),
    )
    parser.add_argument(
        "--high-support-min-groups",
        type=int,
        default=20,
        help="Minimum grouped boxes for high-support candidate diagnostics.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        summarize_entry(entry, high_support_min_groups=args.high_support_min_groups)
        for entry in args.entry
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_rfdetr_mainline_summary: {args.out}")
    for row in rows:
        print(
            f"{row['setting']}: class_ap50={row['class_ap50']} "
            f"loc_ap50={row['loc_ap50']} cand_hit={row['candidate_hit_rate']}"
        )


def summarize_entry(entry: str, high_support_min_groups: int) -> dict[str, str]:
    name, values = parse_entry(entry)
    out = {field: "" for field in FIELDNAMES}
    out["setting"] = name
    out["protocol"] = values.get("protocol", "")
    out["notes"] = values.get("notes", "")
    out["high_support_min_groups"] = str(high_support_min_groups)

    if "class" in values:
        row = read_single_row(Path(values["class"]))
        out["class_ap"] = row.get("ap", "")
        out["class_ap50"] = row.get("ap50", "")
        out["class_ap75"] = row.get("ap75", "")
    if "loc" in values:
        row = read_single_row(Path(values["loc"]))
        out["loc_ap"] = row.get("ap", "")
        out["loc_ap50"] = row.get("ap50", "")
        out["loc_ap75"] = row.get("ap75", "")
    if "slices" in values:
        slice_rows = read_rows(Path(values["slices"]))
        for slice_name, field in (
            ("offcenter", "offcenter_ap50"),
            ("center", "center_ap50"),
            ("small", "small_ap50"),
            ("medium", "medium_ap50"),
            ("large", "large_ap50"),
        ):
            matched = first_row(slice_rows, "slice", slice_name)
            if matched:
                out[field] = matched.get("ap50", "")
    if "candidate" in values:
        row = read_single_row(Path(values["candidate"]))
        out["candidate_hit_rate"] = row.get("candidate_hit_rate", "")
        out["candidate_mean_nearest_iou"] = row.get("mean_nearest_iou", "")
        out["candidate_mean_hit_iou"] = row.get("mean_hit_iou", "")
    if "candidate_per_category" in values:
        rows = read_rows(Path(values["candidate_per_category"]))
        high_support = [
            row
            for row in rows
            if parse_int(row.get("groups", "0")) >= high_support_min_groups
        ]
        if high_support:
            groups = [parse_int(row["groups"]) for row in high_support]
            hits = [parse_int(row["candidate_hits"]) for row in high_support]
            nearest = [float(row.get("mean_nearest_iou", "0") or 0.0) for row in high_support]
            total_groups = sum(groups)
            out["high_support_categories"] = str(len(high_support))
            out["high_support_zero_hit"] = str(sum(1 for hit in hits if hit == 0))
            out["high_support_weighted_hit"] = format_float(
                sum(hits) / total_groups if total_groups else 0.0
            )
            out["high_support_mean_nearest_iou"] = format_float(
                sum(value * group for value, group in zip(nearest, groups, strict=True)) / total_groups
                if total_groups
                else 0.0
            )
    if "candidate_oracle" in values:
        row = read_single_row(Path(values["candidate_oracle"]))
        out["candidate_oracle_ap"] = row.get("ap", "")
        out["candidate_oracle_ap50"] = row.get("ap50", "")
        out["candidate_oracle_ap75"] = row.get("ap75", "")
    if "coverage" in values:
        rows = read_rows(Path(values["coverage"]))
        matched = [
            row
            for row in rows
            if row.get("slice", "all") == "all"
            and row.get("top_k") == "100"
            and row.get("iou_threshold") == "0.5"
        ]
        if matched:
            row = matched[0]
            out["top100_loc_recall"] = row.get("global_topk_loc_recall", "")
            out["top100_class_global_recall"] = row.get("global_topk_class_recall", "")
            out["top100_class_percat_recall"] = row.get("per_category_topk_class_recall", "")
    if "score_loc" in values:
        row = read_single_row(Path(values["score_loc"]))
        out["score_iou_loc_spearman"] = row.get("spearman_score_iou", "")
    if "score_class" in values:
        row = read_single_row(Path(values["score_class"]))
        out["score_iou_class_spearman"] = row.get("spearman_score_iou", "")
    return out


def parse_entry(entry: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in entry.split(",") if part.strip()]
    if not parts:
        raise ValueError("empty --entry")
    name = parts[0]
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"entry field must use key=value: {part!r}")
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return name, values


def read_single_row(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def format_float(value: float) -> str:
    return f"{value:.12g}"


def parse_int(value: str) -> int:
    return int(float(value or "0"))


if __name__ == "__main__":
    main()
