"""Build a compact report for high-support candidate failure categories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = (
    "category_id",
    "category_name",
    "transition",
    "max_groups",
    "train_gt_count",
    "valid_gt_count",
    "test_gt_count",
    "base_groups",
    "base_hits",
    "base_hit_rate",
    "base_mean_nearest_iou",
    "latest_groups",
    "latest_hits",
    "latest_hit_rate",
    "latest_mean_nearest_iou",
    "hit_rate_delta_latest_vs_base",
    "nearest_iou_delta_latest_vs_base",
    "top_wrong_pred_category_id",
    "top_wrong_pred_category_name",
    "top_wrong_pair_count",
    "top_wrong_loc_matched_count",
    "top_wrong_pair_fraction_of_loc_matched",
    "top_wrong_mean_iou",
    "top_wrong_mean_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument(
        "--split-counts",
        type=Path,
        required=True,
        help="Per-category candidate CSV with train/valid/test count columns.",
    )
    parser.add_argument(
        "--confusions",
        type=Path,
        required=True,
        help="COCO category confusion CSV from analyze_coco_category_confusions.py.",
    )
    parser.add_argument(
        "--transition",
        action="append",
        default=["persistent_zero_hit", "regressed_to_zero_hit"],
        help="Transition label to keep. Can be repeated.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transitions = read_rows(args.transitions)
    split_counts = {row["category_id"]: row for row in read_rows(args.split_counts)}
    confusions = top_confusions_by_gt(read_rows(args.confusions))
    keep = set(args.transition)
    rows = []
    for row in transitions:
        if row.get("transition") not in keep:
            continue
        category_id = row["category_id"]
        counts = split_counts.get(category_id, {})
        confusion = confusions.get(category_id, {})
        latest_prefix = latest_run_prefix(row)
        rows.append(
            {
                "category_id": category_id,
                "category_name": row.get("category_name", ""),
                "transition": row.get("transition", ""),
                "max_groups": row.get("max_groups", ""),
                "train_gt_count": counts.get("train_gt_count", ""),
                "valid_gt_count": counts.get("valid_gt_count", ""),
                "test_gt_count": counts.get("test_gt_count", ""),
                "base_groups": row.get("base_groups", ""),
                "base_hits": row.get("base_hits", ""),
                "base_hit_rate": row.get("base_hit_rate", ""),
                "base_mean_nearest_iou": row.get("base_mean_nearest_iou", ""),
                "latest_groups": row.get(f"{latest_prefix}_groups", ""),
                "latest_hits": row.get(f"{latest_prefix}_hits", ""),
                "latest_hit_rate": row.get(f"{latest_prefix}_hit_rate", ""),
                "latest_mean_nearest_iou": row.get(f"{latest_prefix}_mean_nearest_iou", ""),
                "hit_rate_delta_latest_vs_base": row.get("hit_rate_delta_latest_vs_base", ""),
                "nearest_iou_delta_latest_vs_base": row.get("nearest_iou_delta_latest_vs_base", ""),
                "top_wrong_pred_category_id": confusion.get("pred_category_id", ""),
                "top_wrong_pred_category_name": confusion.get("pred_category_name", ""),
                "top_wrong_pair_count": confusion.get("pair_count", ""),
                "top_wrong_loc_matched_count": confusion.get("loc_matched_count", ""),
                "top_wrong_pair_fraction_of_loc_matched": confusion.get("pair_fraction_of_loc_matched", ""),
                "top_wrong_mean_iou": confusion.get("mean_iou", ""),
                "top_wrong_mean_score": confusion.get("mean_score", ""),
            }
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
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_candidate_failure_categories: {args.out}")
    print(f"rows={len(rows)} transitions={','.join(sorted(keep))}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def top_confusions_by_gt(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_gt: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("is_correct_category", "False") == "True":
            continue
        category_id = row["gt_category_id"]
        current = by_gt.get(category_id)
        if current is None or confusion_rank(row) < confusion_rank(current):
            by_gt[category_id] = row
    return by_gt


def confusion_rank(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        -parse_float(row.get("pair_count", "0")),
        -parse_float(row.get("mean_iou", "0")),
        -parse_float(row.get("mean_score", "0")),
    )


def latest_run_prefix(row: dict[str, str]) -> str:
    prefixes = []
    for key in row:
        if key.endswith("_groups") and key not in {"base_groups", "max_groups"}:
            prefixes.append(key.removesuffix("_groups"))
    if not prefixes:
        return "base"
    return prefixes[-1]


def transition_rank(label: str) -> int:
    return {
        "persistent_zero_hit": 0,
        "regressed_to_zero_hit": 1,
        "rescued_from_zero_hit": 2,
    }.get(label, 99)


def parse_float(value: str) -> float:
    return float(value or "0")


if __name__ == "__main__":
    main()
