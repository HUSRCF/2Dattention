"""Build a compact RF-DETR category-candidate repair target table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-categories", type=Path, required=True)
    parser.add_argument("--flow-by-category", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_targets(
        failure_rows=read_rows(args.failure_categories),
        flow_rows=read_rows(args.flow_by_category),
    )
    write_rows(rows, args.out)
    print(f"saved_repair_targets: {args.out}")


def build_targets(
    *,
    failure_rows: list[dict[str, str]],
    flow_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    flows = {
        (row["gt_category_id"], row["label"]): row
        for row in flow_rows
    }
    labels = sorted({row["label"] for row in flow_rows})
    out = []
    for row in failure_rows:
        category_id = row["category_id"]
        merged: dict[str, object] = {
            "category_id": category_id,
            "category_name": row["category_name"],
            "transition": row["transition"],
            "train_gt_count": row.get("train_gt_count", ""),
            "valid_gt_count": row.get("valid_gt_count", ""),
            "test_gt_count": row.get("test_gt_count", ""),
            "base_groups": row.get("base_groups", ""),
            "base_hit_rate": row.get("base_hit_rate", ""),
            "base_mean_nearest_iou": row.get("base_mean_nearest_iou", ""),
            "latest_groups": row.get("latest_groups", ""),
            "latest_hit_rate": row.get("latest_hit_rate", ""),
            "latest_mean_nearest_iou": row.get("latest_mean_nearest_iou", ""),
            "top_wrong_pred_category_name": row.get("top_wrong_pred_category_name", ""),
            "top_wrong_pair_count": row.get("top_wrong_pair_count", ""),
            "top_wrong_mean_iou": row.get("top_wrong_mean_iou", ""),
        }
        for label in labels:
            flow = flows.get((category_id, label), {})
            merged[f"{label}_fixed_samples"] = flow.get("samples", "0")
            merged[f"{label}_fixed_gt_candidate_present"] = flow.get("gt_candidate_present", "0")
            merged[f"{label}_fixed_present_rate"] = flow.get("gt_candidate_present_rate", "0")
            merged[f"{label}_fixed_mean_iou"] = flow.get("mean_nearest_iou", "0")
            merged[f"{label}_fixed_dominant_top_candidate"] = flow.get("dominant_top_candidate", "")
            merged[f"{label}_fixed_dominant_top_count"] = flow.get("dominant_top_candidate_count", "0")
        out.append(merged)
    out.sort(
        key=lambda row: (
            transition_rank(str(row["transition"])),
            -parse_int(row.get("test_gt_count", 0)),
            -parse_int(row.get("train_gt_count", 0)),
            -parse_float(row.get("latest_mean_nearest_iou", 0)),
            str(row["category_name"]),
        )
    )
    return out


def transition_rank(label: str) -> int:
    return {
        "persistent_zero_hit": 0,
        "regressed_to_zero_hit": 1,
    }.get(label, 99)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value: Any) -> int:
    return int(float(value or "0"))


def parse_float(value: Any) -> float:
    return float(value or "0")


if __name__ == "__main__":
    main()
