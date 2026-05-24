"""Build high-IoU hard semantic-confusion pairs from a prediction chain."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--instances-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument(
        "--missing-mode",
        choices=["group", "displayed"],
        default="group",
        help="Filter cases where GT category is absent from the full group or only absent from displayed candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_instances(
        read_rows(args.chain),
        min_iou=args.min_iou,
        missing_mode=args.missing_mode,
    )
    summary = summarize_pairs(rows)
    write_rows(rows, args.instances_out)
    write_rows(summary, args.summary_out)
    print(f"saved_hard_instances: {args.instances_out}")
    print(f"saved_hard_summary: {args.summary_out}")


def build_instances(
    rows: list[dict[str, str]],
    *,
    min_iou: float,
    missing_mode: str,
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if parse_float(row.get("nearest_iou", "0")) < min_iou:
            continue
        if missing_mode == "group" and parse_int(row.get("gt_candidate_present", "0")):
            continue
        if missing_mode == "displayed" and parse_int(row.get("gt_candidate_present_displayed", "0")):
            continue
        candidate = top_candidate(row.get("top_candidates", ""))
        if not candidate["category"]:
            continue
        out.append(
            {
                "label": row["label"],
                "image_id": row["image_id"],
                "annotation_id": row["annotation_id"],
                "gt_category_id": row["category_id"],
                "gt_category_name": row["category_name"],
                "transition": row.get("transition", ""),
                "top_wrong_candidate": candidate["category"],
                "top_wrong_score": format_float(candidate["score"]),
                "nearest_iou": row["nearest_iou"],
                "nearest_group_rank": row.get("nearest_group_rank", ""),
                "nearest_group_top_score": row.get("nearest_group_top_score", ""),
                "nearest_group_size": row.get("nearest_group_size", ""),
                "gt_candidate_present": row.get("gt_candidate_present", ""),
                "gt_candidate_rank_in_group": row.get("gt_candidate_rank_in_group", ""),
                "gt_candidate_present_displayed": row.get("gt_candidate_present_displayed", ""),
            }
        )
    out.sort(
        key=lambda row: (
            str(row["label"]),
            str(row["gt_category_name"]),
            str(row["top_wrong_candidate"]),
            -parse_float(row["nearest_iou"]),
            parse_int(row["image_id"]),
            parse_int(row["annotation_id"]),
        )
    )
    return out


def summarize_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["label"]),
                str(row["gt_category_id"]),
                str(row["gt_category_name"]),
                str(row["transition"]),
                str(row["top_wrong_candidate"]),
            )
        ].append(row)
    out = []
    for (label, gt_category_id, gt_category_name, transition, top_wrong_candidate), group in groups.items():
        out.append(
            {
                "label": label,
                "gt_category_id": gt_category_id,
                "gt_category_name": gt_category_name,
                "transition": transition,
                "top_wrong_candidate": top_wrong_candidate,
                "samples": len(group),
                "mean_nearest_iou": format_float(mean(parse_float(row["nearest_iou"]) for row in group)),
                "mean_nearest_group_rank": format_float(
                    mean(parse_float(row.get("nearest_group_rank", 0)) for row in group)
                ),
                "mean_top_wrong_score": format_float(
                    mean(parse_float(row["top_wrong_score"]) for row in group)
                ),
                "example_image_ids": ";".join(str(row["image_id"]) for row in group[:5]),
                "example_annotation_ids": ";".join(str(row["annotation_id"]) for row in group[:5]),
            }
        )
    out.sort(
        key=lambda row: (
            str(row["label"]),
            -parse_int(row["samples"]),
            -parse_float(row["mean_nearest_iou"]),
            str(row["gt_category_name"]),
            str(row["top_wrong_candidate"]),
        )
    )
    return out


def top_candidate(text: str) -> dict[str, Any]:
    first = text.splitlines()[0].strip() if text.strip() else ""
    if not first:
        return {"category": "", "score": 0.0}
    if ":" not in first:
        return {"category": first, "score": 0.0}
    category, score = first.rsplit(":", 1)
    return {"category": category, "score": parse_float(score)}


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


def mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def parse_int(value: Any) -> int:
    return int(float(value or "0"))


def parse_float(value: Any) -> float:
    return float(value or "0")


def format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    main()
