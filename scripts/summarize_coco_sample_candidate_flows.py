"""Aggregate top candidate category flows from a fixed-sample prediction chain."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--flow-out", type=Path, required=True)
    parser.add_argument("--category-out", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.chain)
    flow_rows = summarize_flows(rows, min_iou=args.min_iou)
    category_rows = summarize_categories(rows, min_iou=args.min_iou)
    write_rows(flow_rows, args.flow_out)
    write_rows(category_rows, args.category_out)
    print(f"saved_flow_summary: {args.flow_out}")
    print(f"saved_category_summary: {args.category_out}")


def summarize_flows(rows: list[dict[str, str]], *, min_iou: float) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        top = top_candidate(row.get("top_candidates", ""))
        groups[(row["label"], row["category_id"], row["category_name"], top["category"])].append(row)
    out = []
    for (label, category_id, category_name, top_category), group in groups.items():
        out.append(
            {
                "label": label,
                "gt_category_id": category_id,
                "gt_category_name": category_name,
                "top_candidate": top_category,
                "samples": len(group),
                "gt_candidate_present": sum(parse_int(row["gt_candidate_present"]) for row in group),
                "mean_nearest_iou": format_float(mean(parse_float(row["nearest_iou"]) for row in group)),
                "iou_ge_min": sum(parse_float(row["nearest_iou"]) >= min_iou for row in group),
                "mean_top_candidate_score": format_float(
                    mean(top_candidate(row.get("top_candidates", ""))["score"] for row in group)
                ),
                "example_image_ids": ";".join(row["image_id"] for row in group[:5]),
            }
        )
    out.sort(
        key=lambda row: (
            str(row["label"]),
            -parse_int(row["samples"]),
            -parse_float(row["mean_nearest_iou"]),
            str(row["gt_category_name"]),
            str(row["top_candidate"]),
        )
    )
    return out


def summarize_categories(rows: list[dict[str, str]], *, min_iou: float) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["category_id"], row["category_name"])].append(row)
    out = []
    for (label, category_id, category_name), group in groups.items():
        top_counts: dict[str, int] = defaultdict(int)
        for row in group:
            top_counts[top_candidate(row.get("top_candidates", ""))["category"]] += 1
        dominant_top, dominant_count = sorted(top_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        out.append(
            {
                "label": label,
                "gt_category_id": category_id,
                "gt_category_name": category_name,
                "samples": len(group),
                "gt_candidate_present": sum(parse_int(row["gt_candidate_present"]) for row in group),
                "gt_candidate_present_rate": format_float(
                    sum(parse_int(row["gt_candidate_present"]) for row in group) / len(group)
                ),
                "mean_nearest_iou": format_float(mean(parse_float(row["nearest_iou"]) for row in group)),
                "iou_ge_min": sum(parse_float(row["nearest_iou"]) >= min_iou for row in group),
                "dominant_top_candidate": dominant_top,
                "dominant_top_candidate_count": dominant_count,
            }
        )
    out.sort(
        key=lambda row: (
            str(row["label"]),
            parse_int(row["gt_candidate_present"]),
            -parse_float(row["mean_nearest_iou"]),
            str(row["gt_category_name"]),
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
