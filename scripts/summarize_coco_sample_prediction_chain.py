"""Summarize fixed COCO sample prediction quality across checkpoint predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize_coco_sample_predictions import format_candidates
from scripts.visualize_coco_sample_predictions import group_predictions_by_image
from scripts.visualize_coco_sample_predictions import nearest_prediction_group
from scripts.visualize_coco_sample_predictions import parse_float
from scripts.visualize_coco_sample_predictions import parse_int
from scripts.visualize_coco_sample_predictions import read_rows
from scripts.visualize_coco_sample_predictions import select_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--prediction-entry",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Prediction JSON entry. Can be repeated.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--bbox-decimals", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-per-category", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = load_categories(args.annotations)
    samples = select_rows(
        read_rows(args.samples),
        max_samples=args.max_samples,
        max_per_category=args.max_per_category,
    )
    entries = [(label, load_predictions(path)) for label, path in parse_entries(args.prediction_entry)]

    detail_rows = []
    summary_rows = []
    for label, predictions_by_image in entries:
        label_rows = []
        for sample in samples:
            nearest = nearest_prediction_group(
                sample,
                predictions_by_image.get(parse_int(sample["image_id"]), [])[: args.top_k],
                bbox_decimals=args.bbox_decimals,
            )
            row = {
                "label": label,
                "image_id": sample["image_id"],
                "annotation_id": sample["annotation_id"],
                "category_id": sample["category_id"],
                "category_name": sample["category_name"],
                "transition": sample.get("transition", ""),
                "area_ratio": sample.get("area_ratio", ""),
                "nearest_iou": format_float(nearest.iou if nearest else 0.0),
                "iou_ge_05": int(bool(nearest and nearest.iou >= 0.5)),
                "iou_ge_075": int(bool(nearest and nearest.iou >= 0.75)),
                "gt_candidate_present": int(bool(nearest and nearest.gt_candidate_present)),
                "top_candidates": format_candidates(nearest.candidates[: args.top_candidates], categories)
                if nearest
                else "",
            }
            detail_rows.append(row)
            label_rows.append(row)
        summary_rows.append(summarize_label(label, label_rows))

    write_rows(detail_rows, args.out)
    write_rows(summary_rows, args.summary_out)
    print(f"saved_detail: {args.out}")
    print(f"saved_summary: {args.summary_out}")


def parse_entries(entries: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"prediction entry must be LABEL=PATH: {entry}")
        label, path = entry.split("=", 1)
        if not label:
            raise ValueError(f"prediction entry label is empty: {entry}")
        parsed.append((label, Path(path)))
    return parsed


def load_categories(path: Path) -> dict[int, str]:
    annotations = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in annotations.get("categories", [])
    }


def load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    return group_predictions_by_image(json.loads(path.read_text(encoding="utf-8")))


def summarize_label(label: str, rows: list[dict[str, object]]) -> dict[str, object]:
    count = len(rows)
    ious = [parse_float(row["nearest_iou"]) for row in rows]
    return {
        "label": label,
        "samples": count,
        "mean_nearest_iou": format_float(sum(ious) / count if count else 0.0),
        "iou_ge_05": sum(parse_int(row["iou_ge_05"]) for row in rows),
        "iou_ge_075": sum(parse_int(row["iou_ge_075"]) for row in rows),
        "gt_candidate_present": sum(parse_int(row["gt_candidate_present"]) for row in rows),
        "gt_candidate_present_rate": format_float(
            sum(parse_int(row["gt_candidate_present"]) for row in rows) / count if count else 0.0
        ),
    }


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    main()
