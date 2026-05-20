"""Summarize COCO category-source predictions with AP and coverage-gap metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_coco_category_coverage_gap import analyze_category_coverage_gap
from scripts.evaluate_coco_predictions import evaluate_coco_predictions
from scripts.summarize_coco_eval_table import parse_entry


FIELDNAMES = (
    "name",
    "source",
    "predictions",
    "ap",
    "ap50",
    "ap75",
    "global_topk_loc_recall50",
    "global_topk_class_recall50",
    "per_category_topk_class_recall50",
    "global_category_retention50",
    "ranking_gap50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Named prediction JSON in the form name=/path/to/predictions.json.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_category_sources(args.annotations, args.entry, top_k=args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_category_source_summary: {args.out}")
    for row in rows:
        print(
            f"{row['name']}: ap50={row['ap50']:.4f} "
            f"class_recall50={row['global_topk_class_recall50']:.4f} "
            f"retention={row['global_category_retention50']:.4f}"
        )


def summarize_category_sources(
    annotations: Path,
    entries: list[str],
    *,
    top_k: int,
) -> list[dict[str, float | int | str]]:
    rows = []
    for entry in entries:
        name, path = parse_entry(entry)
        metrics = evaluate_coco_predictions(annotations, path)
        coverage = analyze_category_coverage_gap(
            annotation_json=annotations,
            prediction_json=path,
            top_ks=(top_k,),
            iou_thresholds=(0.5,),
            slices=("all",),
        )[0]
        rows.append(
            {
                "name": name,
                "source": str(path),
                "predictions": len(json.loads(path.read_text(encoding="utf-8"))),
                "ap": metrics["ap"],
                "ap50": metrics["ap50"],
                "ap75": metrics["ap75"],
                "global_topk_loc_recall50": coverage["global_topk_loc_recall"],
                "global_topk_class_recall50": coverage["global_topk_class_recall"],
                "per_category_topk_class_recall50": coverage["per_category_topk_class_recall"],
                "global_category_retention50": coverage["global_category_retention"],
                "ranking_gap50": coverage["ranking_gap"],
            }
        )
    return rows


if __name__ == "__main__":
    main()
