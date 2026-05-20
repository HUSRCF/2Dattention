"""Summarize proposal recall, RPN localization AP, and anchor-grid oracle metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDNAMES = (
    "name",
    "source_type",
    "slice",
    "recall_top_k",
    "recall_iou_threshold",
    "recall",
    "mean_best_iou",
    "ap",
    "ap50",
    "ap75",
    "gt_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--recall",
        nargs=3,
        action="append",
        metavar=("NAME", "SOURCE_TYPE", "CSV"),
        default=[],
        help="Add a proposal-recall CSV from evaluate_torchvision_proposal_recall or evaluate_anchor_grid_recall.",
    )
    parser.add_argument(
        "--ap",
        nargs=2,
        action="append",
        metavar=("NAME", "CSV"),
        default=[],
        help="Add a slice AP CSV from evaluate_coco_slices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_proposal_diagnostics(
        recall_inputs=[(name, source_type, Path(path)) for name, source_type, path in args.recall],
        ap_inputs=[(name, Path(path)) for name, path in args.ap],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_proposal_diagnostics: {args.out}")


def summarize_proposal_diagnostics(
    recall_inputs: list[tuple[str, str, Path]],
    ap_inputs: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, source_type, path in recall_inputs:
        rows.extend(recall_rows(name=name, source_type=source_type, path=path))
    for name, path in ap_inputs:
        rows.extend(ap_rows(name=name, path=path))
    return rows


def recall_rows(name: str, source_type: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        rows.append(
            {
                "name": name,
                "source_type": source_type,
                "slice": row["slice"],
                "recall_top_k": row["top_k"],
                "recall_iou_threshold": row["iou_threshold"],
                "recall": row["recall"],
                "mean_best_iou": row["mean_best_iou"],
                "ap": "",
                "ap50": "",
                "ap75": "",
                "gt_count": row.get("gt_count", ""),
            }
        )
    return rows


def ap_rows(name: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        rows.append(
            {
                "name": name,
                "source_type": "rpn_objectness_ap",
                "slice": row["slice"],
                "recall_top_k": "",
                "recall_iou_threshold": "",
                "recall": "",
                "mean_best_iou": "",
                "ap": row["ap"],
                "ap50": row["ap50"],
                "ap75": row["ap75"],
                "gt_count": row.get("annotations", ""),
            }
        )
    return rows


if __name__ == "__main__":
    main()
