"""Build a semantic repair config from RF-DETR hard confusion pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_rfdetr_hard_confusion_pairs import parse_float
from scripts.build_rfdetr_hard_confusion_pairs import parse_int
from scripts.build_rfdetr_hard_confusion_pairs import read_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-pair-summary", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--max-negatives-per-class", type=int, default=5)
    parser.add_argument("--max-weight", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(
        hard_pair_rows=read_rows(args.hard_pair_summary),
        annotations=json.loads(args.annotations.read_text(encoding="utf-8")),
        min_samples=args.min_samples,
        min_iou=args.min_iou,
        max_negatives_per_class=args.max_negatives_per_class,
        max_weight=args.max_weight,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved_semantic_repair_config: {args.out}")


def build_config(
    *,
    hard_pair_rows: list[dict[str, str]],
    annotations: dict[str, Any],
    min_samples: int,
    min_iou: float,
    max_negatives_per_class: int,
    max_weight: float,
) -> dict[str, Any]:
    name_to_id = {str(row["name"]): int(row["id"]) for row in annotations.get("categories", [])}
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in hard_pair_rows:
        samples = parse_int(row["samples"])
        mean_iou = parse_float(row["mean_nearest_iou"])
        if samples < min_samples or mean_iou < min_iou:
            continue
        gt_name = row["gt_category_name"]
        wrong_name = row["top_wrong_candidate"]
        if gt_name not in name_to_id:
            raise ValueError(f"missing GT category in annotations: {gt_name}")
        if wrong_name not in name_to_id:
            raise ValueError(f"missing wrong category in annotations: {wrong_name}")
        key = (gt_name, wrong_name)
        item = by_pair.setdefault(
            key,
            {
                "negative_category_id": name_to_id[wrong_name],
                "negative_category_name": wrong_name,
                "source_labels": [],
                "samples": 0,
                "weighted_iou_sum": 0.0,
                "weighted_rank_sum": 0.0,
                "weighted_score_sum": 0.0,
                "example_image_ids": [],
                "example_annotation_ids": [],
            },
        )
        item["source_labels"].append(row["label"])
        item["samples"] += samples
        item["weighted_iou_sum"] += samples * mean_iou
        item["weighted_rank_sum"] += samples * parse_float(row["mean_nearest_group_rank"])
        item["weighted_score_sum"] += samples * parse_float(row["mean_top_wrong_score"])
        item["example_image_ids"].extend(split_ids(row.get("example_image_ids", "")))
        item["example_annotation_ids"].extend(split_ids(row.get("example_annotation_ids", "")))

    by_gt: dict[str, list[dict[str, Any]]] = {}
    for (gt_name, _wrong_name), item in by_pair.items():
        samples = parse_int(item["samples"])
        mean_iou = parse_float(item["weighted_iou_sum"]) / samples if samples else 0.0
        weight = min(max_weight, samples * mean_iou)
        by_gt.setdefault(gt_name, []).append(
            {
                "negative_category_id": item["negative_category_id"],
                "negative_category_name": item["negative_category_name"],
                "source_labels": sorted(set(item["source_labels"])),
                "samples": samples,
                "mean_nearest_iou": round(mean_iou, 6),
                "mean_nearest_group_rank": round(
                    parse_float(item["weighted_rank_sum"]) / samples if samples else 0.0,
                    6,
                ),
                "mean_top_wrong_score": round(
                    parse_float(item["weighted_score_sum"]) / samples if samples else 0.0,
                    6,
                ),
                "weight": round(weight, 6),
                "example_image_ids": unique_ints(item["example_image_ids"])[:5],
                "example_annotation_ids": unique_ints(item["example_annotation_ids"])[:5],
            }
        )

    targets = []
    for gt_name, negatives in by_gt.items():
        negatives.sort(
            key=lambda row: (
                -parse_float(row["weight"]),
                -parse_int(row["samples"]),
                parse_float(row["mean_nearest_group_rank"]),
                row["negative_category_name"],
            )
        )
        kept = negatives[:max_negatives_per_class] if max_negatives_per_class > 0 else negatives
        total_weight = sum(parse_float(row["weight"]) for row in kept)
        targets.append(
            {
                "category_id": name_to_id[gt_name],
                "category_name": gt_name,
                "hard_negative_count": len(kept),
                "total_weight": round(total_weight, 6),
                "hard_negatives": kept,
            }
        )
    targets.sort(key=lambda row: (-parse_float(row["total_weight"]), row["category_name"]))
    return {
        "task": "rfdetr_semantic_candidate_repair",
        "source": "hard_confusion_pairs",
        "parameters": {
            "min_samples": min_samples,
            "min_iou": min_iou,
            "max_negatives_per_class": max_negatives_per_class,
            "max_weight": max_weight,
        },
        "target_count": len(targets),
        "targets": targets,
    }


def split_ids(text: str) -> list[int]:
    if not text:
        return []
    return [parse_int(value) for value in text.split(";") if value]


def unique_ints(values: list[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


if __name__ == "__main__":
    main()
