"""Fuse two category-expanded COCO prediction JSONs for the same proposal boxes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("geomean", "multiply", "primary_plus_secondary"), default="geomean")
    parser.add_argument("--bbox-decimals", type=int, default=3)
    parser.add_argument(
        "--keep-unmatched-primary",
        action="store_true",
        help="Keep primary predictions when the same image/bbox/category is absent from secondary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fused = fuse_category_prior_predictions(
        primary_predictions=json.loads(args.primary.read_text(encoding="utf-8")),
        secondary_predictions=json.loads(args.secondary.read_text(encoding="utf-8")),
        mode=args.mode,
        bbox_decimals=args.bbox_decimals,
        keep_unmatched_primary=args.keep_unmatched_primary,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fused, indent=2) + "\n", encoding="utf-8")
    print(f"saved_fused_category_prior_predictions: {args.out}")
    print(f"predictions: {len(fused)}")


def fuse_category_prior_predictions(
    *,
    primary_predictions: list[dict[str, Any]],
    secondary_predictions: list[dict[str, Any]],
    mode: str,
    bbox_decimals: int = 3,
    keep_unmatched_primary: bool = False,
) -> list[dict[str, Any]]:
    secondary_by_key = {
        prediction_key(row, bbox_decimals=bbox_decimals): float(row.get("score", 0.0))
        for row in secondary_predictions
    }
    rows = []
    for primary in primary_predictions:
        key = prediction_key(primary, bbox_decimals=bbox_decimals)
        secondary_score = secondary_by_key.get(key)
        if secondary_score is None and not keep_unmatched_primary:
            continue
        primary_score = float(primary.get("score", 0.0))
        fused_score = fused_score_from_pair(
            primary_score,
            primary_score if secondary_score is None else secondary_score,
            mode=mode,
        )
        row = dict(primary)
        row["score"] = fused_score
        rows.append(row)
    rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return rows


def prediction_key(prediction: dict[str, Any], *, bbox_decimals: int) -> tuple[Any, ...]:
    return (
        int(prediction["image_id"]),
        int(prediction["category_id"]),
        *(round(float(value), bbox_decimals) for value in prediction["bbox"]),
    )


def fused_score_from_pair(primary_score: float, secondary_score: float, *, mode: str) -> float:
    primary_score = max(0.0, float(primary_score))
    secondary_score = max(0.0, float(secondary_score))
    if mode == "geomean":
        return math.sqrt(primary_score * secondary_score)
    if mode == "multiply":
        return primary_score * secondary_score
    if mode == "primary_plus_secondary":
        return 0.5 * (primary_score + secondary_score)
    raise ValueError(f"unsupported fusion mode: {mode}")


if __name__ == "__main__":
    main()
