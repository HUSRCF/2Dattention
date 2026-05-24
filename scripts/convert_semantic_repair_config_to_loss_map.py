"""Convert semantic repair config category ids into loss class indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--category-id-offset",
        type=int,
        required=True,
        help="Subtract this from category ids to get logits class indices. Use 1 for COCO ids 1..N.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    loss_map = build_loss_map(config, category_id_offset=args.category_id_offset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(loss_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved_loss_map: {args.out}")


def build_loss_map(config: dict[str, Any], *, category_id_offset: int) -> dict[str, Any]:
    entries = []
    for target in config.get("targets", []):
        positive_id = int(target["category_id"])
        positive_index = positive_id - category_id_offset
        if positive_index < 0:
            raise ValueError(f"positive index is negative for category_id={positive_id}")
        negatives = []
        for negative in target.get("hard_negatives", []):
            negative_id = int(negative["negative_category_id"])
            negative_index = negative_id - category_id_offset
            if negative_index < 0:
                raise ValueError(f"negative index is negative for category_id={negative_id}")
            negatives.append(
                {
                    "negative_category_id": negative_id,
                    "negative_class_index": negative_index,
                    "negative_category_name": negative["negative_category_name"],
                    "weight": float(negative["weight"]),
                    "samples": int(negative["samples"]),
                }
            )
        if negatives:
            entries.append(
                {
                    "positive_category_id": positive_id,
                    "positive_class_index": positive_index,
                    "positive_category_name": target["category_name"],
                    "hard_negatives": negatives,
                }
            )
    return {
        "task": "semantic_hard_negative_loss_map",
        "category_id_offset": category_id_offset,
        "entry_count": len(entries),
        "entries": entries,
    }


if __name__ == "__main__":
    main()
