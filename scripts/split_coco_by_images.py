"""Split a COCO annotation file into image-disjoint subsets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-a", type=Path, required=True)
    parser.add_argument("--out-b", type=Path, required=True)
    parser.add_argument("--a-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--keep-all-categories",
        action="store_true",
        help="Preserve the original category table in both outputs instead of dropping absent categories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    a_data, b_data = split_coco_by_images(
        data,
        a_fraction=args.a_fraction,
        seed=args.seed,
        keep_all_categories=args.keep_all_categories,
    )
    args.out_a.parent.mkdir(parents=True, exist_ok=True)
    args.out_b.parent.mkdir(parents=True, exist_ok=True)
    args.out_a.write_text(json.dumps(a_data, indent=2) + "\n", encoding="utf-8")
    args.out_b.write_text(json.dumps(b_data, indent=2) + "\n", encoding="utf-8")
    print(f"saved_a: {args.out_a} images={len(a_data['images'])} annotations={len(a_data['annotations'])}")
    print(f"saved_b: {args.out_b} images={len(b_data['images'])} annotations={len(b_data['annotations'])}")


def split_coco_by_images(
    data: dict[str, Any],
    *,
    a_fraction: float,
    seed: int,
    keep_all_categories: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 < a_fraction < 1.0:
        raise ValueError("a_fraction must be between 0 and 1")
    image_ids = sorted(int(image["id"]) for image in data.get("images", []))
    shuffled = list(image_ids)
    random.Random(seed).shuffle(shuffled)
    a_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * a_fraction)))
    a_ids = set(shuffled[:a_count])
    b_ids = set(shuffled[a_count:])
    return (
        subset_coco(data, a_ids, keep_all_categories=keep_all_categories),
        subset_coco(data, b_ids, keep_all_categories=keep_all_categories),
    )


def subset_coco(data: dict[str, Any], image_ids: set[int], *, keep_all_categories: bool = False) -> dict[str, Any]:
    images = [image for image in data.get("images", []) if int(image["id"]) in image_ids]
    annotations = [
        annotation
        for annotation in data.get("annotations", [])
        if int(annotation["image_id"]) in image_ids
    ]
    if keep_all_categories:
        categories = list(data.get("categories", []))
    else:
        kept_categories = {int(annotation["category_id"]) for annotation in annotations}
        categories = [
            category for category in data.get("categories", []) if int(category["id"]) in kept_categories
        ]
    return {
        **{key: value for key, value in data.items() if key not in {"images", "annotations", "categories"}},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


if __name__ == "__main__":
    main()
