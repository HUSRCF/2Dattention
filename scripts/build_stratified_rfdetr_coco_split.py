"""Build RF-DETR-style image-disjoint COCO splits with train category coverage."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=1000)
    parser.add_argument("--valid-images", type=int, default=200)
    parser.add_argument("--test-images", type=int, default=200)
    parser.add_argument("--min-train-boxes-per-category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of making symlinks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    splits = build_stratified_splits(
        data,
        train_images=args.train_images,
        valid_images=args.valid_images,
        test_images=args.test_images,
        min_train_boxes_per_category=args.min_train_boxes_per_category,
        seed=args.seed,
    )
    for split_name, image_ids in splits.items():
        split_data = subset_coco(data, image_ids)
        split_dir = args.out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(split_data, indent=2) + "\n",
            encoding="utf-8",
        )
        link_or_copy_images(split_data["images"], args.image_root, split_dir, copy_images=args.copy_images)
        print(
            f"saved_{split_name}: images={len(split_data['images'])} "
            f"annotations={len(split_data['annotations'])} path={split_dir}"
        )
    audit = audit_split(data, splits, min_train_boxes_per_category=args.min_train_boxes_per_category)
    audit_path = args.out_dir / "split_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved_audit: {audit_path}")
    print(
        "train_category_coverage:",
        f"zero={audit['train_categories_with_zero_boxes']}",
        f"lt_min={audit['train_categories_below_min']}",
        f"min={audit['train_min_boxes_per_category']}",
    )


def build_stratified_splits(
    data: dict[str, Any],
    *,
    train_images: int,
    valid_images: int,
    test_images: int,
    min_train_boxes_per_category: int,
    seed: int,
) -> dict[str, set[int]]:
    image_ids = [int(image["id"]) for image in data.get("images", [])]
    total_requested = train_images + valid_images + test_images
    if total_requested > len(image_ids):
        raise ValueError(f"requested {total_requested} images from only {len(image_ids)} available")
    rng = random.Random(seed)
    category_ids = sorted(int(category["id"]) for category in data.get("categories", []))
    image_annotations: dict[int, list[int]] = {image_id: [] for image_id in image_ids}
    for annotation in data.get("annotations", []):
        image_annotations[int(annotation["image_id"])].append(int(annotation["category_id"]))
    remaining = set(image_ids)
    train_ids: set[int] = set()
    train_counts = {category_id: 0 for category_id in category_ids}

    # Greedily select images that cover currently underrepresented categories.
    while len(train_ids) < train_images:
        undercovered = {
            category_id
            for category_id, count in train_counts.items()
            if count < min_train_boxes_per_category
        }
        if not undercovered:
            break
        best_image_id = None
        best_score = (-1, -1, 0.0)
        for image_id in remaining:
            labels = image_annotations.get(image_id, [])
            gain = sum(1 for label in labels if label in undercovered)
            if gain <= 0:
                continue
            diversity = len({label for label in labels if label in undercovered})
            score = (gain, diversity, rng.random())
            if score > best_score:
                best_score = score
                best_image_id = image_id
        if best_image_id is None:
            break
        add_train_image(best_image_id, train_ids, remaining, train_counts, image_annotations)

    # Fill train/valid/test sizes with deterministic random image choices.
    fill_split(train_ids, remaining, train_images, rng)
    valid_ids: set[int] = set()
    fill_split(valid_ids, remaining, valid_images, rng)
    test_ids: set[int] = set()
    fill_split(test_ids, remaining, test_images, rng)
    return {"train": train_ids, "valid": valid_ids, "test": test_ids}


def add_train_image(
    image_id: int,
    train_ids: set[int],
    remaining: set[int],
    train_counts: dict[int, int],
    image_annotations: dict[int, list[int]],
) -> None:
    train_ids.add(image_id)
    remaining.remove(image_id)
    for category_id in image_annotations.get(image_id, []):
        train_counts[category_id] = train_counts.get(category_id, 0) + 1


def fill_split(target: set[int], remaining: set[int], target_size: int, rng: random.Random) -> None:
    needed = target_size - len(target)
    if needed <= 0:
        return
    choices = list(remaining)
    rng.shuffle(choices)
    for image_id in choices[:needed]:
        target.add(image_id)
        remaining.remove(image_id)


def subset_coco(data: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    images = [image for image in data.get("images", []) if int(image["id"]) in image_ids]
    annotations = [
        annotation
        for annotation in data.get("annotations", [])
        if int(annotation["image_id"]) in image_ids
    ]
    return {
        **{key: value for key, value in data.items() if key not in {"images", "annotations", "categories"}},
        "images": images,
        "annotations": annotations,
        "categories": list(data.get("categories", [])),
    }


def link_or_copy_images(
    images: list[dict[str, Any]],
    image_root: Path,
    split_dir: Path,
    *,
    copy_images: bool,
) -> None:
    for image in images:
        source = (image_root / image["file_name"]).resolve()
        target = split_dir / image["file_name"]
        if target.exists() or target.is_symlink():
            continue
        if copy_images:
            target.write_bytes(source.read_bytes())
        else:
            os.symlink(source, target)


def audit_split(
    data: dict[str, Any],
    splits: dict[str, set[int]],
    *,
    min_train_boxes_per_category: int,
) -> dict[str, Any]:
    category_ids = sorted(int(category["id"]) for category in data.get("categories", []))
    annotations = data.get("annotations", [])
    split_counts = {}
    for split_name, image_ids in splits.items():
        counts = {category_id: 0 for category_id in category_ids}
        box_count = 0
        for annotation in annotations:
            if int(annotation["image_id"]) not in image_ids:
                continue
            counts[int(annotation["category_id"])] += 1
            box_count += 1
        split_counts[split_name] = {
            "images": len(image_ids),
            "boxes": box_count,
            "categories_with_boxes": sum(count > 0 for count in counts.values()),
            "min_boxes_per_category": min(counts.values()) if counts else 0,
            "categories_with_zero_boxes": sum(count == 0 for count in counts.values()),
            "categories_below_min_boxes": sum(count < min_train_boxes_per_category for count in counts.values()),
        }
    train_counts = split_counts["train"]
    return {
        "splits": split_counts,
        "train_categories_with_zero_boxes": train_counts["categories_with_zero_boxes"],
        "train_categories_below_min": train_counts["categories_below_min_boxes"],
        "train_min_boxes_per_category": train_counts["min_boxes_per_category"],
    }


if __name__ == "__main__":
    main()
