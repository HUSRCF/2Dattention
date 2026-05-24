"""Build a COCO/RF-DETR dataset with hard-category train oversampling.

The script duplicates train image records for categories selected from a
per-category diagnostic CSV, while preserving the original valid/test splits.
Image files are symlinked by default, and duplicated image records keep the same
file_name with new COCO image ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--hard-category-csv",
        type=Path,
        required=True,
        help="CSV with category_id, groups, and candidate_hit_rate columns.",
    )
    parser.add_argument("--hard-min-groups", type=int, default=20)
    parser.add_argument("--hard-max-hit-rate", type=float, default=0.0)
    parser.add_argument("--target-hard-boxes", type=int, default=60)
    parser.add_argument("--max-repeat", type=int, default=5)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_repeat < 1:
        raise ValueError("--max-repeat must be >= 1")
    hard_categories = read_hard_categories(
        args.hard_category_csv,
        min_groups=args.hard_min_groups,
        max_hit_rate=args.hard_max_hit_rate,
    )
    source_train = args.source_dataset_dir / "train" / "_annotations.coco.json"
    train_data = json.loads(source_train.read_text(encoding="utf-8"))
    reserved_image_ids = collect_dataset_image_ids(args.source_dataset_dir)
    oversampled, summary_rows = build_oversampled_train(
        train_data,
        hard_categories=hard_categories,
        target_hard_boxes=args.target_hard_boxes,
        max_repeat=args.max_repeat,
        reserved_image_ids=reserved_image_ids,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_split(
        source_split_dir=args.source_dataset_dir / "train",
        out_split_dir=args.out_dir / "train",
        annotation_data=oversampled,
        copy_images=args.copy_images,
    )
    for split in ("valid", "test"):
        data = json.loads((args.source_dataset_dir / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        write_split(
            source_split_dir=args.source_dataset_dir / split,
            out_split_dir=args.out_dir / split,
            annotation_data=data,
            copy_images=args.copy_images,
        )
    summary_path = args.summary or (args.out_dir / "hard_category_oversample_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved_oversampled_dataset: {args.out_dir}")
    print(f"saved_summary: {summary_path}")
    print(
        f"hard_categories={len(hard_categories)} "
        f"train_images={len(train_data['images'])}->{len(oversampled['images'])} "
        f"train_annotations={len(train_data['annotations'])}->{len(oversampled['annotations'])}"
    )


def read_hard_categories(path: Path, *, min_groups: int, max_hit_rate: float) -> set[int]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    hard = set()
    for row in rows:
        groups = int(float(row.get("groups", 0)))
        hit_rate = float(row.get("candidate_hit_rate", 1.0))
        if groups >= min_groups and hit_rate <= max_hit_rate:
            hard.add(int(row["category_id"]))
    if not hard:
        raise ValueError(f"no hard categories selected from {path}")
    return hard


def build_oversampled_train(
    data: dict[str, Any],
    *,
    hard_categories: set[int],
    target_hard_boxes: int,
    max_repeat: int,
    reserved_image_ids: set[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    counts: dict[int, int] = {}
    for annotation in data.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        annotations_by_image.setdefault(image_id, []).append(annotation)
        counts[category_id] = counts.get(category_id, 0) + 1

    all_reserved_image_ids = set(reserved_image_ids or set())
    all_reserved_image_ids.update(int(image["id"]) for image in data.get("images", []))
    next_image_id = max(all_reserved_image_ids, default=0) + 1
    next_annotation_id = max((int(row.get("id", 0)) for row in data.get("annotations", [])), default=0) + 1
    images_out: list[dict[str, Any]] = []
    annotations_out: list[dict[str, Any]] = []
    repeated_by_category = {category_id: 0 for category_id in hard_categories}
    original_by_category = {category_id: counts.get(category_id, 0) for category_id in hard_categories}
    repeat_histogram: dict[int, int] = {}

    for image in data.get("images", []):
        image_id = int(image["id"])
        image_annotations = annotations_by_image.get(image_id, [])
        image_categories = {int(row["category_id"]) for row in image_annotations}
        selected_categories = image_categories.intersection(hard_categories)
        repeat = 1
        if selected_categories:
            repeat = max(
                1,
                min(
                    max_repeat,
                    max(math.ceil(target_hard_boxes / max(1, counts.get(category_id, 0))) for category_id in selected_categories),
                ),
            )
        repeat_histogram[repeat] = repeat_histogram.get(repeat, 0) + 1
        for copy_index in range(repeat):
            if copy_index == 0:
                new_image_id = image_id
            else:
                new_image_id = next_image_id
                next_image_id += 1
            image_copy = dict(image)
            image_copy["id"] = new_image_id
            if copy_index:
                image_copy["source_image_id"] = image_id
                image_copy["repeat_index"] = copy_index
            images_out.append(image_copy)
            for annotation in image_annotations:
                annotation_copy = dict(annotation)
                if copy_index == 0:
                    annotation_copy["id"] = int(annotation.get("id", next_annotation_id))
                else:
                    annotation_copy["id"] = next_annotation_id
                    next_annotation_id += 1
                    annotation_copy["source_annotation_id"] = int(annotation.get("id", -1))
                annotation_copy["image_id"] = new_image_id
                annotations_out.append(annotation_copy)
                category_id = int(annotation_copy["category_id"])
                if category_id in repeated_by_category:
                    repeated_by_category[category_id] += 1

    category_names = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in data.get("categories", [])
    }
    summary_rows = []
    for category_id in sorted(hard_categories):
        original = original_by_category.get(category_id, 0)
        repeated = repeated_by_category.get(category_id, 0)
        summary_rows.append(
            {
                "category_id": category_id,
                "category_name": category_names.get(category_id, str(category_id)),
                "original_train_gt_count": original,
                "oversampled_train_gt_count": repeated,
                "repeat_gain": 0.0 if original == 0 else repeated / original,
                "target_hard_boxes": target_hard_boxes,
                "max_repeat": max_repeat,
            }
        )
    summary_rows.append(
        {
            "category_id": -1,
            "category_name": "__repeat_histogram__",
            "original_train_gt_count": len(data.get("images", [])),
            "oversampled_train_gt_count": len(images_out),
            "repeat_gain": json.dumps(dict(sorted(repeat_histogram.items()))),
            "target_hard_boxes": target_hard_boxes,
            "max_repeat": max_repeat,
        }
    )
    return (
        {
            **{key: value for key, value in data.items() if key not in {"images", "annotations"}},
            "images": images_out,
            "annotations": annotations_out,
        },
        summary_rows,
    )


def collect_dataset_image_ids(dataset_dir: Path) -> set[int]:
    image_ids: set[int] = set()
    for split in ("train", "valid", "test"):
        annotation_path = dataset_dir / split / "_annotations.coco.json"
        if not annotation_path.exists():
            continue
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_ids.update(int(image["id"]) for image in data.get("images", []))
    return image_ids


def write_split(
    *,
    source_split_dir: Path,
    out_split_dir: Path,
    annotation_data: dict[str, Any],
    copy_images: bool,
) -> None:
    out_split_dir.mkdir(parents=True, exist_ok=True)
    (out_split_dir / "_annotations.coco.json").write_text(json.dumps(annotation_data), encoding="utf-8")
    linked_files = set()
    for image in annotation_data.get("images", []):
        file_name = str(image["file_name"])
        if file_name in linked_files:
            continue
        linked_files.add(file_name)
        source = source_split_dir / file_name
        destination = out_split_dir / file_name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_images:
            shutil.copy2(source, destination)
        else:
            os.symlink(source.resolve(), destination)


if __name__ == "__main__":
    main()
