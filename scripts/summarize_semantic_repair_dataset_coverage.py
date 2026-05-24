"""Summarize semantic-repair target support in an RF-DETR COCO dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--semantic-repair-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_repair_coverage(
        dataset_dir=args.dataset_dir,
        repair_config=json.loads(args.semantic_repair_config.read_text(encoding="utf-8")),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.out, rows)
    print(f"saved_semantic_repair_dataset_coverage: {args.out}")


def summarize_repair_coverage(
    *,
    dataset_dir: Path,
    repair_config: dict[str, Any],
) -> list[dict[str, str]]:
    split_stats = {split: load_split_category_stats(dataset_dir / split / "_annotations.coco.json") for split in SPLITS}
    category_names = set().union(*(stats["category_names"] for stats in split_stats.values()))
    rows: list[dict[str, str]] = []
    for target in repair_config.get("targets", []):
        positive_name = str(target["category_name"])
        positive_id = int(target["category_id"])
        rows.append(
            build_row(
                split_stats=split_stats,
                category_names=category_names,
                role="positive",
                positive_category_id=positive_id,
                positive_category_name=positive_name,
                category_id=positive_id,
                category_name=positive_name,
                negative_category_id="",
                negative_category_name="",
                configured_samples="",
                configured_weight=str(target.get("total_weight", "")),
            )
        )
        for negative in target.get("hard_negatives", []):
            negative_name = str(negative["negative_category_name"])
            negative_id = int(negative["negative_category_id"])
            rows.append(
                build_row(
                    split_stats=split_stats,
                    category_names=category_names,
                    role="negative",
                    positive_category_id=positive_id,
                    positive_category_name=positive_name,
                    category_id=negative_id,
                    category_name=negative_name,
                    negative_category_id=negative_id,
                    negative_category_name=negative_name,
                    configured_samples=str(negative.get("samples", "")),
                    configured_weight=str(negative.get("weight", "")),
                )
            )
    return rows


def load_split_category_stats(annotation_path: Path) -> dict[str, Any]:
    if not annotation_path.exists():
        raise FileNotFoundError(f"missing split annotation file: {annotation_path}")
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    id_to_name = {int(category["id"]): str(category["name"]) for category in data.get("categories", [])}
    box_counts: dict[str, int] = defaultdict(int)
    image_ids_by_name: dict[str, set[int]] = defaultdict(set)
    for annotation in data.get("annotations", []):
        category_id = int(annotation["category_id"])
        category_name = id_to_name.get(category_id)
        if category_name is None:
            raise ValueError(f"annotation uses unknown category_id={category_id} in {annotation_path}")
        box_counts[category_name] += 1
        image_ids_by_name[category_name].add(int(annotation["image_id"]))
    return {
        "category_names": set(id_to_name.values()),
        "box_counts": dict(box_counts),
        "image_counts": {name: len(image_ids) for name, image_ids in image_ids_by_name.items()},
    }


def build_row(
    *,
    split_stats: dict[str, dict[str, Any]],
    category_names: set[str],
    role: str,
    positive_category_id: int,
    positive_category_name: str,
    category_id: int,
    category_name: str,
    negative_category_id: int | str,
    negative_category_name: str,
    configured_samples: str,
    configured_weight: str,
) -> dict[str, str]:
    row = {
        "role": role,
        "positive_category_id": str(positive_category_id),
        "positive_category_name": positive_category_name,
        "category_id": str(category_id),
        "category_name": category_name,
        "negative_category_id": str(negative_category_id),
        "negative_category_name": negative_category_name,
        "configured_samples": configured_samples,
        "configured_weight": configured_weight,
        "present_in_dataset_categories": str(category_name in category_names).lower(),
    }
    for split in SPLITS:
        stats = split_stats[split]
        row[f"{split}_images"] = str(stats["image_counts"].get(category_name, 0))
        row[f"{split}_boxes"] = str(stats["box_counts"].get(category_name, 0))
    return row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "role",
        "positive_category_id",
        "positive_category_name",
        "category_id",
        "category_name",
        "negative_category_id",
        "negative_category_name",
        "configured_samples",
        "configured_weight",
        "present_in_dataset_categories",
        "train_images",
        "train_boxes",
        "valid_images",
        "valid_boxes",
        "test_images",
        "test_boxes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
