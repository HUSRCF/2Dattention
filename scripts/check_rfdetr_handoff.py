"""Check local RF-DETR package availability and prepared dataset layout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


BBOX_EPS = 1e-3
PACKAGES = (
    "rfdetr",
    "torch",
    "torchvision",
    "transformers",
    "timm",
    "supervision",
    "albumentations",
    "faster_coco_eval",
    "pytorch_lightning",
    "torchmetrics",
)
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "packages": package_report(),
        "dataset": dataset_report(args.dataset_dir),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"saved_rfdetr_handoff_check: {args.out}")


def package_report() -> dict[str, dict[str, Any]]:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    report = {}
    for package in PACKAGES:
        spec = importlib.util.find_spec(package)
        package_entry: dict[str, Any] = {
            "installed": spec is not None,
            "origin": spec.origin if spec is not None else "",
        }
        if spec is not None:
            try:
                importlib.import_module(package)
                package_entry["import_ok"] = True
                package_entry["import_error"] = ""
            except Exception as exc:  # noqa: BLE001 - check script should report import smoke failures.
                package_entry["import_ok"] = False
                package_entry["import_error"] = f"{type(exc).__name__}: {exc}"
        else:
            package_entry["import_ok"] = False
            package_entry["import_error"] = "not found"
        report[package] = package_entry
    return report


def dataset_report(dataset_dir: Path) -> dict[str, Any]:
    splits = {}
    split_image_ids = {}
    split_file_names = {}
    for split in SPLITS:
        annotation_path = dataset_dir / split / "_annotations.coco.json"
        split_report: dict[str, Any] = {"annotation_path": str(annotation_path), "exists": annotation_path.exists()}
        if annotation_path.exists():
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            split_image_ids[split] = {int(image["id"]) for image in data.get("images", [])}
            split_file_names[split] = {str(image["file_name"]) for image in data.get("images", [])}
            split_report.update(coco_split_report(data, dataset_dir / split))
            split_report["annotation_sha256"] = file_sha256(annotation_path)
        splits[split] = split_report
    return {
        "dataset_dir": str(dataset_dir),
        "splits": splits,
        "split_consistency": split_consistency_report(splits, split_image_ids, split_file_names),
    }


def coco_split_report(data: dict[str, Any], split_dir: Path) -> dict[str, Any]:
    categories = {int(category["id"]) for category in data.get("categories", [])}
    category_rows = list(data.get("categories", []))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_ids = sorted(int(image["id"]) for image in images)
    image_by_id = {int(image["id"]): image for image in images}
    annotation_categories = {int(annotation["category_id"]) for annotation in data.get("annotations", [])}
    missing_category_ids = sorted(annotation_categories.difference(categories))
    missing_files = [image["file_name"] for image in images if not (split_dir / image["file_name"]).exists()]
    orphan_annotation_ids = [
        annotation.get("id") for annotation in annotations if int(annotation.get("image_id", -1)) not in image_by_id
    ]
    invalid_bbox_ids = []
    out_of_bounds_bbox_ids = []
    for annotation in annotations:
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            invalid_bbox_ids.append(annotation.get("id"))
            continue
        x, y, width, height = [float(value) for value in bbox]
        if width <= 0 or height <= 0:
            invalid_bbox_ids.append(annotation.get("id"))
            continue
        image = image_by_id.get(int(annotation.get("image_id", -1)))
        if image is None:
            continue
        image_width = float(image.get("width", 0))
        image_height = float(image.get("height", 0))
        if image_width > 0 and image_height > 0:
            if (
                x < -BBOX_EPS
                or y < -BBOX_EPS
                or x + width > image_width + BBOX_EPS
                or y + height > image_height + BBOX_EPS
            ):
                out_of_bounds_bbox_ids.append(annotation.get("id"))
    return {
        "images": len(images),
        "image_ids_sha256": integer_list_sha256(image_ids),
        "annotations": len(annotations),
        "categories": len(categories),
        "min_category_id": min(categories) if categories else None,
        "max_category_id": max(categories) if categories else None,
        "category_ids": sorted(categories),
        "category_table_sha256": category_table_sha256(category_rows),
        "missing_category_ids": missing_category_ids,
        "checked_files": len(images),
        "missing_files_count": len(missing_files),
        "missing_files": missing_files[:20],
        "orphan_annotation_ids_count": len(orphan_annotation_ids),
        "orphan_annotation_ids": orphan_annotation_ids[:20],
        "invalid_bbox_ids_count": len(invalid_bbox_ids),
        "invalid_bbox_ids": invalid_bbox_ids[:20],
        "out_of_bounds_bbox_ids_count": len(out_of_bounds_bbox_ids),
        "out_of_bounds_bbox_ids": out_of_bounds_bbox_ids[:20],
        "ok": not (
            missing_category_ids
            or missing_files
            or orphan_annotation_ids
            or invalid_bbox_ids
            or out_of_bounds_bbox_ids
        ),
    }


def split_consistency_report(
    splits: dict[str, dict[str, Any]],
    split_image_ids: dict[str, set[int]],
    split_file_names: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    category_ranges = {
        split: (report.get("min_category_id"), report.get("max_category_id"), report.get("categories"))
        for split, report in splits.items()
        if report.get("exists")
    }
    hashes = {
        split: report.get("annotation_sha256")
        for split, report in splits.items()
        if report.get("exists") and report.get("annotation_sha256")
    }
    train_range = category_ranges.get("train")
    train_category_ids = splits.get("train", {}).get("category_ids")
    train_category_table_sha = splits.get("train", {}).get("category_table_sha256")
    category_range_matches_train = {
        split: (category_range == train_range)
        for split, category_range in category_ranges.items()
        if split != "train"
    }
    category_ids_match_train = {
        split: (report.get("category_ids") == train_category_ids)
        for split, report in splits.items()
        if split != "train" and report.get("exists")
    }
    category_table_matches_train = {
        split: (report.get("category_table_sha256") == train_category_table_sha)
        for split, report in splits.items()
        if split != "train" and report.get("exists")
    }
    image_id_overlap_counts = {}
    file_name_overlap_counts = {}
    split_file_names = split_file_names or {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if left in split_image_ids and right in split_image_ids:
                image_id_overlap_counts[f"{left}_{right}"] = len(
                    split_image_ids[left].intersection(split_image_ids[right])
                )
            if left in split_file_names and right in split_file_names:
                file_name_overlap_counts[f"{left}_{right}"] = len(
                    split_file_names[left].intersection(split_file_names[right])
                )
    return {
        "category_ranges": {
            split: {
                "min_category_id": values[0],
                "max_category_id": values[1],
                "categories": values[2],
            }
            for split, values in category_ranges.items()
        },
        "category_range_matches_train": category_range_matches_train,
        "category_ids_match_train": category_ids_match_train,
        "category_table_matches_train": category_table_matches_train,
        "valid_test_annotations_identical": hashes.get("valid") == hashes.get("test")
        if "valid" in hashes and "test" in hashes
        else None,
        "image_id_overlap_counts": image_id_overlap_counts,
        "image_ids_disjoint": all(count == 0 for count in image_id_overlap_counts.values())
        if image_id_overlap_counts
        else None,
        "file_name_overlap_counts": file_name_overlap_counts,
        "file_names_disjoint": all(count == 0 for count in file_name_overlap_counts.values())
        if file_name_overlap_counts
        else None,
    }


def category_table_sha256(categories: list[dict[str, Any]]) -> str:
    normalized = sorted(categories, key=lambda category: int(category["id"]))
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_list_sha256(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
