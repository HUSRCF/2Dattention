"""Check local RF-DETR package availability and prepared dataset layout."""

from __future__ import annotations

import argparse
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
    for split in SPLITS:
        annotation_path = dataset_dir / split / "_annotations.coco.json"
        split_report: dict[str, Any] = {"annotation_path": str(annotation_path), "exists": annotation_path.exists()}
        if annotation_path.exists():
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            split_report.update(coco_split_report(data, dataset_dir / split))
        splits[split] = split_report
    return {"dataset_dir": str(dataset_dir), "splits": splits}


def coco_split_report(data: dict[str, Any], split_dir: Path) -> dict[str, Any]:
    categories = {int(category["id"]) for category in data.get("categories", [])}
    images = data.get("images", [])
    annotations = data.get("annotations", [])
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
        "annotations": len(annotations),
        "categories": len(categories),
        "min_category_id": min(categories) if categories else None,
        "max_category_id": max(categories) if categories else None,
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


if __name__ == "__main__":
    main()
