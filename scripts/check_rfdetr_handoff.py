"""Check local RF-DETR package availability and prepared dataset layout."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


PACKAGES = ("rfdetr", "torch", "torchvision", "transformers", "timm", "supervision")
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
    report = {}
    for package in PACKAGES:
        spec = importlib.util.find_spec(package)
        report[package] = {
            "installed": spec is not None,
            "origin": spec.origin if spec is not None else "",
        }
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
    annotation_categories = {int(annotation["category_id"]) for annotation in data.get("annotations", [])}
    missing_category_ids = sorted(annotation_categories.difference(categories))
    sampled_images = data.get("images", [])[:20]
    missing_sampled_files = [
        image["file_name"] for image in sampled_images if not (split_dir / image["file_name"]).exists()
    ]
    return {
        "images": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
        "categories": len(categories),
        "min_category_id": min(categories) if categories else None,
        "max_category_id": max(categories) if categories else None,
        "missing_category_ids": missing_category_ids,
        "missing_sampled_files": missing_sampled_files,
        "ok": not missing_category_ids and not missing_sampled_files,
    }


if __name__ == "__main__":
    main()
