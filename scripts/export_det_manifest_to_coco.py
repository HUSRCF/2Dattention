"""Export the ILSVRC DET manifest into COCO-style detection JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/ILSVRC2013_DET_val_supervised/det_val_manifest.csv"),
        help="CSV with image_id,image_path,width,height,label,xmin,ymin,xmax,ymax columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/ILSVRC2013_DET_val_supervised/coco"),
        help="Directory for COCO JSON output.",
    )
    parser.add_argument("--top-classes", type=int, default=0, help="Keep the most frequent labels; <=0 keeps all.")
    parser.add_argument("--max-images", type=int, default=0, help="Keep at most this many images after label filtering.")
    parser.add_argument("--split-csv", type=Path, default=None, help="Optional split CSV from train_det_real.py.")
    parser.add_argument(
        "--split-run-seed",
        type=int,
        default=None,
        help="When split CSV has run_seed, export only this seed; default uses the first seed in the file.",
    )
    parser.add_argument(
        "--file-name-mode",
        choices=("manifest", "basename", "relative"),
        default="relative",
        help="COCO image file_name source. 'relative' is relative to --image-root.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_val"),
        help="Root used by --file-name-mode relative.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation; use <=0 for compact output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    split_map = read_split_map(args.split_csv, args.split_run_seed) if args.split_csv is not None else None
    rows = filter_rows(rows, top_classes=args.top_classes, max_images=0 if split_map is not None else args.max_images)
    if split_map is not None:
        rows = limit_rows_by_images_per_split(rows, max_images=args.max_images, split_map=split_map)
    datasets = build_coco_datasets(
        rows,
        split_map=split_map,
        file_name_mode=args.file_name_mode,
        image_root=args.image_root,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    indent = args.indent if args.indent > 0 else None
    for split_name, dataset in datasets.items():
        path = args.out_dir / f"{split_name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(dataset, handle, ensure_ascii=False, indent=indent)
        print(
            f"wrote {path} images={len(dataset['images'])} "
            f"annotations={len(dataset['annotations'])} categories={len(dataset['categories'])}"
        )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "image_path", "width", "height", "label", "xmin", "ymin", "xmax", "ymax"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest missing required columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def filter_rows(rows: list[dict[str, str]], top_classes: int, max_images: int) -> list[dict[str, str]]:
    if top_classes > 0:
        counts = Counter(row["label"] for row in rows)
        selected = {label for label, _ in counts.most_common(top_classes)}
        rows = [row for row in rows if row["label"] in selected]
    if max_images > 0:
        kept_images: set[str] = set()
        filtered: list[dict[str, str]] = []
        for row in rows:
            image_id = row["image_id"]
            if image_id not in kept_images and len(kept_images) >= max_images:
                continue
            kept_images.add(image_id)
            filtered.append(row)
        rows = filtered
    return rows


def limit_rows_by_images_per_split(
    rows: list[dict[str, str]],
    max_images: int,
    split_map: dict[str, str],
) -> list[dict[str, str]]:
    if max_images <= 0:
        return rows
    kept_images_by_split: dict[str, set[str]] = defaultdict(set)
    filtered: list[dict[str, str]] = []
    for row in rows:
        image_id = row["image_id"]
        split = split_map.get(image_id)
        if split is None:
            continue
        kept_images = kept_images_by_split[split]
        if image_id not in kept_images and len(kept_images) >= max_images:
            continue
        kept_images.add(image_id)
        filtered.append(row)
    return filtered


def read_split_map(path: Path, run_seed: int | None) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty split CSV: {path}")
    if "image_id" not in rows[0] or "split" not in rows[0]:
        raise ValueError("split CSV must include image_id and split columns")
    if "run_seed" in rows[0]:
        seeds = sorted({int(row["run_seed"]) for row in rows})
        selected_seed = seeds[0] if run_seed is None else run_seed
        if selected_seed not in seeds:
            raise ValueError(f"requested split-run-seed {selected_seed} not present; available={seeds}")
        rows = [row for row in rows if int(row["run_seed"]) == selected_seed]
    split_map: dict[str, str] = {}
    for row in rows:
        split_map[row["image_id"]] = row["split"]
    return split_map


def build_coco_datasets(
    rows: list[dict[str, str]],
    split_map: dict[str, str] | None,
    file_name_mode: str,
    image_root: Path,
) -> dict[str, dict[str, Any]]:
    labels = sorted({row["label"] for row in rows})
    category_id = {label: idx + 1 for idx, label in enumerate(labels)}
    image_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if split_map is not None and row["image_id"] not in split_map:
            continue
        image_rows[row["image_id"]].append(row)

    split_names = ["annotations"] if split_map is None else sorted(set(split_map.values()))
    datasets = {
        split_name: {
            "images": [],
            "annotations": [],
            "categories": [{"id": category_id[label], "name": label, "supercategory": "object"} for label in labels],
        }
        for split_name in split_names
    }
    next_image_id = 1
    next_annotation_id = 1
    for image_id_key in sorted(image_rows):
        first = image_rows[image_id_key][0]
        split_name = split_map[image_id_key] if split_map is not None else "annotations"
        if split_name not in datasets:
            continue
        numeric_image_id = next_image_id
        next_image_id += 1
        image_entry = {
            "id": numeric_image_id,
            "file_name": coco_file_name(first["image_path"], mode=file_name_mode, image_root=image_root),
            "width": int(first["width"]),
            "height": int(first["height"]),
        }
        datasets[split_name]["images"].append(image_entry)
        for row in image_rows[image_id_key]:
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])
            width = max(0.0, xmax - xmin)
            height = max(0.0, ymax - ymin)
            datasets[split_name]["annotations"].append(
                {
                    "id": next_annotation_id,
                    "image_id": numeric_image_id,
                    "category_id": category_id[row["label"]],
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            next_annotation_id += 1
    return datasets


def coco_file_name(image_path: str, mode: str, image_root: Path) -> str:
    path = Path(image_path)
    if mode == "manifest":
        return path.as_posix()
    if mode == "basename":
        return path.name
    try:
        return path.relative_to(image_root).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    sys.exit(main())
