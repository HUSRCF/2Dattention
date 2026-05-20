"""Prepare an RF-DETR COCO directory from existing COCO JSON annotations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


SPLIT_FILE = "_annotations.coco.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--valid-json", type=Path, required=True)
    parser.add_argument(
        "--test-json",
        type=Path,
        default=None,
        help="Optional test annotations. If omitted, valid is mirrored into test for RF-DETR API compatibility.",
    )
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("symlink", "copy", "hardlink"), default="symlink")
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = {
        "train": args.train_json,
        "valid": args.valid_json,
        "test": args.test_json or args.valid_json,
    }
    for split_name, annotation_json in splits.items():
        prepared = prepare_rfdetr_split(
            annotation_json=annotation_json,
            image_root=args.image_root,
            split_dir=args.out_dir / split_name,
            link_mode=args.link_mode,
            indent=args.indent,
        )
        print(
            f"wrote {prepared} images={prepared_image_count(prepared)} "
            f"annotations={prepared_annotation_count(prepared)}"
        )
    if args.test_json is None:
        print("note: --test-json omitted; valid split was mirrored into test")


def prepare_rfdetr_split(
    annotation_json: Path,
    image_root: Path,
    split_dir: Path,
    link_mode: str = "symlink",
    indent: int = 2,
) -> Path:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    validate_coco(data)
    split_dir.mkdir(parents=True, exist_ok=True)
    image_name_by_id: dict[int, str] = {}
    for image in data["images"]:
        image_id = int(image["id"])
        source_name = str(image["file_name"])
        source_path = image_root / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"missing image for COCO file_name={source_name}: {source_path}")
        target_name = stable_image_name(image_id, source_name)
        materialize_image(source_path, split_dir / target_name, link_mode=link_mode)
        image_name_by_id[image_id] = target_name
        image["file_name"] = target_name
    annotation_path = split_dir / SPLIT_FILE
    annotation_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent if indent > 0 else None),
        encoding="utf-8",
    )
    return annotation_path


def validate_coco(data: dict[str, Any]) -> None:
    missing = {"images", "annotations", "categories"}.difference(data)
    if missing:
        raise KeyError(f"COCO annotation JSON missing required keys: {sorted(missing)}")


def stable_image_name(image_id: int, source_name: str) -> str:
    suffix = Path(source_name).suffix or ".jpg"
    stem = Path(source_name).stem
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return f"{image_id:012d}_{safe_stem}{suffix}"


def materialize_image(source: Path, target: Path, link_mode: str) -> None:
    if target.exists() or target.is_symlink():
        return
    if link_mode == "symlink":
        target.symlink_to(source.resolve())
    elif link_mode == "hardlink":
        target.hardlink_to(source)
    elif link_mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"unsupported link_mode: {link_mode}")


def prepared_image_count(annotation_path: Path) -> int:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    return len(data["images"])


def prepared_annotation_count(annotation_path: Path) -> int:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    return len(data["annotations"])


if __name__ == "__main__":
    main()
