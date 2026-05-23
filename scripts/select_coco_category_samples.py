"""Select COCO annotations for a category-level diagnostic target list."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = (
    "image_id",
    "file_name",
    "category_id",
    "category_name",
    "transition",
    "annotation_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "area",
    "area_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--category-report",
        type=Path,
        required=True,
        help="CSV with category_id/category_name and optional transition columns.",
    )
    parser.add_argument(
        "--transition",
        action="append",
        default=None,
        help="Optional transition labels to keep from the category report.",
    )
    parser.add_argument("--max-per-category", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = select_category_samples(
        annotations=json.loads(args.annotations.read_text(encoding="utf-8")),
        category_report=read_rows(args.category_report),
        transitions=set(args.transition) if args.transition else None,
        max_per_category=args.max_per_category,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_coco_category_samples: {args.out}")
    print(f"rows={len(rows)}")


def select_category_samples(
    *,
    annotations: dict[str, Any],
    category_report: list[dict[str, str]],
    transitions: set[str] | None,
    max_per_category: int,
) -> list[dict[str, str]]:
    target_categories = target_category_rows(category_report, transitions=transitions)
    image_by_id = {int(image["id"]): image for image in annotations.get("images", [])}
    selected_counts: dict[int, int] = {}
    out = []
    for annotation in annotations.get("annotations", []):
        category_id = int(annotation["category_id"])
        if category_id not in target_categories:
            continue
        if max_per_category > 0 and selected_counts.get(category_id, 0) >= max_per_category:
            continue
        image = image_by_id[int(annotation["image_id"])]
        bbox = [float(value) for value in annotation["bbox"]]
        image_area = max(1.0, float(image["width"]) * float(image["height"]))
        area = max(0.0, bbox[2]) * max(0.0, bbox[3])
        target = target_categories[category_id]
        out.append(
            {
                "image_id": str(annotation["image_id"]),
                "file_name": str(image["file_name"]),
                "category_id": str(category_id),
                "category_name": target.get("category_name", str(category_id)),
                "transition": target.get("transition", ""),
                "annotation_id": str(annotation.get("id", "")),
                "bbox_x": format_float(bbox[0]),
                "bbox_y": format_float(bbox[1]),
                "bbox_w": format_float(bbox[2]),
                "bbox_h": format_float(bbox[3]),
                "area": format_float(area),
                "area_ratio": format_float(area / image_area),
            }
        )
        selected_counts[category_id] = selected_counts.get(category_id, 0) + 1
    out.sort(
        key=lambda row: (
            row["transition"],
            parse_int(row["category_id"]),
            parse_int(row["image_id"]),
            parse_float(row["area"]),
        )
    )
    return out


def target_category_rows(
    rows: list[dict[str, str]],
    *,
    transitions: set[str] | None,
) -> dict[int, dict[str, str]]:
    targets = {}
    for row in rows:
        if transitions is not None and row.get("transition", "") not in transitions:
            continue
        targets[parse_int(row["category_id"])] = row
    return targets


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_int(value: str) -> int:
    return int(float(value or "0"))


def parse_float(value: str) -> float:
    return float(value or "0")


def format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    main()
