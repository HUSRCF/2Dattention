"""Filter COCO annotations into project-specific evaluation slices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slice", choices=("all", "offcenter", "center", "small", "medium", "large"), required=True)
    parser.add_argument("--center-radius", type=float, default=0.25)
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    parser.add_argument("--large-area-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filtered = filter_coco_annotations(
        args.annotations,
        slice_name=args.slice,
        center_radius=args.center_radius,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(filtered), encoding="utf-8")
    print(
        f"saved_slice: {args.out} "
        f"images={len(filtered['images'])} annotations={len(filtered['annotations'])}"
    )


def filter_coco_annotations(
    annotation_json: Path,
    slice_name: str,
    center_radius: float = 0.25,
    small_area_ratio: float = 0.05,
    large_area_ratio: float = 0.25,
) -> dict[str, Any]:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in data["images"]}
    kept_annotations = [
        annotation
        for annotation in data["annotations"]
        if keep_annotation(
            annotation,
            images_by_id[int(annotation["image_id"])],
            slice_name=slice_name,
            center_radius=center_radius,
            small_area_ratio=small_area_ratio,
            large_area_ratio=large_area_ratio,
        )
    ]
    kept_image_ids = {int(annotation["image_id"]) for annotation in kept_annotations}
    kept_images = data["images"] if slice_name == "all" else [
        image for image in data["images"] if int(image["id"]) in kept_image_ids
    ]
    return {
        **{key: value for key, value in data.items() if key not in {"images", "annotations"}},
        "images": kept_images,
        "annotations": kept_annotations,
    }


def keep_annotation(
    annotation: dict[str, Any],
    image: dict[str, Any],
    slice_name: str,
    center_radius: float,
    small_area_ratio: float,
    large_area_ratio: float,
) -> bool:
    if slice_name == "all":
        return True
    area_ratio = bbox_area_ratio(annotation, image)
    if slice_name == "small":
        return area_ratio < small_area_ratio
    if slice_name == "medium":
        return small_area_ratio <= area_ratio < large_area_ratio
    if slice_name == "large":
        return area_ratio >= large_area_ratio
    distance = bbox_center_distance(annotation, image)
    if slice_name == "offcenter":
        return distance >= center_radius
    if slice_name == "center":
        return distance < center_radius
    raise ValueError(f"unsupported slice: {slice_name}")


def bbox_area_ratio(annotation: dict[str, Any], image: dict[str, Any]) -> float:
    _, _, width, height = [float(value) for value in annotation["bbox"]]
    image_area = max(1.0, float(image["width"]) * float(image["height"]))
    return max(0.0, width) * max(0.0, height) / image_area


def bbox_center_distance(annotation: dict[str, Any], image: dict[str, Any]) -> float:
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    image_width = max(1.0, float(image["width"]))
    image_height = max(1.0, float(image["height"]))
    cx = (x + width / 2.0) / image_width
    cy = (y + height / 2.0) / image_height
    return math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)


if __name__ == "__main__":
    main()
