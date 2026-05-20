"""Create COCO crops centered on selected boxes for small-object diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.filter_coco_annotations import keep_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-image-dir", type=Path, required=True)
    parser.add_argument("--file-prefix", default="crop")
    parser.add_argument("--slice", choices=("all", "offcenter", "center", "small", "medium", "large"), default="small")
    parser.add_argument("--crop-scale", type=float, default=6.0)
    parser.add_argument("--min-crop-size", type=float, default=128.0)
    parser.add_argument("--min-visible-fraction", type=float, default=0.25)
    parser.add_argument("--max-crops", type=int, default=None)
    parser.add_argument("--max-crops-per-image", type=int, default=None)
    parser.add_argument("--center-radius", type=float, default=0.25)
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    parser.add_argument("--large-area-ratio", type=float, default=0.25)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cropped = crop_coco_around_boxes(
        annotations=args.annotations,
        image_root=args.image_root,
        out_image_dir=args.out_image_dir,
        slice_name=args.slice,
        crop_scale=args.crop_scale,
        min_crop_size=args.min_crop_size,
        min_visible_fraction=args.min_visible_fraction,
        max_crops=args.max_crops,
        max_crops_per_image=args.max_crops_per_image,
        file_prefix=args.file_prefix,
        center_radius=args.center_radius,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(cropped, ensure_ascii=False, indent=args.indent if args.indent > 0 else None),
        encoding="utf-8",
    )
    print(
        f"saved_cropped_coco: {args.out_json} "
        f"images={len(cropped['images'])} annotations={len(cropped['annotations'])}"
    )


def crop_coco_around_boxes(
    annotations: Path,
    image_root: Path,
    out_image_dir: Path,
    slice_name: str = "small",
    crop_scale: float = 6.0,
    min_crop_size: float = 128.0,
    min_visible_fraction: float = 0.25,
    max_crops: int | None = None,
    max_crops_per_image: int | None = None,
    file_prefix: str = "crop",
    center_radius: float = 0.25,
    small_area_ratio: float = 0.05,
    large_area_ratio: float = 0.25,
) -> dict[str, Any]:
    data = json.loads(annotations.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in data["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in data["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    out_image_dir.mkdir(parents=True, exist_ok=True)
    crop_images: list[dict[str, Any]] = []
    crop_annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    crops_by_source_image: dict[int, int] = {}
    for source_annotation in data["annotations"]:
        source_image = images_by_id[int(source_annotation["image_id"])]
        source_image_id = int(source_image["id"])
        if max_crops_per_image is not None and crops_by_source_image.get(source_image_id, 0) >= max_crops_per_image:
            continue
        if not keep_annotation(
            source_annotation,
            source_image,
            slice_name=slice_name,
            center_radius=center_radius,
            small_area_ratio=small_area_ratio,
            large_area_ratio=large_area_ratio,
        ):
            continue
        crop_box = compute_context_crop(
            source_annotation["bbox"],
            image_width=float(source_image["width"]),
            image_height=float(source_image["height"]),
            crop_scale=crop_scale,
            min_crop_size=min_crop_size,
        )
        source_path = image_root / source_image["file_name"]
        crop_name = (
            f"{file_prefix}_{next_image_id:012d}_src{int(source_image['id']):012d}_"
            f"ann{int(source_annotation['id']):012d}.jpg"
        )
        crop_path = out_image_dir / crop_name
        save_crop(source_path, crop_path, crop_box)
        crop_width = int(round(crop_box[2] - crop_box[0]))
        crop_height = int(round(crop_box[3] - crop_box[1]))
        crop_images.append(
            {
                "id": next_image_id,
                "file_name": crop_name,
                "width": crop_width,
                "height": crop_height,
                "source_image_id": source_image_id,
                "source_file_name": source_image["file_name"],
                "crop_box": list(crop_box),
            }
        )
        for annotation in annotations_by_image[source_image_id]:
            clipped = clip_bbox_to_crop(annotation["bbox"], crop_box)
            if clipped is None:
                continue
            visible_area = clipped[2] * clipped[3]
            original_area = max(1e-6, float(annotation["bbox"][2]) * float(annotation["bbox"][3]))
            if visible_area / original_area < min_visible_fraction:
                continue
            crop_annotations.append(
                {
                    **annotation,
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "bbox": clipped,
                    "area": visible_area,
                }
            )
            next_annotation_id += 1
        next_image_id += 1
        crops_by_source_image[source_image_id] = crops_by_source_image.get(source_image_id, 0) + 1
        if max_crops is not None and len(crop_images) >= max_crops:
            break
    return {
        **{key: value for key, value in data.items() if key not in {"images", "annotations"}},
        "images": crop_images,
        "annotations": crop_annotations,
    }


def compute_context_crop(
    bbox: list[float],
    image_width: float,
    image_height: float,
    crop_scale: float,
    min_crop_size: float,
) -> tuple[float, float, float, float]:
    x, y, width, height = [float(value) for value in bbox]
    side = min(max(min_crop_size, max(width, height) * crop_scale), image_width, image_height)
    cx = x + width / 2.0
    cy = y + height / 2.0
    x1 = min(max(0.0, cx - side / 2.0), image_width - side)
    y1 = min(max(0.0, cy - side / 2.0), image_height - side)
    return x1, y1, x1 + side, y1 + side


def save_crop(source_path: Path, crop_path: Path, crop_box: tuple[float, float, float, float]) -> None:
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        crop = rgb.crop(tuple(round(value) for value in crop_box))
        crop.save(crop_path, quality=95)


def clip_bbox_to_crop(bbox: list[float], crop_box: tuple[float, float, float, float]) -> list[float] | None:
    x, y, width, height = [float(value) for value in bbox]
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    x1 = max(x, crop_x1)
    y1 = max(y, crop_y1)
    x2 = min(x + width, crop_x2)
    y2 = min(y + height, crop_y2)
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    return [x1 - crop_x1, y1 - crop_y1, clipped_width, clipped_height]


if __name__ == "__main__":
    main()
