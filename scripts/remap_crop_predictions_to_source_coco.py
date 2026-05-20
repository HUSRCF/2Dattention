"""Map crop-coordinate COCO detections back to source-image COCO coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-annotations", type=Path, required=True)
    parser.add_argument("--crop-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remapped = remap_crop_predictions(
        crop_annotations=json.loads(args.crop_annotations.read_text(encoding="utf-8")),
        crop_predictions=json.loads(args.crop_predictions.read_text(encoding="utf-8")),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(remapped, indent=2) + "\n", encoding="utf-8")
    print(f"saved_remapped_predictions: {args.out}")
    print(f"predictions: {len(remapped)}")


def remap_crop_predictions(
    crop_annotations: dict[str, Any],
    crop_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    crop_image_by_id = {int(image["id"]): image for image in crop_annotations.get("images", [])}
    remapped = []
    for prediction in crop_predictions:
        crop_image = crop_image_by_id.get(int(prediction["image_id"]))
        if crop_image is None or "source_image_id" not in crop_image or "crop_box" not in crop_image:
            continue
        crop_x1, crop_y1, _, _ = [float(value) for value in crop_image["crop_box"]]
        x, y, width, height = [float(value) for value in prediction["bbox"]]
        if width <= 0 or height <= 0:
            continue
        remapped.append(
            {
                "image_id": int(crop_image["source_image_id"]),
                "category_id": int(prediction["category_id"]),
                "bbox": [x + crop_x1, y + crop_y1, width, height],
                "score": float(prediction["score"]),
            }
        )
    return remapped


if __name__ == "__main__":
    main()
