"""Export RF-DETR predictions for a COCO annotation file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_rfdetr_coco import build_rfdetr_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-size", choices=("nano", "small", "medium", "large", "base"), default="nano")
    parser.add_argument("--weights", type=Path, default=None, help="Optional RF-DETR checkpoint/pretrain weights.")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotation_json.read_text(encoding="utf-8"))
    category_ids = sorted(int(category["id"]) for category in coco.get("categories", []))
    if not category_ids:
        raise ValueError(f"annotation file has no categories: {args.annotation_json}")
    model = build_rfdetr_model(
        args.model_size,
        pretrain_weights=args.weights,
        num_classes=args.num_classes,
        device=None if args.device == "auto" else args.device,
    )
    model_num_classes = infer_model_num_classes(model, default=len(category_ids))
    if model_num_classes > len(category_ids):
        raise ValueError(
            f"model reports {model_num_classes} foreground classes, but annotation only has "
            f"{len(category_ids)} categories"
        )
    image_records = coco.get("images", [])
    if args.max_images is not None:
        image_records = image_records[: args.max_images]
    shape = (args.resolution, args.resolution) if args.resolution is not None else None
    predictions: list[dict[str, float | int]] = []
    for image in image_records:
        image_path = args.image_root / image["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"missing image file for annotation image_id={image['id']}: {image_path}")
        detections = model.predict(load_rgb_image(image_path), threshold=args.threshold, shape=shape)
        predictions.extend(
            detections_to_coco_records(
                detections,
                int(image["id"]),
                category_ids,
                model_num_classes=model_num_classes,
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"saved_rfdetr_predictions: {args.out}")
    print(f"images: {len(image_records)}")
    print(f"predictions: {len(predictions)}")
    print(f"model_num_classes: {model_num_classes}")


def infer_model_num_classes(model: Any, default: int) -> int:
    model_context = getattr(model, "model", None)
    model_args = getattr(model_context, "args", None)
    return int(getattr(model_args, "num_classes", default))


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as pil_image:
        return pil_image.convert("RGB")


def detections_to_coco_records(
    detections: Any,
    image_id: int,
    category_ids: list[int],
    model_num_classes: int | None = None,
) -> list[dict[str, float | int]]:
    boxes = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))), dtype=float)
    scores = np.asarray(getattr(detections, "confidence", np.empty((0,))), dtype=float)
    class_ids = np.asarray(getattr(detections, "class_id", np.empty((0,), dtype=int)), dtype=int)
    if model_num_classes is None:
        model_num_classes = len(category_ids)
    records: list[dict[str, float | int]] = []
    for box, score, class_id in zip(boxes, scores, class_ids, strict=False):
        if class_id < 0 or class_id >= model_num_classes or class_id >= len(category_ids):
            continue
        x, y, width, height = xyxy_to_xywh(box)
        if width <= 0 or height <= 0:
            continue
        records.append(
            {
                "image_id": image_id,
                "category_id": category_ids[int(class_id)],
                "bbox": [x, y, width, height],
                "score": float(score),
            }
        )
    return records


def xyxy_to_xywh(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, x2 - x1, y2 - y1]


if __name__ == "__main__":
    main()
