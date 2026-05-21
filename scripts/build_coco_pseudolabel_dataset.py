"""Build an RF-DETR-compatible COCO pseudo-label dataset from teacher predictions."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, default=None)
    parser.add_argument("--valid-predictions", type=Path, default=None)
    parser.add_argument("--test-predictions", type=Path, default=None)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--topk-per-image", type=int, default=20)
    parser.add_argument("--min-area", type=float, default=4.0)
    parser.add_argument(
        "--include-gt-on-pseudo-splits",
        action="store_true",
        help="Append original GT annotations to splits that also receive pseudo labels.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize image files in the output split directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = {
        "train": args.train_predictions,
        "valid": args.valid_predictions,
        "test": args.test_predictions,
    }
    summaries = []
    for split in SPLITS:
        source_split = args.dataset_dir / split
        out_split = args.out_dir / split
        source_annotations = source_split / "_annotations.coco.json"
        if not source_annotations.exists():
            raise FileNotFoundError(f"missing source annotations: {source_annotations}")
        coco = json.loads(source_annotations.read_text(encoding="utf-8"))
        copy_or_link_images(coco, source_split, out_split, link_mode=args.link_mode)
        prediction_path = prediction_paths[split]
        if prediction_path is None:
            out_coco = coco
            source = "gt"
        else:
            predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
            out_coco, drop_stats = build_pseudo_coco(
                coco,
                predictions,
                min_score=args.min_score,
                topk_per_image=args.topk_per_image,
                min_area=args.min_area,
                include_gt=args.include_gt_on_pseudo_splits,
            )
            source = str(prediction_path)
        out_split.mkdir(parents=True, exist_ok=True)
        out_annotations = out_split / "_annotations.coco.json"
        out_annotations.write_text(json.dumps(out_coco, indent=2) + "\n", encoding="utf-8")
        summaries.append(
            {
                "split": split,
                "source": source,
                "images": len(out_coco.get("images", [])),
                "annotations": len(out_coco.get("annotations", [])),
                "drop_stats": drop_stats if prediction_path is not None else {},
            }
        )
    print("saved_pseudolabel_dataset:", args.out_dir)
    for row in summaries:
        print(
            f"{row['split']}: images={row['images']} annotations={row['annotations']} "
            f"source={row['source']}"
        )
        if row["drop_stats"]:
            print("  drop_stats:", json.dumps(row["drop_stats"], sort_keys=True))


def copy_or_link_images(
    coco: dict[str, Any],
    source_split: Path,
    out_split: Path,
    *,
    link_mode: str,
) -> None:
    out_split.mkdir(parents=True, exist_ok=True)
    for image in coco.get("images", []):
        file_name = str(image["file_name"])
        source_path = source_split / file_name
        out_path = out_split / file_name
        if not source_path.exists():
            raise FileNotFoundError(f"missing source image: {source_path}")
        if out_path.exists() or out_path.is_symlink():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if link_mode == "copy":
            shutil.copy2(source_path, out_path)
        else:
            out_path.symlink_to(source_path.resolve())


def build_pseudo_coco(
    coco: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    min_score: float,
    topk_per_image: int,
    min_area: float,
    include_gt: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    images = coco.get("images", [])
    categories = coco.get("categories", [])
    image_by_id = {int(image["id"]): image for image in images}
    valid_category_ids = {int(category["id"]) for category in categories}
    grouped_predictions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    drop_stats = {
        "raw_predictions": len(predictions),
        "dropped_image_id": 0,
        "dropped_category_id": 0,
        "dropped_score": 0,
        "dropped_bbox": 0,
        "dropped_area": 0,
        "kept_before_topk": 0,
        "dropped_topk": 0,
    }
    for prediction in predictions:
        image_id = int(prediction.get("image_id", -1))
        if image_id not in image_by_id:
            drop_stats["dropped_image_id"] += 1
            continue
        category_id = int(prediction.get("category_id", -1))
        if category_id not in valid_category_ids:
            drop_stats["dropped_category_id"] += 1
            continue
        score = float(prediction.get("score", 0.0))
        if score < min_score:
            drop_stats["dropped_score"] += 1
            continue
        bbox = clip_xywh_bbox(prediction.get("bbox", []), image_by_id[image_id])
        if bbox is None:
            drop_stats["dropped_bbox"] += 1
            continue
        _, _, width, height = bbox
        if width * height < min_area:
            drop_stats["dropped_area"] += 1
            continue
        grouped_predictions[image_id].append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
            }
        )
        drop_stats["kept_before_topk"] += 1
    annotations: list[dict[str, Any]] = []
    next_annotation_id = 1
    if include_gt:
        for annotation in coco.get("annotations", []):
            copied = dict(annotation)
            copied["id"] = next_annotation_id
            next_annotation_id += 1
            annotations.append(copied)
    for image_id in sorted(image_by_id):
        ranked = sorted(grouped_predictions.get(image_id, []), key=lambda item: item["score"], reverse=True)
        drop_stats["dropped_topk"] += max(0, len(ranked) - topk_per_image)
        for prediction in ranked[:topk_per_image]:
            x, y, width, height = prediction["bbox"]
            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": prediction["category_id"],
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "teacher_score": prediction["score"],
                }
            )
            next_annotation_id += 1
    out_coco = {
        **{key: value for key, value in coco.items() if key not in {"annotations"}},
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    return out_coco, drop_stats


def clip_xywh_bbox(raw_bbox: Any, image: dict[str, Any]) -> list[float] | None:
    if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
        return None
    x, y, width, height = [float(value) for value in raw_bbox]
    if width <= 0 or height <= 0:
        return None
    image_width = float(image.get("width", 0))
    image_height = float(image.get("height", 0))
    if image_width <= 0 or image_height <= 0:
        return None
    x1 = max(0.0, min(image_width, x))
    y1 = max(0.0, min(image_height, y))
    x2 = max(0.0, min(image_width, x + width))
    y2 = max(0.0, min(image_height, y + height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    return [x1, y1, clipped_width, clipped_height]


if __name__ == "__main__":
    main()
