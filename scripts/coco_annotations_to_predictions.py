"""Convert COCO annotations into COCO detection prediction rows.

This is useful for evaluating selected pseudo-label annotations against GT with
the same COCOeval path used for exported detector predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--score-field",
        default="teacher_score",
        help="Annotation field to use as prediction score when present.",
    )
    parser.add_argument(
        "--default-score",
        type=float,
        default=1.0,
        help="Fallback score for annotations without --score-field.",
    )
    parser.add_argument(
        "--skip-crowd",
        action="store_true",
        help="Skip annotations with iscrowd=1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = annotations_to_predictions(
        args.annotations,
        score_field=args.score_field,
        default_score=args.default_score,
        skip_crowd=args.skip_crowd,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"saved_predictions: {args.out}")
    print(f"predictions: {len(predictions)}")


def annotations_to_predictions(
    annotation_json: Path,
    *,
    score_field: str,
    default_score: float,
    skip_crowd: bool,
) -> list[dict[str, Any]]:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    image_ids = {int(image["id"]) for image in data.get("images", [])}
    category_ids = {int(category["id"]) for category in data.get("categories", [])}
    predictions: list[dict[str, Any]] = []
    for annotation in data.get("annotations", []):
        if skip_crowd and int(annotation.get("iscrowd", 0)) == 1:
            continue
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in image_ids:
            raise ValueError(f"annotation references missing image_id={image_id}")
        if category_id not in category_ids:
            raise ValueError(f"annotation references missing category_id={category_id}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            raise ValueError(f"annotation has invalid bbox: {annotation.get('id')}")
        score = float(annotation.get(score_field, default_score))
        predictions.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(value) for value in bbox],
                "score": score,
            }
        )
    return predictions


if __name__ == "__main__":
    main()
