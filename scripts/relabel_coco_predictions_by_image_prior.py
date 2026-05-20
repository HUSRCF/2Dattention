"""Relabel COCO detections with an image-level category prior from annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior", choices=["largest", "most_frequent"], default="largest")
    parser.add_argument("--drop-images-without-prior", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relabeled = relabel_predictions_by_image_prior(
        annotation_json=args.annotations,
        prediction_json=args.predictions,
        prior=args.prior,
        drop_images_without_prior=args.drop_images_without_prior,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(relabeled, indent=2) + "\n", encoding="utf-8")
    print(f"saved_image_prior_predictions: {args.out}")
    print(f"predictions: {len(relabeled)}")


def relabel_predictions_by_image_prior(
    *,
    annotation_json: Path,
    prediction_json: Path,
    prior: str = "largest",
    drop_images_without_prior: bool = False,
) -> list[dict[str, Any]]:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_json.read_text(encoding="utf-8"))
    category_by_image = image_category_prior(data, prior=prior)

    relabeled = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        category_id = category_by_image.get(image_id)
        if category_id is None:
            if drop_images_without_prior:
                continue
            relabeled.append(dict(prediction))
            continue
        row = dict(prediction)
        row["category_id"] = category_id
        relabeled.append(row)
    return relabeled


def image_category_prior(data: dict[str, Any], *, prior: str = "largest") -> dict[int, int]:
    if prior not in {"largest", "most_frequent"}:
        raise ValueError(f"unknown prior: {prior}")

    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in data.get("annotations", []):
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    category_by_image: dict[int, int] = {}
    for image_id, annotations in by_image.items():
        if prior == "largest":
            best = max(
                annotations,
                key=lambda annotation: (
                    float(annotation.get("area", bbox_area(annotation.get("bbox", [0, 0, 0, 0])))),
                    -int(annotation["category_id"]),
                ),
            )
            category_by_image[image_id] = int(best["category_id"])
        else:
            counts: dict[int, int] = {}
            areas: dict[int, float] = {}
            for annotation in annotations:
                category_id = int(annotation["category_id"])
                counts[category_id] = counts.get(category_id, 0) + 1
                areas[category_id] = areas.get(category_id, 0.0) + float(
                    annotation.get("area", bbox_area(annotation.get("bbox", [0, 0, 0, 0])))
                )
            category_by_image[image_id] = max(
                counts,
                key=lambda category_id: (counts[category_id], areas[category_id], -category_id),
            )
    return category_by_image


def bbox_area(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


if __name__ == "__main__":
    main()
