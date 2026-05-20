"""Relabel COCO detections with a torchvision ImageNet classifier prior."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--class-index", type=Path, required=True)
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet18"], default="resnet18")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--use-source-image-id",
        action="store_true",
        help="Use image['source_image_id'] as the prediction image id; useful for crop metadata.",
    )
    parser.add_argument(
        "--restrict-to-annotation-categories",
        action="store_true",
        help="If top-1 is outside annotation categories, use highest-probability annotation category.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    class_index = json.loads(args.class_index.read_text(encoding="utf-8"))
    index_to_synset = {int(index): row[0] for index, row in class_index.items()}
    category_id_by_synset = {
        str(category["name"]): int(category["id"]) for category in data.get("categories", [])
    }
    allowed_category_ids = set(category_id_by_synset.values())
    dataset = CocoImageDataset(data=data, image_root=args.image_root, use_source_image_id=args.use_source_image_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model, preprocess = build_model(args.model)
    model = model.to(device).eval()

    image_category_by_id: dict[int, int] = {}
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for images, image_ids, gt_category_ids in loader:
            images = images.to(device)
            logits = model(images)
            probs = logits.softmax(dim=1).cpu()
            for row_probs, image_id, gt_category_id in zip(
                probs,
                image_ids.tolist(),
                gt_category_ids.tolist(),
                strict=False,
            ):
                chosen_category = choose_category(
                    row_probs,
                    index_to_synset=index_to_synset,
                    category_id_by_synset=category_id_by_synset,
                    restrict_to_annotation_categories=args.restrict_to_annotation_categories,
                    allowed_category_ids=allowed_category_ids,
                )
                if chosen_category is not None:
                    image_category_by_id[int(image_id)] = int(chosen_category)
                top_indices = row_probs.topk(k=min(5, row_probs.numel())).indices.tolist()
                top_categories = [
                    category_id_by_synset[index_to_synset[index]]
                    for index in top_indices
                    if index_to_synset[index] in category_id_by_synset
                ]
                correct_top1 += int(chosen_category == int(gt_category_id))
                correct_top5 += int(int(gt_category_id) in top_categories)
                total += 1

    relabeled = relabel_predictions(
        json.loads(args.predictions.read_text(encoding="utf-8")),
        image_category_by_id,
    )
    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.out_predictions.write_text(json.dumps(relabeled, indent=2) + "\n", encoding="utf-8")
    row = {
        "device": str(device),
        "model": args.model,
        "images": total,
        "mapped_images": len(image_category_by_id),
        "top1_acc": correct_top1 / max(1, total),
        "top5_acc": correct_top5 / max(1, total),
        "restricted": int(args.restrict_to_annotation_categories),
        "out_predictions": str(args.out_predictions),
    }
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"saved_imagenet_prior_predictions: {args.out_predictions}")
    print(f"saved_csv: {args.out_csv}")
    print(f"top1={row['top1_acc']:.4f} top5={row['top5_acc']:.4f} mapped={len(image_category_by_id)}/{total}")


class CocoImageDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, *, data: dict[str, Any], image_root: Path, use_source_image_id: bool = False) -> None:
        self.images = sorted(data.get("images", []), key=lambda image: int(image["id"]))
        self.image_root = image_root
        self.use_source_image_id = use_source_image_id
        self.gt_category_by_image = largest_category_by_image(data)
        _, self.preprocess = build_model("resnet18")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image = self.images[index]
        image_id = int(image["id"])
        output_image_id = int(image.get("source_image_id", image_id)) if self.use_source_image_id else image_id
        pil_image = Image.open(self.image_root / image["file_name"]).convert("RGB")
        return (
            self.preprocess(pil_image),
            torch.tensor(output_image_id, dtype=torch.long),
            torch.tensor(self.gt_category_by_image.get(image_id, -1), dtype=torch.long),
        )


def build_model(name: str):
    if name != "resnet18":
        raise ValueError(f"unsupported model: {name}")
    weights = models.ResNet18_Weights.DEFAULT
    return models.resnet18(weights=weights), weights.transforms()


def choose_category(
    probs: Tensor,
    *,
    index_to_synset: dict[int, str],
    category_id_by_synset: dict[str, int],
    restrict_to_annotation_categories: bool,
    allowed_category_ids: set[int],
) -> int | None:
    for index in torch.argsort(probs, descending=True).tolist():
        synset = index_to_synset[int(index)]
        category_id = category_id_by_synset.get(synset)
        if category_id is None:
            continue
        if restrict_to_annotation_categories and category_id not in allowed_category_ids:
            continue
        return int(category_id)
    return None


def largest_category_by_image(data: dict[str, Any]) -> dict[int, int]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for annotation in data.get("annotations", []):
        rows.setdefault(int(annotation["image_id"]), []).append(annotation)
    category_by_image = {}
    for image_id, annotations in rows.items():
        best = max(
            annotations,
            key=lambda annotation: (
                float(annotation.get("area", bbox_area(annotation.get("bbox", [0, 0, 0, 0])))),
                -int(annotation["category_id"]),
            ),
        )
        category_by_image[image_id] = int(best["category_id"])
    return category_by_image


def bbox_area(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def relabel_predictions(
    predictions: list[dict[str, Any]],
    image_category_by_id: dict[int, int],
) -> list[dict[str, Any]]:
    rows = []
    for prediction in predictions:
        row = dict(prediction)
        category_id = image_category_by_id.get(int(row["image_id"]))
        if category_id is not None:
            row["category_id"] = int(category_id)
        rows.append(row)
    return rows


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


if __name__ == "__main__":
    main()
