"""Train a frozen-backbone classifier on GT crops and relabel proposal crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_coco_image_prior_classifier import (
    build_classifier,
    resolve_device,
    set_seed,
)
from scripts.fuse_coco_predictions import xywh_iou


@dataclass(frozen=True)
class CropSample:
    image_id: int
    file_name: str
    bbox: tuple[float, float, float, float]
    label_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, default=None)
    parser.add_argument("--eval-annotations", type=Path, required=True)
    parser.add_argument("--eval-image-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--backbone", choices=["tiny", "resnet18_frozen", "resnet50_frozen"], default="resnet50_frozen")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--top-k-categories", type=int, default=5)
    parser.add_argument("--category-score-mode", choices=["keep", "multiply"], default="multiply")
    parser.add_argument("--train-crop-source", choices=["gt", "matched_predictions"], default="gt")
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_data = json.loads(args.train_annotations.read_text(encoding="utf-8"))
    eval_data = json.loads(args.eval_annotations.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    category_ids = sorted({int(category["id"]) for category in train_data.get("categories", [])})
    category_to_index = {category_id: index for index, category_id in enumerate(category_ids)}
    index_to_category = {index: category_id for category_id, index in category_to_index.items()}

    if args.train_crop_source == "gt":
        train_samples = build_crop_samples(train_data, category_to_index=category_to_index)
    else:
        if args.train_predictions is None:
            raise ValueError("--train-predictions is required when --train-crop-source matched_predictions")
        train_samples = build_matched_prediction_crop_samples(
            train_data,
            predictions=json.loads(args.train_predictions.read_text(encoding="utf-8")),
            category_to_index=category_to_index,
            iou_threshold=args.match_iou_threshold,
        )
    eval_samples = build_crop_samples(eval_data, category_to_index=category_to_index)
    model, preprocess = build_classifier(
        args.backbone,
        num_classes=len(category_ids),
        embed_dim=args.embed_dim,
        image_size=args.image_size,
    )
    model = model.to(device)
    train_loader = DataLoader(
        CocoCropDataset(samples=train_samples, image_root=args.train_image_root, transform=preprocess),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_loader = DataLoader(
        CocoCropDataset(samples=eval_samples, image_root=args.eval_image_root, transform=preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    train_loss = 0.0
    train_acc = 0.0
    cycle_loader = iter_cycle(train_loader)
    model.train()
    for _ in range(args.steps):
        crops, labels = next(cycle_loader)
        crops = crops.to(device)
        labels = labels.to(device)
        logits = model(crops)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())
        train_acc = float((logits.argmax(dim=1) == labels).float().mean().item())

    eval_metrics = evaluate_crop_classifier(model, eval_loader, device)
    relabeled = relabel_prediction_crops(
        model=model,
        predictions=predictions,
        annotations=eval_data,
        image_root=args.eval_image_root,
        index_to_category=index_to_category,
        transform=preprocess,
        device=device,
        top_k=args.top_k_categories,
        score_mode=args.category_score_mode,
        batch_size=args.batch_size,
    )
    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.out_predictions.write_text(json.dumps(relabeled, indent=2) + "\n", encoding="utf-8")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_csv,
        {
            "device": str(device),
            "seed": args.seed,
            "backbone": args.backbone,
            "steps": args.steps,
            "classes": len(category_ids),
            "train_crops": len(train_samples),
            "eval_crops": len(eval_samples),
            "applied_predictions": len(relabeled),
            "train_loss": train_loss,
            "train_batch_acc": train_acc,
            "top_k_categories": args.top_k_categories,
            "category_score_mode": args.category_score_mode,
            "train_crop_source": args.train_crop_source,
            "match_iou_threshold": args.match_iou_threshold,
            **eval_metrics,
            "out_predictions": str(args.out_predictions),
        },
    )
    print(f"saved_crop_prior_predictions: {args.out_predictions}")
    print(f"saved_csv: {args.out_csv}")
    print(f"eval_crop_acc={eval_metrics['eval_crop_acc']:.4f} top5={eval_metrics['eval_crop_top5_acc']:.4f}")


class CocoCropDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, *, samples: list[CropSample], image_root: Path, transform: Any) -> None:
        self.samples = samples
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        image = Image.open(self.image_root / sample.file_name).convert("RGB")
        crop = crop_image(image, sample.bbox)
        return self.transform(crop), torch.tensor(sample.label_index, dtype=torch.long)


def build_crop_samples(data: dict[str, Any], *, category_to_index: dict[int, int]) -> list[CropSample]:
    file_by_image = {int(image["id"]): str(image["file_name"]) for image in data.get("images", [])}
    rows = []
    for annotation in data.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in file_by_image or category_id not in category_to_index:
            continue
        bbox = tuple(float(value) for value in annotation["bbox"])
        if len(bbox) != 4 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue
        rows.append(
            CropSample(
                image_id=image_id,
                file_name=file_by_image[image_id],
                bbox=bbox,
                label_index=category_to_index[category_id],
            )
        )
    if not rows:
        raise ValueError("no crop samples were built")
    return rows


def build_matched_prediction_crop_samples(
    data: dict[str, Any],
    *,
    predictions: list[dict[str, Any]],
    category_to_index: dict[int, int],
    iou_threshold: float,
) -> list[CropSample]:
    file_by_image = {int(image["id"]): str(image["file_name"]) for image in data.get("images", [])}
    gt_by_image = annotations_by_image(data, category_to_index=category_to_index)
    rows = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        if image_id not in file_by_image:
            continue
        gt_rows = gt_by_image.get(image_id, [])
        if not gt_rows:
            continue
        bbox = tuple(float(value) for value in prediction["bbox"])
        if len(bbox) != 4 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue
        best = max(gt_rows, key=lambda row: xywh_iou(list(bbox), list(row["bbox"])))
        best_iou = xywh_iou(list(bbox), list(best["bbox"]))
        if best_iou < iou_threshold:
            continue
        rows.append(
            CropSample(
                image_id=image_id,
                file_name=file_by_image[image_id],
                bbox=bbox,
                label_index=int(best["label_index"]),
            )
        )
    if not rows:
        raise ValueError("no matched prediction crop samples were built")
    return rows


def annotations_by_image(
    data: dict[str, Any],
    *,
    category_to_index: dict[int, int],
) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for annotation in data.get("annotations", []):
        category_id = int(annotation["category_id"])
        if category_id not in category_to_index:
            continue
        bbox = tuple(float(value) for value in annotation["bbox"])
        if len(bbox) != 4 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue
        rows.setdefault(int(annotation["image_id"]), []).append(
            {
                "bbox": bbox,
                "label_index": category_to_index[category_id],
            }
        )
    return rows


def crop_image(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    x, y, width, height = bbox
    left = max(0, min(image.width - 1, int(round(x))))
    top = max(0, min(image.height - 1, int(round(y))))
    right = max(left + 1, min(image.width, int(round(x + max(width, 1.0)))))
    bottom = max(top + 1, min(image.height, int(round(y + max(height, 1.0)))))
    return image.crop((left, top, right, bottom))


@torch.no_grad()
def evaluate_crop_classifier(model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    correct_top5 = 0
    total = 0
    for crops, labels in loader:
        crops = crops.to(device)
        labels = labels.to(device)
        logits = model(crops)
        topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
        correct += int((topk[:, 0] == labels).sum().item())
        correct_top5 += int((topk == labels[:, None]).any(dim=1).sum().item())
        total += int(labels.numel())
    return {
        "eval_crop_acc": correct / max(1, total),
        "eval_crop_top5_acc": correct_top5 / max(1, total),
    }


@torch.no_grad()
def relabel_prediction_crops(
    *,
    model: nn.Module,
    predictions: list[dict[str, Any]],
    annotations: dict[str, Any],
    image_root: Path,
    index_to_category: dict[int, int],
    transform: Any,
    device: torch.device,
    top_k: int,
    score_mode: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    file_by_image = {int(image["id"]): str(image["file_name"]) for image in annotations.get("images", [])}
    apply_rows = [row for row in predictions if int(row["image_id"]) in file_by_image]
    relabeled: list[dict[str, Any]] = []
    for start in range(0, len(apply_rows), batch_size):
        batch = apply_rows[start : start + batch_size]
        crops = []
        for prediction in batch:
            image = Image.open(image_root / file_by_image[int(prediction["image_id"])]).convert("RGB")
            crops.append(transform(crop_image(image, tuple(float(value) for value in prediction["bbox"]))))
        logits = model(torch.stack(crops).to(device))
        probs = logits.softmax(dim=1).cpu()
        topk = probs.topk(k=min(top_k, probs.shape[1]), dim=1)
        for prediction, indices, values in zip(batch, topk.indices.tolist(), topk.values.tolist(), strict=False):
            for label_index, category_score in zip(indices, values, strict=True):
                row = dict(prediction)
                row["category_id"] = int(index_to_category[int(label_index)])
                if score_mode == "multiply":
                    row["score"] = float(row.get("score", 1.0)) * float(category_score)
                elif score_mode != "keep":
                    raise ValueError(f"unsupported score mode: {score_mode}")
                relabeled.append(row)
    return relabeled


def iter_cycle(loader: DataLoader[tuple[Tensor, Tensor]]):
    while True:
        for batch in loader:
            yield batch


def write_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
