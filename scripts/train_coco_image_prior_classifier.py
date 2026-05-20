"""Train a lightweight image-level COCO category prior and relabel detections."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image


@dataclass(frozen=True)
class ImagePriorSample:
    image_id: int
    file_name: str
    label_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, required=True)
    parser.add_argument("--eval-annotations", type=Path, required=True)
    parser.add_argument("--eval-image-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--prior", choices=["largest", "most_frequent"], default="largest")
    parser.add_argument("--backbone", choices=["tiny", "resnet18_frozen", "resnet50_frozen"], default="tiny")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--top-k-categories", type=int, default=1)
    parser.add_argument("--category-score-mode", choices=["keep", "multiply"], default="keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_data = json.loads(args.train_annotations.read_text(encoding="utf-8"))
    eval_data = json.loads(args.eval_annotations.read_text(encoding="utf-8"))
    category_ids = sorted({int(category["id"]) for category in train_data.get("categories", [])})
    category_to_index = {category_id: index for index, category_id in enumerate(category_ids)}
    index_to_category = {index: category_id for category_id, index in category_to_index.items()}

    train_samples = build_samples(train_data, prior=args.prior, category_to_index=category_to_index)
    eval_samples = build_samples(eval_data, prior=args.prior, category_to_index=category_to_index)
    model, preprocess = build_classifier(
        args.backbone,
        num_classes=len(category_ids),
        embed_dim=args.embed_dim,
        image_size=args.image_size,
    )
    train_dataset = CocoImagePriorDataset(
        samples=train_samples,
        image_root=args.train_image_root,
        transform=preprocess,
    )
    eval_dataset = CocoImagePriorDataset(
        samples=eval_samples,
        image_root=args.eval_image_root,
        transform=preprocess,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loss = 0.0
    train_acc = 0.0
    cycle_loader = iter_cycle(train_loader)
    model.train()
    for _ in range(args.steps):
        images, labels, _ = next(cycle_loader)
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())
        train_acc = float((logits.argmax(dim=1) == labels).float().mean().item())

    eval_metrics, eval_predictions = evaluate_topk(
        model,
        eval_loader,
        device,
        top_k=max(5, args.top_k_categories),
    )
    image_category_by_id = topk_categories_to_category_ids(eval_predictions, index_to_category)
    relabeled = relabel_predictions(
        json.loads(args.predictions.read_text(encoding="utf-8")),
        image_category_by_id,
        top_k=args.top_k_categories,
        score_mode=args.category_score_mode,
    )
    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.out_predictions.write_text(json.dumps(relabeled, indent=2) + "\n", encoding="utf-8")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_csv,
        {
            "device": str(device),
            "seed": args.seed,
            "prior": args.prior,
            "backbone": args.backbone,
            "steps": args.steps,
            "classes": len(category_ids),
            "train_images": len(train_samples),
            "eval_images": len(eval_samples),
            "train_loss": train_loss,
            "train_batch_acc": train_acc,
            "top_k_categories": args.top_k_categories,
            "category_score_mode": args.category_score_mode,
            **eval_metrics,
            "out_predictions": str(args.out_predictions),
        },
    )
    print(f"saved_image_prior_predictions: {args.out_predictions}")
    print(f"saved_csv: {args.out_csv}")
    print(
        f"eval_acc={eval_metrics['eval_acc']:.4f} "
        f"top5={eval_metrics['eval_top5_acc']:.4f} "
        f"classes={len(category_ids)}"
    )


class CocoImagePriorDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        *,
        samples: list[ImagePriorSample],
        image_root: Path,
        transform: transforms.Compose,
    ) -> None:
        self.samples = samples
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        image = Image.open(self.image_root / sample.file_name).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(sample.label_index, dtype=torch.long),
            torch.tensor(sample.image_id, dtype=torch.long),
        )


class TinyImagePriorClassifier(nn.Module):
    def __init__(self, *, num_classes: int, embed_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        pooled = features.mean(dim=(2, 3))
        return self.head(pooled)


class FrozenResNetImagePriorClassifier(nn.Module):
    def __init__(self, *, name: str, num_classes: int) -> None:
        super().__init__()
        if name == "resnet18_frozen":
            weights = models.ResNet18_Weights.DEFAULT
            base = models.resnet18(weights=weights)
        elif name == "resnet50_frozen":
            weights = models.ResNet50_Weights.DEFAULT
            base = models.resnet50(weights=weights)
        else:
            raise ValueError(f"unsupported frozen backbone: {name}")
        feature_dim = int(base.fc.in_features)
        base.fc = nn.Identity()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.backbone = base
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features)


def build_classifier(
    backbone: str,
    *,
    num_classes: int,
    embed_dim: int,
    image_size: int,
) -> tuple[nn.Module, transforms.Compose]:
    if backbone == "tiny":
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        return TinyImagePriorClassifier(num_classes=num_classes, embed_dim=embed_dim), transform
    if backbone == "resnet18_frozen":
        weights = models.ResNet18_Weights.DEFAULT
        return FrozenResNetImagePriorClassifier(name=backbone, num_classes=num_classes), weights.transforms()
    if backbone == "resnet50_frozen":
        weights = models.ResNet50_Weights.DEFAULT
        return FrozenResNetImagePriorClassifier(name=backbone, num_classes=num_classes), weights.transforms()
    raise ValueError(f"unsupported backbone: {backbone}")


def build_samples(
    data: dict[str, Any],
    *,
    prior: str,
    category_to_index: dict[int, int],
) -> list[ImagePriorSample]:
    file_by_image = {int(image["id"]): str(image["file_name"]) for image in data.get("images", [])}
    category_by_image = image_category_prior(data, prior=prior)
    samples = []
    for image_id, category_id in sorted(category_by_image.items()):
        if image_id not in file_by_image or category_id not in category_to_index:
            continue
        samples.append(
            ImagePriorSample(
                image_id=image_id,
                file_name=file_by_image[image_id],
                label_index=category_to_index[category_id],
            )
        )
    if not samples:
        raise ValueError("no image-prior samples were built")
    return samples


def image_category_prior(data: dict[str, Any], *, prior: str) -> dict[int, int]:
    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in data.get("annotations", []):
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    category_by_image: dict[int, int] = {}
    for image_id, annotations in by_image.items():
        if prior == "largest":
            best = max(
                annotations,
                key=lambda row: (
                    float(row.get("area", bbox_area(row.get("bbox", [0, 0, 0, 0])))),
                    -int(row["category_id"]),
                ),
            )
            category_by_image[image_id] = int(best["category_id"])
        elif prior == "most_frequent":
            counts: dict[int, int] = {}
            areas: dict[int, float] = {}
            for row in annotations:
                category_id = int(row["category_id"])
                counts[category_id] = counts.get(category_id, 0) + 1
                areas[category_id] = areas.get(category_id, 0.0) + float(
                    row.get("area", bbox_area(row.get("bbox", [0, 0, 0, 0])))
                )
            category_by_image[image_id] = max(
                counts,
                key=lambda category_id: (counts[category_id], areas[category_id], -category_id),
            )
        else:
            raise ValueError(f"unknown prior: {prior}")
    return category_by_image


def bbox_area(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> tuple[dict[str, float], dict[int, int]]:
    model.eval()
    correct = 0
    correct_top5 = 0
    total = 0
    predictions: dict[int, int] = {}
    with torch.no_grad():
        for images, labels, image_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predicted = logits.argmax(dim=1)
            topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            correct += int((predicted == labels).sum().item())
            correct_top5 += int((topk == labels[:, None]).any(dim=1).sum().item())
            total += int(labels.numel())
            for image_id, label_index in zip(image_ids.tolist(), predicted.cpu().tolist(), strict=False):
                predictions[int(image_id)] = int(label_index)
    return {
        "eval_acc": correct / max(1, total),
        "eval_top5_acc": correct_top5 / max(1, total),
    }, predictions


def evaluate_topk(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    *,
    top_k: int,
) -> tuple[dict[str, float], dict[int, list[tuple[int, float]]]]:
    model.eval()
    correct = 0
    correct_top5 = 0
    total = 0
    predictions: dict[int, list[tuple[int, float]]] = {}
    with torch.no_grad():
        for images, labels, image_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = logits.softmax(dim=1)
            topk = probs.topk(k=min(top_k, logits.shape[1]), dim=1)
            predicted = topk.indices[:, 0]
            correct += int((predicted == labels).sum().item())
            correct_top5 += int((topk.indices[:, : min(5, topk.indices.shape[1])] == labels[:, None]).any(dim=1).sum().item())
            total += int(labels.numel())
            for image_id, indices, values in zip(
                image_ids.tolist(),
                topk.indices.cpu().tolist(),
                topk.values.cpu().tolist(),
                strict=False,
            ):
                predictions[int(image_id)] = [
                    (int(label_index), float(score)) for label_index, score in zip(indices, values, strict=True)
                ]
    return {
        "eval_acc": correct / max(1, total),
        "eval_top5_acc": correct_top5 / max(1, total),
    }, predictions


def topk_categories_to_category_ids(
    predictions: dict[int, list[tuple[int, float]]],
    index_to_category: dict[int, int],
) -> dict[int, list[tuple[int, float]]]:
    return {
        image_id: [(int(index_to_category[label_index]), float(score)) for label_index, score in rows]
        for image_id, rows in predictions.items()
    }


def relabel_predictions(
    predictions: list[dict[str, Any]],
    category_by_image: dict[int, int] | dict[int, list[tuple[int, float]]],
    *,
    top_k: int = 1,
    score_mode: str = "keep",
) -> list[dict[str, Any]]:
    rows = []
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        categories = normalized_category_rows(category_by_image.get(image_id))
        if not categories:
            rows.append(dict(prediction))
            continue
        for category_id, category_score in categories[:top_k]:
            row = dict(prediction)
            row["category_id"] = int(category_id)
            if score_mode == "multiply":
                row["score"] = float(row.get("score", 1.0)) * float(category_score)
            elif score_mode != "keep":
                raise ValueError(f"unsupported score mode: {score_mode}")
            rows.append(row)
    return rows


def normalized_category_rows(value: Any) -> list[tuple[int, float]]:
    if value is None:
        return []
    if isinstance(value, int):
        return [(value, 1.0)]
    return [(int(category_id), float(score)) for category_id, score in value]


def iter_cycle(loader: DataLoader[tuple[Tensor, Tensor, Tensor]]):
    while True:
        for batch in loader:
            yield batch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def write_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
