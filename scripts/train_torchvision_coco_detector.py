"""Train a small torchvision detector on exported COCO-style annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument("--out", type=Path, default=Path("results/torchvision_coco_detector_smoke.csv"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-eval-images", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--weights",
        choices=("none", "coco"),
        default="none",
        help="Use COCO-pretrained detector weights if available; may download if not cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_dataset = CocoDetectionLite(args.train_json, args.image_root, image_size=args.image_size)
    eval_dataset = CocoDetectionLite(args.eval_json, args.image_root, image_size=args.image_size)
    if args.max_train_images > 0:
        train_dataset = Subset(train_dataset, list(range(min(args.max_train_images, len(train_dataset)))))
    if args.max_eval_images > 0:
        eval_dataset = Subset(eval_dataset, list(range(min(args.max_eval_images, len(eval_dataset)))))
    num_classes = max_category_id(train_dataset, eval_dataset) + 1

    model = build_model(num_classes=num_classes, image_size=args.image_size, weights=args.weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_detection)
    iterator = cycle(loader)
    rows = []
    print(f"device: {device}")
    print(f"train_images: {len(train_dataset)}")
    print(f"eval_images: {len(eval_dataset)}")
    print(f"num_classes: {num_classes}")
    print(f"weights: {args.weights}")
    print("step,loss,eval_iou,eval_ap50,eval_ap50_class")
    for step in range(1, args.steps + 1):
        model.train()
        images, targets = next(iterator)
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_detector(model, eval_dataset, device=device)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                **metrics,
            }
            rows.append(row)
            print(
                f"{step},{row['loss']:.4f},{row['eval_iou']:.3f},"
                f"{row['eval_ap50']:.3f},{row['eval_ap50_class']:.3f}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "eval_iou", "eval_ap50", "eval_ap50_class"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv: {args.out}")


class CocoDetectionLite(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Small COCO detection dataset without pycocotools runtime assumptions."""

    def __init__(self, annotation_json: Path, image_root: Path, image_size: int) -> None:
        with annotation_json.open(encoding="utf-8") as handle:
            data = json.load(handle)
        self.image_root = image_root
        self.image_size = image_size
        self.images = sorted(data["images"], key=lambda item: int(item["id"]))
        annotations_by_image: dict[int, list[dict[str, Any]]] = {int(image["id"]): [] for image in self.images}
        for annotation in data["annotations"]:
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        self.annotations_by_image = annotations_by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_info = self.images[index]
        path = self.image_root / image_info["file_name"]
        image = Image.open(path).convert("RGB")
        original_width, original_height = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        scale_x = self.image_size / max(1, original_width)
        scale_y = self.image_size / max(1, original_height)
        boxes = []
        labels = []
        for annotation in self.annotations_by_image.get(int(image_info["id"]), []):
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            boxes.append(
                [
                    x * scale_x,
                    y * scale_y,
                    (x + width) * scale_x,
                    (y + height) * scale_y,
                ]
            )
            labels.append(int(annotation["category_id"]))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(image_info["id"])], dtype=torch.int64),
        }
        return F.to_tensor(image), target


def build_model(num_classes: int, image_size: int, weights: str) -> torch.nn.Module:
    if weights == "coco":
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
            min_size=image_size,
            max_size=image_size,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    return fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        min_size=image_size,
        max_size=image_size,
    )


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_detection(batch: list[tuple[Tensor, dict[str, Tensor]]]) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def max_category_id(*datasets: Dataset[Any]) -> int:
    maximum = 0
    for dataset in datasets:
        base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        if not isinstance(base_dataset, CocoDetectionLite):
            continue
        for annotations in base_dataset.annotations_by_image.values():
            for annotation in annotations:
                maximum = max(maximum, int(annotation["category_id"]))
    return maximum


@torch.no_grad()
def evaluate_detector(model: torch.nn.Module, dataset: Dataset[Any], device: torch.device) -> dict[str, float]:
    model.eval()
    predictions = []
    gt_by_image: dict[int, tuple[Tensor, Tensor]] = {}
    for image, target in DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_detection):
        image_tensor = image[0].to(device)
        output = model([image_tensor])[0]
        image_id = int(target[0]["image_id"].item())
        gt_by_image[image_id] = (target[0]["boxes"].cpu(), target[0]["labels"].cpu())
        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu()
        scores = output["scores"].detach().cpu()
        for box, label, score in zip(boxes, labels, scores, strict=False):
            predictions.append(
                {
                    "image_id": image_id,
                    "box": box,
                    "label": int(label.item()),
                    "score": float(score.item()),
                }
            )
    return {
        "eval_iou": mean_best_iou(predictions, gt_by_image),
        "eval_ap50": ap_at_iou(predictions, gt_by_image, iou_threshold=0.5, class_aware=False),
        "eval_ap50_class": ap_at_iou(predictions, gt_by_image, iou_threshold=0.5, class_aware=True),
    }


def mean_best_iou(predictions: list[dict[str, Any]], gt_by_image: dict[int, tuple[Tensor, Tensor]]) -> float:
    best_values = []
    by_image: dict[int, list[Tensor]] = defaultdict_list_boxes(predictions)
    for image_id, (gt_boxes, _) in gt_by_image.items():
        pred_boxes = torch.stack(by_image[image_id]) if by_image[image_id] else torch.empty((0, 4))
        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            best_values.extend([0.0] * len(gt_boxes))
            continue
        overlaps = box_iou(gt_boxes, pred_boxes)
        best_values.extend(overlaps.max(dim=1).values.tolist())
    return float(sum(best_values) / max(1, len(best_values)))


def ap_at_iou(
    predictions: list[dict[str, Any]],
    gt_by_image: dict[int, tuple[Tensor, Tensor]],
    iou_threshold: float,
    class_aware: bool,
) -> float:
    total_gt = sum(len(boxes) for boxes, _ in gt_by_image.values())
    if total_gt == 0:
        return 0.0
    matched: dict[int, set[int]] = {image_id: set() for image_id in gt_by_image}
    sorted_predictions = sorted(predictions, key=lambda item: float(item["score"]), reverse=True)
    tp = []
    fp = []
    for prediction in sorted_predictions:
        image_id = int(prediction["image_id"])
        gt_boxes, gt_labels = gt_by_image[image_id]
        if gt_boxes.numel() == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue
        overlaps = box_iou(prediction["box"].reshape(1, 4), gt_boxes).squeeze(0)
        if class_aware:
            label_mask = gt_labels == int(prediction["label"])
            overlaps = torch.where(label_mask, overlaps, torch.zeros_like(overlaps))
        best_iou, best_idx = overlaps.max(dim=0)
        if float(best_iou) >= iou_threshold and int(best_idx) not in matched[image_id]:
            matched[image_id].add(int(best_idx))
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return 0.0
    tp_tensor = torch.tensor(tp).cumsum(dim=0)
    fp_tensor = torch.tensor(fp).cumsum(dim=0)
    recall = tp_tensor / total_gt
    precision = tp_tensor / torch.clamp(tp_tensor + fp_tensor, min=1.0)
    return voc_ap(recall, precision)


def defaultdict_list_boxes(predictions: list[dict[str, Any]]) -> dict[int, list[Tensor]]:
    by_image: dict[int, list[Tensor]] = {}
    for prediction in predictions:
        by_image.setdefault(int(prediction["image_id"]), []).append(prediction["box"])
    return by_image


def voc_ap(recall: Tensor, precision: Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for idx in range(mpre.numel() - 1, 0, -1):
        mpre[idx - 1] = torch.maximum(mpre[idx - 1], mpre[idx])
    changing = torch.where(mrec[1:] != mrec[:-1])[0]
    if changing.numel() == 0:
        return 0.0
    ap = torch.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1])
    if math.isnan(float(ap)):
        return 0.0
    return float(ap)


if __name__ == "__main__":
    sys.exit(main())
