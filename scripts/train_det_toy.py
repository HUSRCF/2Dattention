"""Train tiny DETR variants on a synthetic square-detection task."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import get_best_device  # noqa: E402
from attention2d.detection import DetectionCriterion, TinyAnchorRegionDETR  # noqa: E402
from attention2d.detection.matcher import box_cxcywh_to_xyxy, generalized_box_iou  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("learned", "anchor"), default=["learned", "anchor"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/det_toy_compare.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    rows = []
    print("device:", device)
    print("model,step,loss,eval_iou,eval_recall50,images_per_sec")
    for model_name in args.models:
        torch.manual_seed(args.seed + (1000 if model_name == "anchor" else 0))
        random.seed(args.seed)
        model = TinyAnchorRegionDETR(
            embed_dim=args.embed_dim,
            num_classes=1,
            num_queries=args.num_queries,
            query_init=model_name,
        ).to(device)
        criterion = DetectionCriterion(num_classes=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        start = time.perf_counter()
        last_loss = 0.0
        for step in range(1, args.steps + 1):
            images, targets = sample_square_detection_batch(
                batch_size=args.batch_size,
                image_size=args.image_size,
                device=device,
            )
            outputs = model(images)
            losses = criterion(outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            last_loss = float(losses["loss"].item())
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate_toy(model, args.batch_size, args.image_size, device)
                speed = (step * args.batch_size) / max(time.perf_counter() - start, 1e-9)
                row = {
                    "model": model_name,
                    "step": step,
                    "loss": last_loss,
                    "eval_iou": metrics["iou"],
                    "eval_recall50": metrics["recall50"],
                    "images_per_sec": speed,
                }
                rows.append(row)
                print(
                    f"{model_name},{step},{last_loss:.4f},"
                    f"{metrics['iou']:.3f},{metrics['recall50']:.3f},{speed:.2f}"
                )
    write_rows(args.out, rows)
    print("saved_csv:", args.out)


def sample_square_detection_batch(
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    targets = []
    for idx in range(batch_size):
        side = random.randint(image_size // 8, image_size // 3)
        left = random.randint(0, image_size - side - 1)
        top = random.randint(0, image_size - side - 1)
        color = torch.tensor([0.1, 0.8, 0.2], device=device).view(3, 1, 1)
        images[idx, :, top : top + side, left : left + side] = color
        cx = (left + side / 2) / image_size
        cy = (top + side / 2) / image_size
        box = torch.tensor([[cx, cy, side / image_size, side / image_size]], device=device)
        targets.append(
            {
                "labels": torch.zeros(1, dtype=torch.long, device=device),
                "boxes": box,
            }
        )
    noise = torch.randn_like(images) * 0.02
    return (images + noise).clamp(0, 1), targets


@torch.no_grad()
def evaluate_toy(
    model: TinyAnchorRegionDETR,
    batch_size: int,
    image_size: int,
    device: torch.device,
    batches: int = 8,
) -> dict[str, float]:
    model.eval()
    ious = []
    recalls = []
    for _ in range(batches):
        images, targets = sample_square_detection_batch(batch_size, image_size, device)
        outputs = model(images)
        probs = outputs["pred_logits"].softmax(dim=-1)[..., 0]
        query_idx = probs.argmax(dim=1)
        pred_boxes = outputs["pred_boxes"][torch.arange(batch_size, device=device), query_idx]
        target_boxes = torch.cat([target["boxes"] for target in targets], dim=0)
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )
        iou = torch.diag(giou).clamp(min=0)
        ious.append(iou.cpu())
        recalls.append((iou >= 0.5).float().cpu())
    model.train()
    all_ious = torch.cat(ious)
    all_recalls = torch.cat(recalls)
    return {
        "iou": float(all_ious.mean().item()),
        "recall50": float(all_recalls.mean().item()),
    }


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "step", "loss", "eval_iou", "eval_recall50", "images_per_sec"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
