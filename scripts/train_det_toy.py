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
from attention2d.detection.matcher import box_cxcywh_to_xyxy, box_iou  # noqa: E402


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
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/det_toy_compare.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    rows = []
    print("device:", device)
    print("model,run_seed,step,loss,eval_iou,eval_recall50,best_iou,best_step,images_per_sec")
    for seed_idx in range(args.seeds):
        run_seed = args.seed + seed_idx
        for model_name in args.models:
            torch.manual_seed(run_seed)
            random.seed(run_seed)
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
            best_iou = -1.0
            best_step = 0
            for step in range(1, args.steps + 1):
                random.seed(10_000_000 + run_seed * 100_000 + step)
                images, targets = sample_square_detection_batch(
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    device=device,
                    torch_seed=30_000_000 + run_seed * 100_000 + step,
                )
                outputs = model(images)
                losses = criterion(outputs, targets)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
                last_loss = float(losses["loss"].item())
                if step % args.eval_every == 0 or step == args.steps:
                    metrics = evaluate_toy(
                        model,
                        args.batch_size,
                        args.image_size,
                        device,
                        seed=20_000_000 + run_seed,
                    )
                    if metrics["iou"] > best_iou:
                        best_iou = metrics["iou"]
                        best_step = step
                    speed = (step * args.batch_size) / max(time.perf_counter() - start, 1e-9)
                    row = {
                        "model": model_name,
                        "run_seed": run_seed,
                        "step": step,
                        "loss": last_loss,
                        "eval_iou": metrics["iou"],
                        "eval_recall50": metrics["recall50"],
                        "best_iou": best_iou,
                        "best_step": best_step,
                        "images_per_sec": speed,
                    }
                    rows.append(row)
                    print(
                        f"{model_name},{run_seed},{step},{last_loss:.4f},"
                        f"{metrics['iou']:.3f},{metrics['recall50']:.3f},"
                        f"{best_iou:.3f},{best_step},{speed:.2f}"
                    )
    write_rows(args.out, rows)
    print("saved_csv:", args.out)
    print_summary(rows)
    print_paired_summary(rows, reference_model="learned")


def sample_square_detection_batch(
    batch_size: int,
    image_size: int,
    device: torch.device,
    torch_seed: int | None = None,
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
    generator = None
    if torch_seed is not None:
        generator = torch.Generator(device=device).manual_seed(torch_seed)
    noise = torch.randn(images.shape, device=device, generator=generator) * 0.02
    return (images + noise).clamp(0, 1), targets


@torch.no_grad()
def evaluate_toy(
    model: TinyAnchorRegionDETR,
    batch_size: int,
    image_size: int,
    device: torch.device,
    batches: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    model.eval()
    ious = []
    recalls = []
    for batch_idx in range(batches):
        random.seed(seed + batch_idx)
        images, targets = sample_square_detection_batch(
            batch_size,
            image_size,
            device,
            torch_seed=seed + batch_idx,
        )
        outputs = model(images)
        probs = outputs["pred_logits"].softmax(dim=-1)[..., 0]
        query_idx = probs.argmax(dim=1)
        pred_boxes = outputs["pred_boxes"][torch.arange(batch_size, device=device), query_idx]
        target_boxes = torch.cat([target["boxes"] for target in targets], dim=0)
        iou_matrix = box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )
        iou = torch.diag(iou_matrix)
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
            fieldnames=[
                "model",
                "run_seed",
                "step",
                "loss",
                "eval_iou",
                "eval_recall50",
                "best_iou",
                "best_step",
                "images_per_sec",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    final_rows = final_rows_by_model_seed(rows)
    print("summary_model,final_iou_mean,best_iou_mean,recall50_mean,images_per_sec_mean")
    for model in sorted({str(row["model"]) for row in final_rows}):
        model_rows = [row for row in final_rows if row["model"] == model]
        print(
            f"{model},"
            f"{mean([float(row['eval_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['best_iou']) for row in model_rows]):.3f},"
            f"{mean([float(row['eval_recall50']) for row in model_rows]):.3f},"
            f"{mean([float(row['images_per_sec']) for row in model_rows]):.2f}"
        )


def print_paired_summary(rows: list[dict[str, float | int | str]], reference_model: str) -> None:
    final_rows = final_rows_by_model_seed(rows)
    if reference_model not in {str(row["model"]) for row in final_rows}:
        return
    print(f"paired_det_toy_vs,{reference_model}")
    print("paired_model,final_iou_delta_mean,final_iou_wins,best_iou_delta_mean,best_iou_wins")
    run_seeds = sorted({int(row["run_seed"]) for row in final_rows})
    for model in sorted({str(row["model"]) for row in final_rows}):
        if model == reference_model:
            continue
        final_deltas = []
        best_deltas = []
        for run_seed in run_seeds:
            ref = find_row(final_rows, model=reference_model, run_seed=run_seed)
            cur = find_row(final_rows, model=model, run_seed=run_seed)
            if ref is None or cur is None:
                continue
            final_deltas.append(float(cur["eval_iou"]) - float(ref["eval_iou"]))
            best_deltas.append(float(cur["best_iou"]) - float(ref["best_iou"]))
        if not final_deltas:
            continue
        print(
            f"{model},"
            f"{mean(final_deltas):.3f},{wins_higher(final_deltas)},"
            f"{mean(best_deltas):.3f},{wins_higher(best_deltas)}"
        )


def final_rows_by_model_seed(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    latest: dict[tuple[str, int], dict[str, float | int | str]] = {}
    for row in rows:
        key = (str(row["model"]), int(row["run_seed"]))
        if key not in latest or int(row["step"]) > int(latest[key]["step"]):
            latest[key] = row
    return list(latest.values())


def find_row(
    rows: list[dict[str, float | int | str]],
    model: str,
    run_seed: int,
) -> dict[str, float | int | str] | None:
    for row in rows:
        if row["model"] == model and int(row["run_seed"]) == run_seed:
            return row
    return None


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def wins_higher(values: list[float]) -> str:
    return f"{sum(value > 0 for value in values)}/{len(values)}"


if __name__ == "__main__":
    main()
