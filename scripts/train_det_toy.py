"""Train tiny DETR variants on a synthetic square-detection task."""

from __future__ import annotations

import argparse
import csv
import itertools
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


MODEL_CONFIGS = {
    "learned": ("anchor", "learned"),
    "anchor": ("anchor", "anchor"),
    "local_learned": ("local", "learned"),
    "local_anchor": ("local", "anchor"),
    "anchor_learned": ("anchor", "learned"),
    "anchor_anchor": ("anchor", "anchor"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=["learned", "anchor"],
        help=(
            "Model variants. Legacy aliases: learned=anchor_learned, "
            "anchor=anchor_anchor."
        ),
    )
    parser.add_argument("--reference-model", choices=tuple(MODEL_CONFIGS), default="learned")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument(
        "--toy-mode",
        choices=("single", "multi", "distractor", "multi_distractor"),
        default="single",
    )
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/det_toy_compare.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_objects > args.num_queries:
        raise ValueError("--max-objects must be <= --num-queries for the current brute-force matcher")
    device = get_best_device()
    rows = []
    print("device:", device)
    print("model,run_seed,step,loss,eval_iou,eval_recall50,best_iou,best_step,images_per_sec")
    for seed_idx in range(args.seeds):
        run_seed = args.seed + seed_idx
        for model_name in args.models:
            feature_mode, query_init = MODEL_CONFIGS[model_name]
            torch.manual_seed(run_seed)
            random.seed(run_seed)
            model = TinyAnchorRegionDETR(
                embed_dim=args.embed_dim,
                num_classes=1,
                num_queries=args.num_queries,
                feature_mode=feature_mode,
                query_init=query_init,
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
                    toy_mode=args.toy_mode,
                    max_objects=args.max_objects,
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
                        toy_mode=args.toy_mode,
                        max_objects=args.max_objects,
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
    print_paired_summary(rows, reference_model=args.reference_model)


def sample_square_detection_batch(
    batch_size: int,
    image_size: int,
    device: torch.device,
    toy_mode: str = "single",
    max_objects: int = 3,
    torch_seed: int | None = None,
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    targets = []
    for idx in range(batch_size):
        target_count, distractor_count = object_counts(toy_mode, max_objects)
        occupied_boxes: list[list[float]] = []
        boxes = []
        for _ in range(distractor_count):
            occupied_boxes.append(
                draw_square(
                    images[idx],
                    image_size,
                    device,
                    color=(0.85, 0.15, 0.15),
                    existing_boxes=occupied_boxes,
                )
            )
        for _ in range(target_count):
            box = draw_square(
                images[idx],
                image_size,
                device,
                color=(0.1, 0.8, 0.2),
                existing_boxes=occupied_boxes,
            )
            boxes.append(box)
            occupied_boxes.append(box)
        box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        targets.append(
            {
                "labels": torch.zeros(target_count, dtype=torch.long, device=device),
                "boxes": box_tensor,
            }
        )
    generator = None
    if torch_seed is not None:
        generator = torch.Generator(device=device).manual_seed(torch_seed)
    noise = torch.randn(images.shape, device=device, generator=generator) * 0.02
    return (images + noise).clamp(0, 1), targets


def object_counts(toy_mode: str, max_objects: int) -> tuple[int, int]:
    max_objects = max(1, max_objects)
    if toy_mode == "single":
        return 1, 0
    if toy_mode == "multi":
        return random.randint(1, max_objects), 0
    if toy_mode == "distractor":
        return 1, random.randint(1, max_objects)
    if toy_mode == "multi_distractor":
        return random.randint(1, max_objects), random.randint(1, max_objects)
    raise ValueError(f"unknown toy_mode: {toy_mode}")


def draw_square(
    image: Tensor,
    image_size: int,
    device: torch.device,
    color: tuple[float, float, float],
    existing_boxes: list[list[float]] | None = None,
    max_iou: float = 0.02,
) -> list[float]:
    existing_boxes = existing_boxes or []
    box = sample_non_overlapping_square(image_size, existing_boxes, max_iou=max_iou)
    cx, cy, width, height = box
    side = int(round(width * image_size))
    left = int(round((cx - width / 2) * image_size))
    top = int(round((cy - height / 2) * image_size))
    color_tensor = torch.tensor(color, device=device).view(3, 1, 1)
    image[:, top : top + side, left : left + side] = color_tensor
    return box


def sample_non_overlapping_square(
    image_size: int,
    existing_boxes: list[list[float]],
    max_iou: float,
    attempts: int = 100,
) -> list[float]:
    best_box = None
    best_overlap = float("inf")
    for _ in range(attempts):
        box = sample_square_box(image_size)
        overlap = max((cxcywh_iou(box, existing) for existing in existing_boxes), default=0.0)
        if overlap <= max_iou:
            return box
        if overlap < best_overlap:
            best_box = box
            best_overlap = overlap
    if best_box is None:
        return sample_square_box(image_size)
    return best_box


def sample_square_box(image_size: int) -> list[float]:
    side = random.randint(image_size // 8, image_size // 3)
    left = random.randint(0, image_size - side - 1)
    top = random.randint(0, image_size - side - 1)
    cx = (left + side / 2) / image_size
    cy = (top + side / 2) / image_size
    return [cx, cy, side / image_size, side / image_size]


def cxcywh_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = cxcywh_to_xyxy_list(box_a)
    bx1, by1, bx2, by2 = cxcywh_to_xyxy_list(box_b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def cxcywh_to_xyxy_list(box: list[float]) -> tuple[float, float, float, float]:
    cx, cy, width, height = box
    return (
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    )


@torch.no_grad()
def evaluate_toy(
    model: TinyAnchorRegionDETR,
    batch_size: int,
    image_size: int,
    device: torch.device,
    toy_mode: str = "single",
    max_objects: int = 3,
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
            toy_mode=toy_mode,
            max_objects=max_objects,
            torch_seed=seed + batch_idx,
        )
        outputs = model(images)
        for sample_idx, target in enumerate(targets):
            matched_iou = match_targets_by_iou(
                outputs["pred_boxes"][sample_idx],
                target["boxes"],
            )
            ious.append(matched_iou.cpu())
            recalls.append((matched_iou >= 0.5).float().cpu())
    model.train()
    all_ious = torch.cat(ious)
    all_recalls = torch.cat(recalls)
    return {
        "iou": float(all_ious.mean().item()),
        "recall50": float(all_recalls.mean().item()),
    }


def match_targets_by_iou(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    target_count = target_boxes.shape[0]
    if target_count == 0:
        return target_boxes.new_zeros((0,))
    query_count = pred_boxes.shape[0]
    if target_count > query_count:
        raise ValueError("target_count must be <= query_count for match_targets_by_iou")
    matched_count = target_count
    iou_matrix = box_iou(
        box_cxcywh_to_xyxy(pred_boxes),
        box_cxcywh_to_xyxy(target_boxes),
    )
    target_indices = torch.arange(matched_count, device=target_boxes.device)
    best_values = None
    best_score = None
    for query_indices_tuple in itertools.permutations(range(query_count), matched_count):
        query_indices = torch.tensor(query_indices_tuple, device=target_boxes.device)
        values = iou_matrix[query_indices, target_indices]
        score = values.sum()
        if best_score is None or bool(score > best_score):
            best_score = score
            best_values = values
    if best_values is None:
        return target_boxes.new_zeros((target_count,))
    return best_values


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
