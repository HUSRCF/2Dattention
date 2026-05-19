"""Train a tiny real-DET model and save fixed-sample prediction overlays."""

from __future__ import annotations

import argparse
import sys
from itertools import cycle
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from attention2d import get_best_device  # noqa: E402
from attention2d.detection.anchor_region_detr import TinyAnchorRegionDETR  # noqa: E402
from attention2d.detection.losses import DetectionCriterion  # noqa: E402
from train_det_real import (  # noqa: E402
    REAL_MODEL_CONFIGS,
    RealDetDataset,
    box_cxcywh_to_xyxy,
    build_label_map,
    build_splits,
    det_collate,
    filter_samples,
    forward_real_detector,
    load_real_det_samples,
    sample_matches_slice,
    targets_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument(
        "--anno-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_bbox_val/ILSVRC2013_DET_bbox_val"),
    )
    parser.add_argument("--model", choices=tuple(REAL_MODEL_CONFIGS), default="local_anchor_residual_query")
    parser.add_argument("--top-classes", type=int, default=10)
    parser.add_argument("--label-map-source", choices=("all", "train"), default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--eval-slice-filter",
        choices=("none", "small", "medium", "large", "center", "offcenter", "offcenter_only"),
        default="offcenter_only",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--local-blocks", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--no-object-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-visuals", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("results/det_real_prediction_overlays"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model.endswith("_quality_head") or "matchqual" in args.model or args.model.endswith("_calib"):
        raise ValueError("visualization script currently supports base detector variants, not quality/calib variants")
    device = get_best_device()
    all_samples = load_real_det_samples(args.anno_root, args.image_root)
    raw_by_id = {sample.image_id: sample for sample in all_samples}
    label_source_samples = all_samples
    if args.label_map_source == "train":
        order = torch.randperm(len(all_samples), generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_size = max(1, min(len(order), int(len(order) * args.train_frac)))
        label_source_samples = [all_samples[idx] for idx in order[:train_size]]
    label_to_id = build_label_map(label_source_samples, top_classes=args.top_classes)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    samples = filter_samples(all_samples, set(label_to_id), max_samples=args.max_samples)
    train_set, _, eval_set, _ = build_splits(
        samples=samples,
        label_to_id=label_to_id,
        image_size=args.image_size,
        max_objects=args.max_objects,
        train_frac=args.train_frac,
        calibration_frac=0.0,
        calibration_source="heldout",
        eval_slice_filter=args.eval_slice_filter,
        seed=args.seed,
    )
    model = build_model(args, num_classes=len(label_to_id)).to(device)
    criterion = DetectionCriterion(num_classes=len(label_to_id), no_object_weight=args.no_object_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=det_collate,
        generator=torch.Generator().manual_seed(10_000_000 + args.seed),
    )
    train(model, criterion, optimizer, cycle(loader), args.steps, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("device:", device)
    print("eval_samples:", len(eval_set))
    print("out_dir:", args.out_dir)
    save_overlays(model, eval_set, raw_by_id, id_to_label, args, device)


def build_model(args: argparse.Namespace, num_classes: int) -> TinyAnchorRegionDETR:
    feature_mode, query_init, query_refine, mask_aux_mode, gate_init, _ = REAL_MODEL_CONFIGS[args.model]
    if mask_aux_mode != "none":
        raise ValueError("visualization script currently supports models without auxiliary mask losses")
    return TinyAnchorRegionDETR(
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        num_queries=args.num_queries,
        local_blocks=args.local_blocks,
        feature_mode=feature_mode,
        query_init=query_init,
        query_refine=query_refine,
        query_mask_gate_init=gate_init,
    )


def train(
    model: TinyAnchorRegionDETR,
    criterion: DetectionCriterion,
    optimizer: torch.optim.Optimizer,
    loader_iter,
    steps: int,
    device: torch.device,
) -> None:
    model.train()
    for step in range(1, steps + 1):
        images, targets = next(loader_iter)
        images = images.to(device)
        targets = targets_to_device(targets, device)
        outputs = forward_real_detector(model, images, targets)
        losses = criterion(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        if step == 1 or step == steps or step % 100 == 0:
            print(f"step={step} loss={float(losses['loss'].detach().cpu()):.4f}")


def save_overlays(
    model: TinyAnchorRegionDETR,
    eval_set: Subset,
    raw_by_id: dict[str, object],
    id_to_label: dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not isinstance(eval_set.dataset, RealDetDataset):
        raise TypeError("expected RealDetDataset-backed eval subset")
    selected = select_eval_indices(eval_set, args.max_visuals, args.max_objects, args.eval_slice_filter)
    model.eval()
    panels = []
    with torch.no_grad():
        for rank, raw_idx in enumerate(selected, start=1):
            image_tensor, target = eval_set.dataset[int(raw_idx)]
            outputs = model(image_tensor.unsqueeze(0).to(device))
            sample = eval_set.dataset.samples[int(raw_idx)]
            panel = draw_prediction_panel(
                image_tensor=image_tensor,
                raw_sample=raw_by_id[sample.image_id],
                target=target,
                outputs={key: value[0].cpu() for key, value in outputs.items() if isinstance(value, Tensor)},
                id_to_label=id_to_label,
                top_k=args.top_k,
            )
            out_path = args.out_dir / f"{rank:02d}_{sample.image_id}.jpg"
            panel.save(out_path)
            panels.append((sample.image_id, panel))
            print("saved_overlay:", out_path)
    save_contact_sheet(panels, args.out_dir / "contact_sheet.jpg")


def select_eval_indices(eval_set: Subset, max_visuals: int, max_objects: int, slice_name: str) -> list[int]:
    if not isinstance(eval_set.dataset, RealDetDataset):
        raise TypeError("expected RealDetDataset-backed eval subset")
    selected = []
    for raw_idx in eval_set.indices:
        sample = eval_set.dataset.samples[int(raw_idx)]
        if sample_matches_slice(sample, max_objects=max_objects, slice_name=slice_name):
            selected.append(int(raw_idx))
        if len(selected) >= max_visuals:
            break
    return selected


def draw_prediction_panel(
    image_tensor: Tensor,
    raw_sample,
    target: dict[str, Tensor],
    outputs: dict[str, Tensor],
    id_to_label: dict[int, str],
    top_k: int,
) -> Image.Image:
    image = tensor_to_image(image_tensor)
    width, height = image.size
    raw_panel = image.copy()
    gt_panel = image.copy()
    pred_panel = image.copy()
    draw_raw = ImageDraw.Draw(raw_panel)
    draw_gt = ImageDraw.Draw(gt_panel)
    draw_pred = ImageDraw.Draw(pred_panel)
    font = ImageFont.load_default()
    for box in raw_sample.boxes:
        raw_box = torch.tensor(
            [
                ((box.xmin + box.xmax) / 2.0) / raw_sample.width,
                ((box.ymin + box.ymax) / 2.0) / raw_sample.height,
                max(1e-6, (box.xmax - box.xmin) / raw_sample.width),
                max(1e-6, (box.ymax - box.ymin) / raw_sample.height),
            ],
            dtype=torch.float32,
        )
        draw_box(draw_raw, raw_box, width, height, fill=(0, 120, 255), label=f"raw:{box.label}", font=font)
    for label, box in zip(target["labels"], target["boxes"]):
        draw_box(draw_gt, box, width, height, fill=(0, 220, 0), label=id_to_label[int(label.item())], font=font)
    probs = outputs["pred_logits"].softmax(dim=-1)[:, :-1]
    scores, labels = probs.max(dim=-1)
    order = scores.argsort(descending=True)[:top_k]
    target_xyxy = box_cxcywh_to_xyxy(target["boxes"])
    ious = box_iou_safe(box_cxcywh_to_xyxy(outputs["pred_boxes"]), target_xyxy)
    for rank, query_idx_tensor in enumerate(order, start=1):
        query_idx = int(query_idx_tensor.item())
        best_iou = float(ious[query_idx].max().item()) if ious.numel() else 0.0
        label = f"{rank}:{id_to_label[int(labels[query_idx].item())]} {float(scores[query_idx].item()):.2f} iou={best_iou:.2f}"
        color = (255, 180, 0) if best_iou >= 0.5 else (255, 60, 60)
        draw_box(draw_pred, outputs["pred_boxes"][query_idx], width, height, fill=color, label=label, font=font)
    return hstack([image, raw_panel, gt_panel, pred_panel], ["image", "raw gt", "retained gt", "top predictions"])


def tensor_to_image(tensor: Tensor) -> Image.Image:
    array = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array).resize((256, 256), resample=Image.BILINEAR)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: Tensor,
    width: int,
    height: int,
    fill: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    cx, cy, bw, bh = [float(value) for value in box.tolist()]
    xmin = (cx - bw / 2) * width
    xmax = (cx + bw / 2) * width
    ymin = (cy - bh / 2) * height
    ymax = (cy + bh / 2) * height
    draw.rectangle((xmin, ymin, xmax, ymax), outline=fill, width=3)
    draw.text((max(0, xmin + 2), max(0, ymin + 2)), label[:36], fill=fill, font=font)


def hstack(images: list[Image.Image], labels: list[str]) -> Image.Image:
    font = ImageFont.load_default()
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images)
    canvas = Image.new("RGB", (tile_w * len(images), tile_h + 16), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (image, label) in enumerate(zip(images, labels)):
        x = idx * tile_w
        canvas.paste(image, (x, 16))
        draw.text((x + 2, 2), label, fill=(0, 0, 0), font=font)
    return canvas


def save_contact_sheet(panels: list[tuple[str, Image.Image]], path: Path) -> None:
    if not panels:
        return
    cols = 2
    rows = math_ceil(len(panels), cols)
    tile_w = max(panel.width for _, panel in panels)
    tile_h = max(panel.height for _, panel in panels)
    canvas = Image.new("RGB", (cols * tile_w, rows * (tile_h + 16)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (image_id, panel) in enumerate(panels):
        row, col = divmod(idx, cols)
        x = col * tile_w
        y = row * (tile_h + 16)
        draw.text((x + 2, y + 2), image_id, fill=(0, 0, 0), font=font)
        canvas.paste(panel, (x, y + 16))
    canvas.save(path)
    print("saved_contact_sheet:", path)


def math_ceil(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def box_iou_safe(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    from train_det_real import box_iou

    return box_iou(boxes1, boxes2)


if __name__ == "__main__":
    main()
