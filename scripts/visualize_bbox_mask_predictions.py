"""Train selected bbox-mask models and save fixed-sample overlay visualizations."""

from __future__ import annotations

import argparse
import sys
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
from compare_bbox_heatmap_models import HEATMAP_MODEL_NAMES, HeatmapWrapper  # noqa: E402
from compare_bbox_mask_models import (  # noqa: E402
    BBoxMaskDataset,
    build_mask_train_loader,
    mask_metrics_from_probs,
    train_mask_model,
)
from compare_bbox_probe_models import load_largest_bbox_samples  # noqa: E402
from compare_imagefolder_models import (  # noqa: E402
    MODEL_SEED_OFFSETS,
    build_model,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument(
        "--anno-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_bbox_val/ILSVRC2013_DET_bbox_val"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=HEATMAP_MODEL_NAMES,
        default=["anchor_only_no_prefill", "no_prefill_local_mix", "fpn_sum_lite"],
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--mask-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("results/bbox_mask_overlays"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    samples = load_largest_bbox_samples(args.anno_root, args.image_root)
    dataset = BBoxMaskDataset(samples=samples, image_size=args.image_size, mask_size=args.mask_size)
    train_set, eval_set = split_dataset(dataset, args.train_frac, args.seed)
    selected = select_eval_indices(eval_set, samples_per_bucket=args.samples_per_bucket)
    selected_set = Subset(eval_set, selected)
    selected_loader = DataLoader(selected_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("device:", device)
    print("selected_samples:", len(selected))
    print("out_dir:", args.out_dir)

    for model_name in args.models:
        model_seed = args.seed + MODEL_SEED_OFFSETS[model_name]
        torch.manual_seed(model_seed)
        backbone = build_model(
            name=model_name,
            embed_dim=args.embed_dim,
            image_size=args.image_size,
            num_classes=2,
        )
        model = HeatmapWrapper(backbone=backbone, embed_dim=args.embed_dim, heatmap_size=args.mask_size).to(device)
        train_loader = build_mask_train_loader(
            train_set=train_set,
            batch_size=args.batch_size,
            seed=10_000_000 + args.seed,
        )
        train_mask_model(
            model=model,
            train_loader=train_loader,
            eval_loader=selected_loader,
            device=device,
            steps=args.steps,
            lr=args.lr,
            eval_every=0,
            dice_weight=args.dice_weight,
            threshold=args.threshold,
        )
        save_model_overlays(
            model=model,
            dataset=selected_set,
            model_name=model_name,
            out_dir=args.out_dir,
            device=device,
            threshold=args.threshold,
            mask_size=args.mask_size,
        )


def select_eval_indices(dataset: Subset, samples_per_bucket: int) -> list[int]:
    buckets: dict[str, list[int]] = {
        "small": [],
        "medium": [],
        "large": [],
        "center": [],
        "offcenter": [],
    }
    for local_idx in range(len(dataset)):
        _, center, mask = dataset[local_idx]
        area = float(mask.mean().item())
        dist = float((center - torch.tensor([0.5, 0.5])).norm().item())
        if area < 0.10:
            buckets["small"].append(local_idx)
        elif area < 0.30:
            buckets["medium"].append(local_idx)
        else:
            buckets["large"].append(local_idx)
        if dist < 0.10:
            buckets["center"].append(local_idx)
        elif dist > 0.22:
            buckets["offcenter"].append(local_idx)

    selected: list[int] = []
    seen: set[int] = set()
    for bucket in ("small", "medium", "large", "center", "offcenter"):
        for idx in buckets[bucket][:samples_per_bucket]:
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
    return selected


def save_model_overlays(
    model: torch.nn.Module,
    dataset: Subset,
    model_name: str,
    out_dir: Path,
    device: torch.device,
    threshold: float,
    mask_size: int,
) -> None:
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            image_tensor, center, target = dataset[idx]
            logits = model(image_tensor.unsqueeze(0).to(device))
            probs = logits.sigmoid().cpu().squeeze(0)
            metrics = mask_metrics_from_probs(
                probs=probs.unsqueeze(0),
                centers=center.unsqueeze(0),
                targets=target.unsqueeze(0),
                threshold=threshold,
                mask_size=mask_size,
            )
            panel = make_panel(image_tensor, target, probs, threshold, title=model_name, metrics=metrics)
            panel.save(model_dir / f"sample_{idx:02d}.png")


def make_panel(
    image_tensor: Tensor,
    target: Tensor,
    probs: Tensor,
    threshold: float,
    title: str,
    metrics: dict[str, float],
) -> Image.Image:
    image = tensor_to_pil(image_tensor)
    gt_overlay = overlay_mask(image, target, color=(0, 220, 0), alpha=100)
    pred_overlay = overlay_mask(image, probs >= threshold, color=(255, 180, 0), alpha=120)
    error = error_map(image, target, probs >= threshold)
    pred_heat = heatmap_to_pil(probs).resize(image.size, Image.Resampling.NEAREST)
    scale = 3
    panels = [
        panel.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
        for panel in [image, gt_overlay, pred_overlay, error, pred_heat]
    ]
    labels = [
        title[:22],
        "GT",
        f"Pred IoU {metrics['iou']:.3f}",
        "Err R=FP B=FN",
        f"Prob Dice {metrics['dice']:.3f}",
    ]
    width, height = panels[0].size
    label_h = 24
    canvas = Image.new("RGB", (width * len(panels), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, (panel, label) in enumerate(zip(panels, labels)):
        x = col * width
        canvas.paste(panel, (x, label_h))
        draw.text((x + 2, 2), label, fill=(0, 0, 0), font=font)
    return canvas


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def overlay_mask(image: Image.Image, mask: Tensor, color: tuple[int, int, int], alpha: int) -> Image.Image:
    mask_img = mask.float().unsqueeze(0).unsqueeze(0)
    mask_img = torch.nn.functional.interpolate(mask_img, size=image.size[::-1], mode="nearest").squeeze()
    overlay = Image.new("RGBA", image.size, color + (0,))
    alpha_mask = Image.fromarray((mask_img.numpy() * alpha).astype("uint8"))
    overlay.putalpha(alpha_mask)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def error_map(image: Image.Image, target: Tensor, pred: Tensor) -> Image.Image:
    target_bool = target.bool()
    pred_bool = pred.bool()
    fp = pred_bool & ~target_bool
    fn = target_bool & ~pred_bool
    base = image.convert("RGBA")
    red = overlay_mask(image, fp.float(), color=(255, 0, 0), alpha=120).convert("RGBA")
    blue = overlay_mask(image, fn.float(), color=(0, 80, 255), alpha=120).convert("RGBA")
    mixed = Image.blend(base, red, 0.65)
    mixed = Image.blend(mixed, blue, 0.65)
    return mixed.convert("RGB")


def heatmap_to_pil(probs: Tensor) -> Image.Image:
    values = probs.float()
    values = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)
    red = (values * 255).numpy().astype("uint8")
    blue = ((1.0 - values) * 120).numpy().astype("uint8")
    green = (values.sqrt() * 180).numpy().astype("uint8")
    rgb = torch.tensor(list(zip(red.flatten(), green.flatten(), blue.flatten())), dtype=torch.uint8)
    rgb = rgb.view(values.shape[0], values.shape[1], 3).numpy()
    return Image.fromarray(rgb)


if __name__ == "__main__":
    main()
