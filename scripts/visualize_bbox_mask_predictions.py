"""Train selected bbox-mask models and save fixed-sample overlay visualizations."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--display-size", type=int, default=192)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument(
        "--image-ids",
        nargs="+",
        default=None,
        help="fixed image ids to visualize, e.g. ILSVRC2012_val_00045292",
    )
    parser.add_argument(
        "--image-id-manifest",
        type=Path,
        default=None,
        help="optional CSV manifest with an image_id column; used before bucket sampling",
    )
    parser.add_argument("--manifest-top-k", type=int, default=8)
    parser.add_argument(
        "--comparison-name",
        default="comparison_grid.png",
        help="filename for the cross-model fixed-sample comparison grid",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/bbox_mask_overlays"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    samples = load_largest_bbox_samples(args.anno_root, args.image_root)
    dataset = BBoxMaskDataset(samples=samples, image_size=args.image_size, mask_size=args.mask_size)
    train_set, eval_set = split_dataset(dataset, args.train_frac, args.seed)
    image_ids = fixed_image_ids(args)
    if image_ids:
        selected = select_image_id_indices(dataset, image_ids)
        selected_set = Subset(dataset, selected)
    else:
        selected = select_eval_indices(eval_set, samples_per_bucket=args.samples_per_bucket)
        selected_set = Subset(eval_set, selected)
    selected_loader = DataLoader(selected_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(args.out_dir, args.models, args.comparison_name)
    print("device:", device)
    print("selected_samples:", len(selected))
    print("fixed_image_ids:", ",".join(image_ids) if image_ids else "none")
    print("out_dir:", args.out_dir)

    predictions: dict[str, list[tuple[Tensor, dict[str, float]]]] = {}
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
        predictions[model_name] = collect_predictions(
            model=model,
            dataset=selected_set,
            device=device,
            threshold=args.threshold,
            mask_size=args.mask_size,
        )
    save_cross_model_grid(
        dataset=selected_set,
        model_predictions=predictions,
        out_path=args.out_dir / args.comparison_name,
        threshold=args.threshold,
        display_size=args.display_size,
    )


def fixed_image_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.image_id_manifest is not None:
        with args.image_id_manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "image_id" not in (reader.fieldnames or []):
                raise ValueError(f"manifest has no image_id column: {args.image_id_manifest}")
            for row in reader:
                image_id = row["image_id"].strip()
                if image_id:
                    ids.append(image_id)
                if args.manifest_top_k > 0 and len(ids) >= args.manifest_top_k:
                    break
    if args.image_ids:
        ids.extend(args.image_ids)
    deduped = []
    seen = set()
    for image_id in ids:
        if image_id in seen:
            continue
        deduped.append(image_id)
        seen.add(image_id)
    return deduped


def select_image_id_indices(dataset: BBoxMaskDataset, image_ids: list[str]) -> list[int]:
    by_id = {sample.image_id: idx for idx, sample in enumerate(dataset.samples)}
    missing = [image_id for image_id in image_ids if image_id not in by_id]
    if missing:
        raise ValueError(f"image ids not found in dataset: {missing}")
    return [by_id[image_id] for image_id in image_ids]


def clear_previous_outputs(out_dir: Path, model_names: list[str], comparison_name: str) -> None:
    comparison_path = out_dir / comparison_name
    if comparison_path.exists():
        comparison_path.unlink()
    for model_name in model_names:
        model_dir = out_dir / model_name
        if not model_dir.exists():
            continue
        for path in model_dir.glob("*.png"):
            path.unlink()


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
            sample = subset_sample(dataset, idx)
            panel.save(model_dir / f"sample_{idx:02d}_{sample.image_id}.png")


def collect_predictions(
    model: torch.nn.Module,
    dataset: Subset,
    device: torch.device,
    threshold: float,
    mask_size: int,
) -> list[tuple[Tensor, dict[str, float]]]:
    model.eval()
    rows = []
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
            rows.append((probs, metrics))
    return rows


def save_cross_model_grid(
    dataset: Subset,
    model_predictions: dict[str, list[tuple[Tensor, dict[str, float]]]],
    out_path: Path,
    threshold: float,
    display_size: int,
) -> None:
    if not model_predictions:
        return
    model_names = list(model_predictions)
    rows = []
    for idx in range(len(dataset)):
        _, _, target = dataset[idx]
        sample = subset_sample(dataset, idx)
        image = load_display_image(sample.image_path, size=display_size)
        panels = [image, overlay_mask(image, target, color=(0, 220, 0), alpha=100)]
        labels = [sample.image_id[-8:], "GT"]
        for model_name in model_names:
            probs, metrics = model_predictions[model_name][idx]
            panels.append(overlay_mask(image, probs >= threshold, color=(255, 180, 0), alpha=120))
            labels.append(f"{model_name[:12]} {metrics['iou']:.2f}")
        rows.append(make_row(panels, labels, scale=1))

    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    grid = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def subset_sample(dataset: Subset, local_idx: int):
    if isinstance(dataset.dataset, Subset):
        inner_idx = dataset.indices[local_idx]
        return subset_sample(dataset.dataset, inner_idx)
    return dataset.dataset.samples[dataset.indices[local_idx]]


def load_display_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return image.resize((size, size), Image.Resampling.LANCZOS)


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


def make_row(panels: list[Image.Image], labels: list[str], scale: int) -> Image.Image:
    resized = [
        panel.resize((panel.width * scale, panel.height * scale), Image.Resampling.NEAREST)
        for panel in panels
    ]
    width = resized[0].width
    height = resized[0].height
    label_h = 24
    canvas = Image.new("RGB", (width * len(resized), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, (panel, label) in enumerate(zip(resized, labels)):
        x = col * width
        canvas.paste(panel, (x, label_h))
        draw.text((x + 2, 2), label[:22], fill=(0, 0, 0), font=font)
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
