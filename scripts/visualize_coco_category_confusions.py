"""Save overlays for high-IoU wrong-category COCO predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_coco_score_iou_correlation import xyxy_iou
from scripts.evaluate_coco_prediction_recall import predictions_grouped_by_image
from scripts.rescore_coco_predictions_by_oracle_iou import xywh_to_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-visuals", type=int, default=24)
    parser.add_argument("--max-per-pair", type=int, default=4)
    parser.add_argument("--max-per-image", type=int, default=2)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--contact-cols", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    image_dir = args.image_dir or args.annotations.parent
    image_by_id = {int(image["id"]): image for image in annotations.get("images", [])}
    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in annotations.get("categories", [])
    }
    predictions_by_image = predictions_grouped_by_image(predictions)
    samples = collect_confusion_samples(
        annotations=annotations,
        image_by_id=image_by_id,
        predictions_by_image=predictions_by_image,
        top_k=args.top_k,
        iou_threshold=args.iou_threshold,
    )
    selected = select_representative_samples(
        samples,
        max_visuals=args.max_visuals,
        max_per_pair=args.max_per_pair,
        max_per_image=args.max_per_image,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    manifest_rows = []
    for index, sample in enumerate(selected, start=1):
        panel = draw_sample(sample, image_by_id, categories, image_dir, args.thumb_width)
        out_path = args.out_dir / f"{index:02d}_img{sample['image_id']}_gt{sample['gt_category_id']}_pred{sample['pred_category_id']}.jpg"
        panel.save(out_path, quality=92)
        panels.append(panel)
        manifest_rows.append(
            {
                "rank": index,
                "image_id": sample["image_id"],
                "file_name": sample["file_name"],
                "gt_category_id": sample["gt_category_id"],
                "gt_category_name": categories.get(sample["gt_category_id"], str(sample["gt_category_id"])),
                "pred_category_id": sample["pred_category_id"],
                "pred_category_name": categories.get(sample["pred_category_id"], str(sample["pred_category_id"])),
                "iou": sample["iou"],
                "score": sample["score"],
                "overlay_path": str(out_path),
                "top_k": args.top_k,
                "iou_threshold": args.iou_threshold,
                "max_per_pair": args.max_per_pair,
                "max_per_image": args.max_per_image,
            }
        )
        print(
            f"saved_confusion_overlay: {out_path} "
            f"{categories.get(sample['gt_category_id'], sample['gt_category_id'])}"
            f"->{categories.get(sample['pred_category_id'], sample['pred_category_id'])} "
            f"iou={sample['iou']:.3f}"
        )
    if manifest_rows:
        manifest_path = args.out_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"saved_manifest: {manifest_path}")
    if panels:
        contact_path = args.out_dir / "contact_sheet.jpg"
        save_contact_sheet(panels, contact_path, cols=args.contact_cols)
        print(f"saved_contact_sheet: {contact_path}")


def collect_confusion_samples(
    *,
    annotations: dict[str, Any],
    image_by_id: dict[int, dict[str, Any]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    top_k: int,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    samples = []
    for annotation in annotations.get("annotations", []):
        image_id = int(annotation["image_id"])
        gt_category_id = int(annotation["category_id"])
        gt_box = xywh_to_xyxy(annotation["bbox"])
        best_prediction = None
        best_iou = 0.0
        for prediction in predictions_by_image.get(image_id, [])[:top_k]:
            iou = xyxy_iou(gt_box, prediction["box"])
            if iou > best_iou:
                best_iou = iou
                best_prediction = prediction
        if best_prediction is None or best_iou < iou_threshold:
            continue
        pred_category_id = int(best_prediction["category_id"])
        if pred_category_id == gt_category_id:
            continue
        image = image_by_id[image_id]
        samples.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "gt_category_id": gt_category_id,
                "pred_category_id": pred_category_id,
                "gt_box": gt_box,
                "pred_box": best_prediction["box"],
                "score": float(best_prediction.get("score", 0.0)),
                "iou": best_iou,
            }
        )
    return samples


def select_representative_samples(
    samples: list[dict[str, Any]],
    *,
    max_visuals: int,
    max_per_pair: int,
    max_per_image: int,
) -> list[dict[str, Any]]:
    pair_counts: dict[tuple[int, int], int] = {}
    for sample in samples:
        key = (sample["gt_category_id"], sample["pred_category_id"])
        pair_counts[key] = pair_counts.get(key, 0) + 1
    samples.sort(
        key=lambda sample: (
            -pair_counts[(sample["gt_category_id"], sample["pred_category_id"])],
            -sample["iou"],
            -sample["score"],
            sample["image_id"],
        )
    )
    selected = []
    selected_pair_counts: dict[tuple[int, int], int] = {}
    selected_image_counts: dict[int, int] = {}
    for sample in samples:
        pair_key = (sample["gt_category_id"], sample["pred_category_id"])
        image_id = int(sample["image_id"])
        if selected_pair_counts.get(pair_key, 0) >= max_per_pair:
            continue
        if selected_image_counts.get(image_id, 0) >= max_per_image:
            continue
        selected.append(sample)
        selected_pair_counts[pair_key] = selected_pair_counts.get(pair_key, 0) + 1
        selected_image_counts[image_id] = selected_image_counts.get(image_id, 0) + 1
        if len(selected) >= max_visuals:
            break
    return selected


def draw_sample(
    sample: dict[str, Any],
    image_by_id: dict[int, dict[str, Any]],
    categories: dict[int, str],
    image_dir: Path,
    thumb_width: int,
) -> Image.Image:
    image = Image.open(image_dir / sample["file_name"]).convert("RGB")
    scale = thumb_width / max(1, image.width)
    panel = image.resize((thumb_width, max(1, round(image.height * scale))))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    gt_label = categories.get(sample["gt_category_id"], str(sample["gt_category_id"]))
    pred_label = categories.get(sample["pred_category_id"], str(sample["pred_category_id"]))
    draw_box(draw, sample["gt_box"], scale, (0, 220, 0), f"GT {gt_label}", font)
    draw_box(
        draw,
        sample["pred_box"],
        scale,
        (240, 40, 40),
        f"P {pred_label} s={sample['score']:.2f} iou={sample['iou']:.2f}",
        font,
    )
    return panel


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    scale: float,
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = [float(value) * scale for value in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    text_box = draw.textbbox((x1, y1), label, font=font)
    draw.rectangle(text_box, fill=(0, 0, 0))
    draw.text((x1, y1), label, fill=color, font=font)


def save_contact_sheet(panels: list[Image.Image], out_path: Path, *, cols: int) -> None:
    cols = max(1, cols)
    rows = (len(panels) + cols - 1) // cols
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (cols * cell_width, rows * cell_height), (255, 255, 255))
    for index, panel in enumerate(panels):
        x = (index % cols) * cell_width
        y = (index // cols) * cell_height
        sheet.paste(panel, (x, y))
    sheet.save(out_path, quality=92)


if __name__ == "__main__":
    main()
