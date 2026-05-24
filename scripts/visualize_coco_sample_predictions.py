"""Visualize selected COCO samples with nearest prediction candidate groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--bbox-decimals", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--max-per-category", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=420)
    parser.add_argument("--contact-cols", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = select_rows(
        read_rows(args.samples),
        max_samples=args.max_samples,
        max_per_category=args.max_per_category,
    )
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in annotations.get("categories", [])
    }
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions_by_image = group_predictions_by_image(predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    panels = []
    for index, sample in enumerate(samples, start=1):
        nearest = nearest_prediction_group(
            sample,
            predictions_by_image.get(parse_int(sample["image_id"]), [])[: args.top_k],
            bbox_decimals=args.bbox_decimals,
        )
        panel = draw_sample(
            sample,
            nearest,
            categories=categories,
            image_dir=args.image_dir,
            thumb_width=args.thumb_width,
            top_candidates=args.top_candidates,
        )
        out_path = args.out_dir / (
            f"{index:02d}_img{sample['image_id']}_cat{sample['category_id']}"
            f"_ann{sample['annotation_id']}.jpg"
        )
        panel.save(out_path, quality=92)
        panels.append(panel)
        manifest.append(
            {
                **sample,
                "rank": index,
                "nearest_iou": format_float(nearest.iou if nearest else 0.0),
                "nearest_box": json.dumps(nearest.box if nearest else []),
                "gt_candidate_present": int(nearest.gt_candidate_present if nearest else False),
                "top_candidates": format_candidates(nearest.candidates[: args.top_candidates], categories)
                if nearest
                else "",
                "overlay_path": str(out_path),
            }
        )
        print(f"saved_sample_prediction_overlay: {out_path}")
    if manifest:
        write_manifest(manifest, args.out_dir / "manifest.csv")
    if panels:
        save_contact_sheet(panels, args.out_dir / "contact_sheet.jpg", cols=args.contact_cols)
        print(f"saved_contact_sheet: {args.out_dir / 'contact_sheet.jpg'}")


class PredictionGroup:
    def __init__(
        self,
        *,
        box: list[float],
        candidates: list[dict[str, Any]],
        iou: float,
        gt_candidate_present: bool,
    ) -> None:
        self.box = box
        self.candidates = candidates
        self.iou = iou
        self.gt_candidate_present = gt_candidate_present


def nearest_prediction_group(
    sample: dict[str, str],
    predictions: list[dict[str, Any]],
    *,
    bbox_decimals: int,
) -> PredictionGroup | None:
    gt_category = parse_int(sample["category_id"])
    gt_box = xywh_to_xyxy(
        [
            parse_float(sample["bbox_x"]),
            parse_float(sample["bbox_y"]),
            parse_float(sample["bbox_w"]),
            parse_float(sample["bbox_h"]),
        ]
    )
    best: PredictionGroup | None = None
    for group in group_by_bbox(predictions, bbox_decimals=bbox_decimals):
        box = xywh_to_xyxy([float(value) for value in group[0]["bbox"]])
        iou = xyxy_iou(gt_box, box)
        candidates = sorted(group, key=lambda row: float(row.get("score", 0.0)), reverse=True)
        candidate_present = any(parse_int(row["category_id"]) == gt_category for row in candidates)
        current = PredictionGroup(
            box=box,
            candidates=candidates,
            iou=iou,
            gt_candidate_present=candidate_present,
        )
        if best is None or current.iou > best.iou:
            best = current
    return best


def group_predictions_by_image(predictions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(parse_int(row["image_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return grouped


def group_by_bbox(predictions: list[dict[str, Any]], *, bbox_decimals: int) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for row in predictions:
        key = tuple(round(float(value), bbox_decimals) for value in row["bbox"])
        grouped.setdefault(key, []).append(row)
    return list(grouped.values())


def draw_sample(
    sample: dict[str, str],
    nearest: PredictionGroup | None,
    *,
    categories: dict[int, str],
    image_dir: Path,
    thumb_width: int,
    top_candidates: int,
) -> Image.Image:
    image = Image.open(image_dir / sample["file_name"]).convert("RGB")
    scale = thumb_width / max(1, image.width)
    panel = image.resize((thumb_width, max(1, round(image.height * scale))))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    gt_box = xywh_to_xyxy(
        [
            parse_float(sample["bbox_x"]),
            parse_float(sample["bbox_y"]),
            parse_float(sample["bbox_w"]),
            parse_float(sample["bbox_h"]),
        ]
    )
    draw_box(
        draw,
        gt_box,
        scale,
        (0, 220, 0),
        f"GT {sample['category_name']} {sample.get('transition', '')}",
        font,
    )
    if nearest is not None:
        pred_color = (40, 160, 255) if nearest.gt_candidate_present else (240, 40, 40)
        draw_box(
            draw,
            nearest.box,
            scale,
            pred_color,
            f"near iou={nearest.iou:.2f} gt_in_cands={int(nearest.gt_candidate_present)}",
            font,
        )
        candidate_text = format_candidates(nearest.candidates[:top_candidates], categories)
        draw_text_block(draw, candidate_text, (4, max(4, panel.height - 54)), font)
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


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    text_box = draw.multiline_textbbox(xy, text, font=font, spacing=2)
    draw.rectangle(text_box, fill=(0, 0, 0))
    draw.multiline_text(xy, text, fill=(255, 255, 255), font=font, spacing=2)


def format_candidates(rows: list[dict[str, Any]], categories: dict[int, str]) -> str:
    chunks = []
    for row in rows:
        category_id = parse_int(row["category_id"])
        chunks.append(f"{categories.get(category_id, category_id)}:{float(row.get('score', 0.0)):.2f}")
    return "\n".join(chunks)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: list[dict[str, str]],
    *,
    max_samples: int,
    max_per_category: int,
) -> list[dict[str, str]]:
    rows = sorted(
        rows,
        key=lambda row: (
            transition_rank(row.get("transition", "")),
            parse_int(row["category_id"]),
            -parse_float(row.get("area_ratio", "0")),
            parse_int(row["image_id"]),
        ),
    )
    selected = []
    category_counts: dict[str, int] = {}
    for row in rows:
        category_id = row["category_id"]
        if max_per_category > 0 and category_counts.get(category_id, 0) >= max_per_category:
            continue
        selected.append(row)
        category_counts[category_id] = category_counts.get(category_id, 0) + 1
        if max_samples > 0 and len(selected) >= max_samples:
            break
    return selected


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


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_manifest: {path}")


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def xyxy_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - inter
    return 0.0 if union <= 0 else inter / union


def transition_rank(label: str) -> int:
    return {
        "persistent_zero_hit": 0,
        "regressed_to_zero_hit": 1,
    }.get(label, 99)


def parse_int(value: Any) -> int:
    return int(float(value or "0"))


def parse_float(value: Any) -> float:
    return float(value or "0")


def format_float(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    main()
