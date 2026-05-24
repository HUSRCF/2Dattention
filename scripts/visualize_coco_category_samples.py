"""Visualize selected COCO category sample rows with GT boxes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=48)
    parser.add_argument("--max-per-category", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--contact-cols", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = select_rows(
        read_rows(args.samples),
        max_samples=args.max_samples,
        max_per_category=args.max_per_category,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    panels = []
    for index, row in enumerate(rows, start=1):
        panel = draw_sample(row, image_dir=args.image_dir, thumb_width=args.thumb_width)
        out_path = args.out_dir / (
            f"{index:02d}_img{row['image_id']}_cat{row['category_id']}"
            f"_ann{row['annotation_id']}.jpg"
        )
        panel.save(out_path, quality=92)
        panels.append(panel)
        manifest.append({**row, "rank": index, "overlay_path": str(out_path)})
        print(f"saved_category_sample_overlay: {out_path}")
    if manifest:
        write_manifest(manifest, args.out_dir / "manifest.csv")
    if panels:
        save_contact_sheet(panels, args.out_dir / "contact_sheet.jpg", cols=args.contact_cols)
        print(f"saved_contact_sheet: {args.out_dir / 'contact_sheet.jpg'}")


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
            int(row["category_id"]),
            -float(row.get("area_ratio", "0") or 0.0),
            int(row["image_id"]),
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


def draw_sample(row: dict[str, str], *, image_dir: Path, thumb_width: int) -> Image.Image:
    image = Image.open(image_dir / row["file_name"]).convert("RGB")
    scale = thumb_width / max(1, image.width)
    panel = image.resize((thumb_width, max(1, round(image.height * scale))))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    bbox = xywh_to_xyxy(
        [
            float(row["bbox_x"]),
            float(row["bbox_y"]),
            float(row["bbox_w"]),
            float(row["bbox_h"]),
        ]
    )
    label = (
        f"GT {row['category_name']} "
        f"{row.get('transition', '')} "
        f"area={float(row.get('area_ratio', '0') or 0.0):.3f}"
    )
    draw_box(draw, bbox, scale, (0, 220, 0), label, font)
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


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


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


def transition_rank(label: str) -> int:
    return {
        "persistent_zero_hit": 0,
        "regressed_to_zero_hit": 1,
    }.get(label, 99)


if __name__ == "__main__":
    main()
