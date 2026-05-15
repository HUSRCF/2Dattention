"""Select clearer ILSVRC DET validation images for qualitative overlays.

The selector stays inside the same DET validation domain used by the current
experiments. It ranks samples by box size, object separation, image resolution,
edge sharpness, and border clipping, then exports a manifest plus GT overlay
panels for manual figure curation.
"""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Box:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    truncated: int
    difficult: int
    clamped: bool


@dataclass(frozen=True)
class Candidate:
    image_id: str
    image_path: Path
    xml_path: Path
    width: int
    height: int
    boxes: tuple[Box, ...]
    largest_index: int
    largest: Box
    group: str
    score: float
    area_frac: float
    center_x: float
    center_y: float
    center_dist: float
    max_iou: float
    min_center_dist: float
    border_margin: float
    sharpness: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument(
        "--anno-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_bbox_val/ILSVRC2013_DET_bbox_val"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/clear_det_overlay_samples"),
        help="directory for the manifest, overlays, and contact sheet",
    )
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--extra-fill", type=int, default=8)
    parser.add_argument("--min-area", type=float, default=0.02)
    parser.add_argument("--max-overlap", type=float, default=0.55)
    parser.add_argument("--min-image-side", type=int, default=250)
    parser.add_argument("--min-box-px", type=int, default=24)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument(
        "--dominance-ratio",
        type=float,
        default=0.0,
        help="optional largest/second-largest area ratio filter; 0 disables it",
    )
    parser.add_argument("--max-per-label", type=int, default=0, help="0 disables label capping")
    parser.add_argument("--thumb-size", type=int, default=220)
    parser.add_argument("--max-contact-cols", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_candidates(
        anno_root=args.anno_root,
        image_root=args.image_root,
        min_area=args.min_area,
        max_overlap=args.max_overlap,
        min_image_side=args.min_image_side,
        min_box_px=args.min_box_px,
        min_margin=args.min_margin,
        dominance_ratio=args.dominance_ratio,
    )
    selected = select_grouped(
        candidates,
        per_group=args.per_group,
        extra_fill=args.extra_fill,
        max_per_label=args.max_per_label,
    )

    overlays_dir = args.out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_overlays(overlays_dir)
    write_manifest(args.out_dir / "manifest.csv", selected)
    overlay_paths = []
    for rank, candidate in enumerate(selected, start=1):
        overlay_path = overlays_dir / f"{rank:02d}_{candidate.group}_{candidate.image_id}.jpg"
        save_overlay(candidate, overlay_path, thumb_size=args.thumb_size)
        overlay_paths.append(overlay_path)
    save_contact_sheet(
        selected,
        overlay_paths,
        args.out_dir / "contact_sheet.jpg",
        max_cols=args.max_contact_cols,
    )

    print("candidates:", len(candidates))
    print("selected:", len(selected))
    print("manifest:", args.out_dir / "manifest.csv")
    print("overlays:", overlays_dir)
    print("contact_sheet:", args.out_dir / "contact_sheet.jpg")
    print_group_summary(selected)


def load_candidates(
    anno_root: Path,
    image_root: Path,
    min_area: float,
    max_overlap: float,
    min_image_side: int,
    min_box_px: int,
    min_margin: float,
    dominance_ratio: float,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for xml_path in sorted(anno_root.glob("*.xml")):
        image_id, width, height, boxes = parse_annotation(xml_path)
        image_path = image_root / f"{image_id}.JPEG"
        if not image_path.exists() or not boxes:
            continue
        if min(width, height) < min_image_side:
            continue

        largest_index, largest = max(enumerate(boxes), key=lambda item: box_area(item[1]))
        if largest.xmax - largest.xmin < min_box_px or largest.ymax - largest.ymin < min_box_px:
            continue
        area_frac = box_area(largest) / max(1.0, float(width * height))
        if area_frac < min_area:
            continue
        max_iou = max_pairwise_iou(boxes)
        if max_iou > max_overlap:
            continue
        if dominance_ratio > 0 and len(boxes) > 1:
            areas = sorted((box_area(box) for box in boxes), reverse=True)
            if areas[1] > 0 and areas[0] / areas[1] < dominance_ratio:
                continue

        center_x, center_y = box_center(largest, width, height)
        center_dist = math.hypot(center_x - 0.5, center_y - 0.5)
        min_center_dist = min_pairwise_center_distance(boxes, width, height)
        border_margin = normalized_border_margin(largest, width, height)
        if border_margin < min_margin:
            continue
        sharpness = image_sharpness(image_path)
        group = sample_group(area_frac=area_frac, center_dist=center_dist)
        score = quality_score(
            area_frac=area_frac,
            max_iou=max_iou,
            min_center_dist=min_center_dist,
            border_margin=border_margin,
            width=width,
            height=height,
            object_count=len(boxes),
            sharpness=sharpness,
        )
        candidates.append(
            Candidate(
                image_id=image_id,
                image_path=image_path,
                xml_path=xml_path,
                width=width,
                height=height,
                boxes=tuple(boxes),
                largest_index=largest_index,
                largest=largest,
                group=group,
                score=score,
                area_frac=area_frac,
                center_x=center_x,
                center_y=center_y,
                center_dist=center_dist,
                max_iou=max_iou,
                min_center_dist=min_center_dist,
                border_margin=border_margin,
                sharpness=sharpness,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def parse_annotation(xml_path: Path) -> tuple[str, int, int, list[Box]]:
    root = ET.parse(xml_path).getroot()
    image_id = required_text(root, "filename")
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing size in {xml_path}")
    width = int(required_text(size, "width"))
    height = int(required_text(size, "height"))
    boxes = []
    for obj in root.findall("object"):
        label = required_text(obj, "name")
        truncated = optional_int(obj, "truncated")
        difficult = optional_int(obj, "difficult")
        box = obj.find("bndbox")
        if box is None:
            continue
        raw_xmin = int(required_text(box, "xmin"))
        raw_ymin = int(required_text(box, "ymin"))
        raw_xmax = int(required_text(box, "xmax"))
        raw_ymax = int(required_text(box, "ymax"))
        xmin = max(0, min(width, raw_xmin))
        ymin = max(0, min(height, raw_ymin))
        xmax = max(0, min(width, raw_xmax))
        ymax = max(0, min(height, raw_ymax))
        clamped = (xmin, ymin, xmax, ymax) != (raw_xmin, raw_ymin, raw_xmax, raw_ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append(
            Box(
                label=label,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                truncated=truncated,
                difficult=difficult,
                clamped=clamped,
            )
        )
    return image_id, width, height, boxes


def required_text(root: ET.Element, tag: str) -> str:
    child = root.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"missing XML tag: {tag}")
    return child.text.strip()


def optional_int(root: ET.Element, tag: str) -> int:
    child = root.find(tag)
    if child is None or child.text is None:
        return 0
    try:
        return int(child.text.strip())
    except ValueError:
        return 0


def select_grouped(
    candidates: list[Candidate],
    per_group: int,
    extra_fill: int,
    max_per_label: int,
) -> list[Candidate]:
    groups = [
        "small_center",
        "small_offcenter",
        "medium_center",
        "medium_offcenter",
        "large_center",
        "large_offcenter",
    ]
    selected: list[Candidate] = []
    seen: set[str] = set()
    label_counts: dict[str, int] = {}
    for group in groups:
        group_items = [candidate for candidate in candidates if candidate.group == group]
        picked = 0
        for candidate in group_items:
            if label_capped(candidate, label_counts, max_per_label):
                continue
            selected.append(candidate)
            seen.add(candidate.image_id)
            label_counts[candidate.largest.label] = label_counts.get(candidate.largest.label, 0) + 1
            picked += 1
            if picked >= per_group:
                break
    for candidate in candidates:
        if len(selected) >= per_group * len(groups) + extra_fill:
            break
        if candidate.image_id in seen:
            continue
        if label_capped(candidate, label_counts, max_per_label):
            continue
        selected.append(candidate)
        seen.add(candidate.image_id)
        label_counts[candidate.largest.label] = label_counts.get(candidate.largest.label, 0) + 1
    return selected


def label_capped(candidate: Candidate, label_counts: dict[str, int], max_per_label: int) -> bool:
    return max_per_label > 0 and label_counts.get(candidate.largest.label, 0) >= max_per_label


def sample_group(area_frac: float, center_dist: float) -> str:
    if area_frac < 0.10:
        size = "small"
    elif area_frac < 0.30:
        size = "medium"
    else:
        size = "large"
    position = "center" if center_dist < 0.18 else "offcenter"
    return f"{size}_{position}"


def quality_score(
    area_frac: float,
    max_iou: float,
    min_center_dist: float,
    border_margin: float,
    width: int,
    height: int,
    object_count: int,
    sharpness: float,
) -> float:
    area_score = min(area_frac / 0.35, 1.0)
    separation_score = 0.7 * (1.0 - max_iou) + 0.3 * min(min_center_dist / 0.45, 1.0)
    margin_score = min(max(border_margin, 0.0) / 0.08, 1.0)
    resolution_score = min(min(width, height) / 500.0, 1.0)
    count_score = 1.0 if object_count <= 3 else max(0.3, 1.0 - 0.12 * (object_count - 3))
    sharpness_score = min(sharpness / 1800.0, 1.0)
    return (
        0.25 * area_score
        + 0.25 * separation_score
        + 0.15 * margin_score
        + 0.15 * resolution_score
        + 0.15 * sharpness_score
        + 0.05 * count_score
    )


def box_area(box: Box) -> int:
    return max(0, box.xmax - box.xmin) * max(0, box.ymax - box.ymin)


def box_center(box: Box, width: int, height: int) -> tuple[float, float]:
    return ((box.xmin + box.xmax) / (2.0 * width), (box.ymin + box.ymax) / (2.0 * height))


def normalized_border_margin(box: Box, width: int, height: int) -> float:
    return min(
        box.xmin / width,
        box.ymin / height,
        1.0 - box.xmax / width,
        1.0 - box.ymax / height,
    )


def max_pairwise_iou(boxes: list[Box]) -> float:
    if len(boxes) < 2:
        return 0.0
    return max(box_iou(left, right) for i, left in enumerate(boxes) for right in boxes[i + 1 :])


def min_pairwise_center_distance(boxes: list[Box], width: int, height: int) -> float:
    if len(boxes) < 2:
        return 1.0
    centers = [box_center(box, width, height) for box in boxes]
    return min(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for i, left in enumerate(centers)
        for right in centers[i + 1 :]
    )


def box_iou(left: Box, right: Box) -> float:
    ixmin = max(left.xmin, right.xmin)
    iymin = max(left.ymin, right.ymin)
    ixmax = min(left.xmax, right.xmax)
    iymax = min(left.ymax, right.ymax)
    inter = max(0, ixmax - ixmin) * max(0, iymax - iymin)
    union = box_area(left) + box_area(right) - inter
    return inter / union if union > 0 else 0.0


def image_sharpness(image_path: Path) -> float:
    try:
        with Image.open(image_path) as image:
            gray = image.convert("L")
            gray.thumbnail((256, 256))
            array = np.asarray(gray, dtype=np.float32)
    except OSError:
        return 0.0
    if array.shape[0] < 2 or array.shape[1] < 2:
        return 0.0
    grad_x = np.diff(array, axis=1)
    grad_y = np.diff(array, axis=0)
    return float(grad_x.var() + grad_y.var())


def write_manifest(path: Path, selected: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "group",
                "score",
                "image_id",
                "image_path",
                "xml_path",
                "width",
                "height",
                "object_count",
                "labels",
                "largest_index",
                "largest_label",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "xmin_norm",
                "ymin_norm",
                "xmax_norm",
                "ymax_norm",
                "largest_truncated",
                "largest_difficult",
                "largest_clamped",
                "area_frac",
                "center_x",
                "center_y",
                "center_dist",
                "max_iou",
                "min_center_dist",
                "border_margin",
                "sharpness",
            ]
        )
        for rank, candidate in enumerate(selected, start=1):
            labels = ";".join(sorted({box.label for box in candidate.boxes}))
            writer.writerow(
                [
                    rank,
                    candidate.group,
                    f"{candidate.score:.6f}",
                    candidate.image_id,
                    candidate.image_path,
                    candidate.xml_path,
                    candidate.width,
                    candidate.height,
                    len(candidate.boxes),
                    labels,
                    candidate.largest_index,
                    candidate.largest.label,
                    candidate.largest.xmin,
                    candidate.largest.ymin,
                    candidate.largest.xmax,
                    candidate.largest.ymax,
                    f"{candidate.largest.xmin / candidate.width:.6f}",
                    f"{candidate.largest.ymin / candidate.height:.6f}",
                    f"{candidate.largest.xmax / candidate.width:.6f}",
                    f"{candidate.largest.ymax / candidate.height:.6f}",
                    candidate.largest.truncated,
                    candidate.largest.difficult,
                    int(candidate.largest.clamped),
                    f"{candidate.area_frac:.6f}",
                    f"{candidate.center_x:.6f}",
                    f"{candidate.center_y:.6f}",
                    f"{candidate.center_dist:.6f}",
                    f"{candidate.max_iou:.6f}",
                    f"{candidate.min_center_dist:.6f}",
                    f"{candidate.border_margin:.6f}",
                    f"{candidate.sharpness:.3f}",
                ]
            )


def clear_previous_overlays(path: Path) -> None:
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for old_path in path.glob(pattern):
            old_path.unlink()


def save_overlay(candidate: Candidate, path: Path, thumb_size: int) -> None:
    with Image.open(candidate.image_path) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box in candidate.boxes:
        is_largest = box == candidate.largest
        color = (255, 60, 40) if is_largest else (255, 210, 0)
        width = 4 if is_largest else 2
        draw.rectangle((box.xmin, box.ymin, box.xmax, box.ymax), outline=color, width=width)
        draw.text((box.xmin + 2, max(0, box.ymin - 12)), box.label, fill=color, font=font)
    caption = f"{candidate.group} | area {candidate.area_frac:.2f} | ov {candidate.max_iou:.2f}"
    image = fit_thumbnail(image, thumb_size)
    canvas = Image.new("RGB", (image.width, image.height + 18), "white")
    canvas.paste(image, (0, 18))
    ImageDraw.Draw(canvas).text((2, 2), caption[:80], fill=(0, 0, 0), font=font)
    canvas.save(path, quality=92)


def fit_thumbnail(image: Image.Image, size: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((size, size), Image.Resampling.LANCZOS)
    return copy


def save_contact_sheet(
    selected: list[Candidate],
    overlay_paths: list[Path],
    path: Path,
    max_cols: int,
) -> None:
    if not overlay_paths:
        return
    tiles = [Image.open(overlay_path).convert("RGB") for overlay_path in overlay_paths]
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    cols = min(max_cols, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        sheet.paste(tile, (x, y))
    sheet.save(path, quality=92)
    for tile in tiles:
        tile.close()


def print_group_summary(selected: list[Candidate]) -> None:
    counts: dict[str, int] = {}
    for candidate in selected:
        counts[candidate.group] = counts.get(candidate.group, 0) + 1
    for group in sorted(counts):
        print(f"{group}: {counts[group]}")


if __name__ == "__main__":
    main()
