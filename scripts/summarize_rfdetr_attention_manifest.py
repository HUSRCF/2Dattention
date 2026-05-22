"""Flatten RF-DETR deformable-attention manifest JSON into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-out", type=Path, required=True)
    parser.add_argument("--query-out", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-out",
        type=Path,
        default=None,
        help="Optional JSON summary with correlation and failure-bucket diagnostics.",
    )
    parser.add_argument(
        "--bucket-contact-dir",
        type=Path,
        default=None,
        help="Optional directory for contact sheets grouped by query diagnostic bucket.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    image_rows = [image_summary_row(row) for row in rows]
    query_rows = [query_summary_row(row, query) for row in rows for query in row.get("query_overlays", [])]
    write_csv(args.image_out, image_rows)
    write_csv(args.query_out, query_rows)
    diagnostics = build_diagnostics(image_rows, query_rows)
    if args.diagnostic_out is not None:
        args.diagnostic_out.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostic_out.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    if args.bucket_contact_dir is not None:
        write_bucket_contact_sheets(args.bucket_contact_dir, query_rows)
    print(f"saved_image_attention_csv: {args.image_out}")
    print(f"saved_query_attention_csv: {args.query_out}")
    if args.diagnostic_out is not None:
        print(f"saved_attention_diagnostics: {args.diagnostic_out}")
    if args.bucket_contact_dir is not None:
        print(f"saved_bucket_contact_dir: {args.bucket_contact_dir}")
    print(f"images: {len(image_rows)}")
    print(f"queries: {len(query_rows)}")
    print(f"query_mean_iou: {diagnostics['query_mean_iou']}")
    print(f"query_mean_gt_attention_mass: {diagnostics['query_mean_gt_attention_mass']}")
    print(f"mass_iou_pearson: {diagnostics['mass_iou_pearson']}")
    print(f"mass_iou_spearman: {diagnostics['mass_iou_spearman']}")


def image_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    query_rows = row.get("query_overlays", [])
    query_gt_mass = [float(query["query_gt_attention_mass"]) for query in query_rows]
    return {
        "image_id": row["image_id"],
        "file_name": row["file_name"],
        "gt_boxes": row["gt_boxes"],
        "predictions": row["predictions"],
        "captured_layers": row["captured_layers"],
        "gt_attention_mass": row["gt_attention_mass"],
        "top_pred_attention_mass": row["top_pred_attention_mass"],
        "attention_entropy": row["attention_entropy"],
        "attention_peak_x": row["attention_peak_x"],
        "attention_peak_y": row["attention_peak_y"],
        "query_overlay_count": len(query_rows),
        "query_gt_attention_mass_mean": mean(query_gt_mass),
        "query_gt_attention_mass_min": min(query_gt_mass) if query_gt_mass else "",
        "query_gt_attention_mass_max": max(query_gt_mass) if query_gt_mass else "",
        "query_zero_gt_attention_count": sum(value <= 0 for value in query_gt_mass),
    }


def query_summary_row(row: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    gt_mass = query["query_gt_attention_mass"]
    best_iou = query.get("best_gt_iou", "")
    class_id = query["class_id"]
    best_gt_category_id = query.get("best_gt_category_id", "")
    return {
        "image_id": row["image_id"],
        "file_name": row["file_name"],
        "gt_boxes": row["gt_boxes"],
        "predictions": row["predictions"],
        "rank": query["rank"],
        "query_index": query["query_index"],
        "class_id": class_id,
        "score": query["score"],
        "best_gt_iou": best_iou,
        "best_gt_index": query.get("best_gt_index", ""),
        "best_gt_category_id": best_gt_category_id,
        "nearest_gt_category_match": nearest_gt_category_match(class_id, best_gt_category_id),
        "box_x1": query["box_xyxy"][0],
        "box_y1": query["box_xyxy"][1],
        "box_x2": query["box_xyxy"][2],
        "box_y2": query["box_xyxy"][3],
        "query_gt_attention_mass": gt_mass,
        "diagnostic_bucket": diagnostic_bucket(gt_mass, best_iou),
        "query_top_pred_attention_mass": query["query_top_pred_attention_mass"],
        "query_attention_entropy": query["query_attention_entropy"],
        "query_attention_peak_x": query["query_attention_peak_x"],
        "query_attention_peak_y": query["query_attention_peak_y"],
        "out": query["out"],
    }


def mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return float(sum(values) / len(values))


def diagnostic_bucket(gt_mass: Any, best_iou: Any) -> str:
    if gt_mass == "" or best_iou == "":
        return "missing"
    mass = float(gt_mass)
    iou = float(best_iou)
    if mass <= 0.0:
        return "zero_mass"
    if mass < 0.2 and iou >= 0.5:
        return "low_mass_high_iou"
    if mass >= 0.5 and iou < 0.5:
        return "high_mass_low_iou"
    if mass >= 0.5 and iou >= 0.5:
        return "aligned_hit"
    if mass < 0.5 and iou < 0.5:
        return "aligned_miss"
    return "mixed"


def nearest_gt_category_match(class_id: Any, category_id: Any) -> int | str:
    if class_id == "" or category_id == "":
        return ""
    # RF-DETR labels are zero-based after export; COCO category ids in this repo are one-based.
    return int(int(class_id) + 1 == int(category_id))


def build_diagnostics(image_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (float(row["query_gt_attention_mass"]), float(row["best_gt_iou"]))
        for row in query_rows
        if row.get("query_gt_attention_mass") != "" and row.get("best_gt_iou") != ""
    ]
    masses = [pair[0] for pair in pairs]
    ious = [pair[1] for pair in pairs]
    high_iou = [pair for pair in pairs if pair[1] >= 0.5]
    low_iou = [pair for pair in pairs if pair[1] < 0.5]
    zero_mass = [pair for pair in pairs if pair[0] <= 0.0]
    low_mass_high_iou = [pair for pair in pairs if pair[0] < 0.2 and pair[1] >= 0.5]
    high_mass_low_iou = [pair for pair in pairs if pair[0] >= 0.5 and pair[1] < 0.5]
    category_rows = [row for row in query_rows if row.get("nearest_gt_category_match") != ""]
    category_hits = [int(row["nearest_gt_category_match"]) for row in category_rows]
    iou50_rows = [row for row in category_rows if float(row["best_gt_iou"]) >= 0.5]
    iou50_category_hits = [int(row["nearest_gt_category_match"]) for row in iou50_rows]
    bucket_category_rates = {}
    for bucket in sorted({str(row["diagnostic_bucket"]) for row in category_rows}):
        bucket_rows = [row for row in category_rows if str(row["diagnostic_bucket"]) == bucket]
        bucket_category_rates[bucket] = {
            "count": len(bucket_rows),
            "nearest_gt_category_match_rate": mean([int(row["nearest_gt_category_match"]) for row in bucket_rows]),
        }
    return {
        "images": len(image_rows),
        "queries": len(query_rows),
        "query_pairs_with_iou": len(pairs),
        "query_mean_iou": mean(ious),
        "query_median_iou": median(ious),
        "query_iou50_count": len(high_iou),
        "query_iou50_rate": ratio(len(high_iou), len(pairs)),
        "query_mean_gt_attention_mass": mean(masses),
        "query_median_gt_attention_mass": median(masses),
        "query_zero_gt_attention_mass_count": len(zero_mass),
        "query_zero_gt_attention_mass_rate": ratio(len(zero_mass), len(pairs)),
        "mass_iou_pearson": pearson(masses, ious),
        "mass_iou_spearman": spearman(masses, ious),
        "low_mass_high_iou_count": len(low_mass_high_iou),
        "low_mass_high_iou_rate": ratio(len(low_mass_high_iou), len(pairs)),
        "high_mass_low_iou_count": len(high_mass_low_iou),
        "high_mass_low_iou_rate": ratio(len(high_mass_low_iou), len(pairs)),
        "high_iou_mean_mass": mean([pair[0] for pair in high_iou]),
        "low_iou_mean_mass": mean([pair[0] for pair in low_iou]),
        "nearest_gt_category_match_count": sum(category_hits),
        "nearest_gt_category_match_rate": mean(category_hits),
        "iou50_nearest_gt_category_match_count": sum(iou50_category_hits),
        "iou50_nearest_gt_category_match_rate": mean(iou50_category_hits),
        "bucket_category_match": bucket_category_rates,
    }


def ratio(numerator: int, denominator: int) -> float | str:
    if denominator == 0:
        return ""
    return float(numerator / denominator)


def median(values: list[float]) -> float | str:
    if not values:
        return ""
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)


def pearson(xs: list[float], ys: list[float]) -> float | str:
    if len(xs) < 2 or len(xs) != len(ys):
        return ""
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0:
        return ""
    return float(numerator / denominator)


def spearman(xs: list[float], ys: list[float]) -> float | str:
    if len(xs) < 2 or len(xs) != len(ys):
        return ""
    return pearson(ranks(xs), ranks(ys))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for original_idx, _ in indexed[idx:end]:
            out[original_idx] = avg_rank
        idx = end
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_bucket_contact_sheets(path: Path, rows: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    path.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["diagnostic_bucket"]), []).append(row)
    for bucket, bucket_rows in grouped.items():
        images = []
        for row in bucket_rows:
            image_path = Path(str(row["out"]))
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((360, 270))
            tile = Image.new("RGB", (380, 320), "white")
            tile.paste(image, ((380 - image.width) // 2, 0))
            draw = ImageDraw.Draw(tile)
            draw.text(
                (8, 276),
                f"id={row['image_id']} q={row['query_index']} score={float(row['score']):.3f}",
                fill=(0, 0, 0),
            )
            draw.text(
                (8, 296),
                f"IoU={float(row['best_gt_iou']):.3f} mass={float(row['query_gt_attention_mass']):.3f}",
                fill=(0, 0, 0),
            )
            images.append(tile)
        if images:
            contact = make_contact_sheet(images)
            contact.save(path / f"{bucket}.jpg", quality=92)


def make_contact_sheet(images: list[Any], columns: int = 3) -> Any:
    from PIL import Image

    if not images:
        raise ValueError("Expected at least one image.")
    width, height = images[0].size
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for idx, image in enumerate(images):
        x = (idx % columns) * width
        y = (idx // columns) * height
        sheet.paste(image, (x, y))
    return sheet


if __name__ == "__main__":
    main()
