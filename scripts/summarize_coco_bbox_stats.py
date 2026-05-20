"""Summarize COCO bbox scale statistics for detector protocol design."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[128, 384])
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    rows = summarize_bbox_stats(data, resolutions=args.resolutions, small_area_ratio=args.small_area_ratio)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved_bbox_stats: {args.out}")
    for row in rows:
        print(
            f"resolution={row['resolution']} annotations={row['annotations']} "
            f"small_frac={row['small_fraction']:.3f} "
            f"median_short_px={row['median_short_px']:.2f} p10_short_px={row['p10_short_px']:.2f} "
            f"median_area_px={row['median_area_px']:.1f}"
        )


def summarize_bbox_stats(
    data: dict[str, Any],
    resolutions: list[int],
    small_area_ratio: float = 0.05,
) -> list[dict[str, float | int]]:
    images_by_id = {int(image["id"]): image for image in data.get("images", [])}
    base_records = []
    for annotation in data.get("annotations", []):
        image = images_by_id[int(annotation["image_id"])]
        image_width = max(1.0, float(image["width"]))
        image_height = max(1.0, float(image["height"]))
        _, _, bbox_width, bbox_height = [float(value) for value in annotation["bbox"]]
        area_ratio = max(0.0, bbox_width) * max(0.0, bbox_height) / (image_width * image_height)
        base_records.append(
            {
                "width_ratio": max(0.0, bbox_width) / image_width,
                "height_ratio": max(0.0, bbox_height) / image_height,
                "area_ratio": area_ratio,
            }
        )
    return [summarize_at_resolution(base_records, resolution, small_area_ratio) for resolution in resolutions]


def summarize_at_resolution(
    records: list[dict[str, float]],
    resolution: int,
    small_area_ratio: float,
) -> dict[str, float | int]:
    if not records:
        return empty_row(resolution)
    widths = [record["width_ratio"] * resolution for record in records]
    heights = [record["height_ratio"] * resolution for record in records]
    short_sides = [min(width, height) for width, height in zip(widths, heights, strict=True)]
    areas = [record["area_ratio"] * resolution * resolution for record in records]
    area_ratios = [record["area_ratio"] for record in records]
    small_count = sum(area < small_area_ratio for area in area_ratios)
    return {
        "resolution": resolution,
        "annotations": len(records),
        "small_fraction": small_count / len(records),
        "mean_short_px": mean(short_sides),
        "median_short_px": median(short_sides),
        "p10_short_px": percentile(short_sides, 0.10),
        "p25_short_px": percentile(short_sides, 0.25),
        "p75_short_px": percentile(short_sides, 0.75),
        "mean_area_px": mean(areas),
        "median_area_px": median(areas),
        "p10_area_px": percentile(areas, 0.10),
        "p25_area_px": percentile(areas, 0.25),
        "p75_area_px": percentile(areas, 0.75),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def empty_row(resolution: int) -> dict[str, float | int]:
    return {
        "resolution": resolution,
        "annotations": 0,
        "small_fraction": 0.0,
        "mean_short_px": 0.0,
        "median_short_px": 0.0,
        "p10_short_px": 0.0,
        "p25_short_px": 0.0,
        "p75_short_px": 0.0,
        "mean_area_px": 0.0,
        "median_area_px": 0.0,
        "p10_area_px": 0.0,
        "p25_area_px": 0.0,
        "p75_area_px": 0.0,
    }


if __name__ == "__main__":
    main()
