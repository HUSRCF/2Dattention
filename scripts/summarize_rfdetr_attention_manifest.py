"""Flatten RF-DETR deformable-attention manifest JSON into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-out", type=Path, required=True)
    parser.add_argument("--query-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    image_rows = [image_summary_row(row) for row in rows]
    query_rows = [query_summary_row(row, query) for row in rows for query in row.get("query_overlays", [])]
    write_csv(args.image_out, image_rows)
    write_csv(args.query_out, query_rows)
    print(f"saved_image_attention_csv: {args.image_out}")
    print(f"saved_query_attention_csv: {args.query_out}")
    print(f"images: {len(image_rows)}")
    print(f"queries: {len(query_rows)}")


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
    return {
        "image_id": row["image_id"],
        "file_name": row["file_name"],
        "gt_boxes": row["gt_boxes"],
        "predictions": row["predictions"],
        "rank": query["rank"],
        "query_index": query["query_index"],
        "class_id": query["class_id"],
        "score": query["score"],
        "best_gt_iou": query.get("best_gt_iou", ""),
        "best_gt_index": query.get("best_gt_index", ""),
        "best_gt_category_id": query.get("best_gt_category_id", ""),
        "box_x1": query["box_xyxy"][0],
        "box_y1": query["box_xyxy"][1],
        "box_x2": query["box_xyxy"][2],
        "box_y2": query["box_xyxy"][3],
        "query_gt_attention_mass": query["query_gt_attention_mass"],
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
