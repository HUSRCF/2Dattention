"""Summarize torchvision detector smoke CSV and prediction JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("--predictions", nargs="*", type=Path, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("file,rows,final_step,final_loss,final_iou,final_ap50,final_ap50_class,best_ap50,best_iou")
    for path in args.csv_paths:
        summary = summarize_csv(path)
        print(
            f"{path},{summary['rows']},{summary['final_step']},"
            f"{summary['final_loss']:.4f},{summary['final_iou']:.3f},"
            f"{summary['final_ap50']:.3f},{summary['final_ap50_class']:.3f},"
            f"{summary['best_ap50']:.3f},{summary['best_iou']:.3f}"
        )
    if args.predictions:
        print("prediction_file,predictions,images,mean_score,max_score")
        for path in args.predictions:
            summary = summarize_predictions(path)
            print(
                f"{path},{summary['predictions']},{summary['images']},"
                f"{summary['mean_score']:.4f},{summary['max_score']:.4f}"
            )


def summarize_csv(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    final = rows[-1]
    return {
        "rows": len(rows),
        "final_step": int(final["step"]),
        "final_loss": float(final["loss"]),
        "final_iou": float(final["eval_iou"]),
        "final_ap50": float(final["eval_ap50"]),
        "final_ap50_class": float(final["eval_ap50_class"]),
        "best_ap50": max(float(row["eval_ap50"]) for row in rows),
        "best_iou": max(float(row["eval_iou"]) for row in rows),
    }


def summarize_predictions(path: Path) -> dict[str, float | int]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not rows:
        return {"predictions": 0, "images": 0, "mean_score": 0.0, "max_score": 0.0}
    scores = [float(row["score"]) for row in rows]
    return {
        "predictions": len(rows),
        "images": len({int(row["image_id"]) for row in rows}),
        "mean_score": sum(scores) / len(scores),
        "max_score": max(scores),
    }


if __name__ == "__main__":
    sys.exit(main())
