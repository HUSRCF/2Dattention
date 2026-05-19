"""Run the external-detector COCOeval stage-gate on exported predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_coco_predictions import METRIC_NAMES, evaluate_coco_predictions
from scripts.evaluate_coco_slices import FIELDNAMES as SLICE_FIELDNAMES
from scripts.evaluate_coco_slices import evaluate_coco_slices
from scripts.summarize_torchvision_detector_results import summarize_csv, summarize_predictions


SUMMARY_FIELDS = (
    "name",
    "runner_final_step",
    "runner_final_iou",
    "runner_final_ap50",
    "runner_final_ap50_class",
    "predictions",
    "prediction_images",
    "mean_score",
    "max_score",
    *[f"coco_{name}" for name in METRIC_NAMES],
    "offcenter_ap50",
    "center_ap50",
    "small_ap50",
    "medium_ap50",
    "large_ap50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--runner-csv", type=Path, default=None)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--center-radius", type=float, default=0.25)
    parser.add_argument("--small-area-ratio", type=float, default=0.05)
    parser.add_argument("--large-area-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = evaluate_external_detector_protocol(
        annotations=args.annotations,
        predictions=args.predictions,
        out_prefix=args.out_prefix,
        runner_csv=args.runner_csv,
        name=args.name,
        center_radius=args.center_radius,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    print(f"saved_cocoeval: {artifacts['cocoeval_csv']}")
    print(f"saved_slices: {artifacts['slices_csv']}")
    print(f"saved_summary: {artifacts['summary_csv']}")


def evaluate_external_detector_protocol(
    annotations: Path,
    predictions: Path,
    out_prefix: Path,
    runner_csv: Path | None = None,
    name: str | None = None,
    center_radius: float = 0.25,
    small_area_ratio: float = 0.05,
    large_area_ratio: float = 0.25,
) -> dict[str, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cocoeval_csv = out_prefix.with_name(f"{out_prefix.name}_cocoeval.csv")
    slices_csv = out_prefix.with_name(f"{out_prefix.name}_slices.csv")
    summary_csv = out_prefix.with_name(f"{out_prefix.name}_summary.csv")

    coco_metrics = evaluate_coco_predictions(annotations, predictions)
    write_rows(cocoeval_csv, list(METRIC_NAMES), [coco_metrics])

    slice_rows = evaluate_coco_slices(
        annotations,
        predictions,
        center_radius=center_radius,
        small_area_ratio=small_area_ratio,
        large_area_ratio=large_area_ratio,
    )
    write_rows(slices_csv, list(SLICE_FIELDNAMES), slice_rows)

    summary = build_protocol_summary(
        name=name or out_prefix.name,
        predictions=predictions,
        coco_metrics=coco_metrics,
        slice_rows=slice_rows,
        runner_csv=runner_csv,
    )
    write_rows(summary_csv, list(SUMMARY_FIELDS), [summary])
    return {"cocoeval_csv": cocoeval_csv, "slices_csv": slices_csv, "summary_csv": summary_csv}


def build_protocol_summary(
    name: str,
    predictions: Path,
    coco_metrics: dict[str, float],
    slice_rows: list[dict[str, Any]],
    runner_csv: Path | None = None,
) -> dict[str, Any]:
    prediction_summary = summarize_predictions(predictions)
    runner_summary = summarize_csv(runner_csv) if runner_csv is not None else {}
    row: dict[str, Any] = {
        "name": name,
        "runner_final_step": runner_summary.get("final_step", ""),
        "runner_final_iou": runner_summary.get("final_iou", ""),
        "runner_final_ap50": runner_summary.get("final_ap50", ""),
        "runner_final_ap50_class": runner_summary.get("final_ap50_class", ""),
        "predictions": prediction_summary["predictions"],
        "prediction_images": prediction_summary["images"],
        "mean_score": prediction_summary["mean_score"],
        "max_score": prediction_summary["max_score"],
        **{f"coco_{metric_name}": coco_metrics[metric_name] for metric_name in METRIC_NAMES},
    }
    rows_by_slice = {str(slice_row["slice"]): slice_row for slice_row in slice_rows}
    for slice_name in ("offcenter", "center", "small", "medium", "large"):
        row[f"{slice_name}_ap50"] = rows_by_slice.get(slice_name, {}).get("ap50", "")
    return row


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
