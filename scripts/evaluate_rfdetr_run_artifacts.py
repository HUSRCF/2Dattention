"""Generate the standard RF-DETR post-run evaluation artifact bundle."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_coco_category_coverage_gap import FIELDNAMES as COVERAGE_FIELDNAMES
from scripts.analyze_coco_category_coverage_gap import PER_CATEGORY_FIELDNAMES
from scripts.analyze_coco_category_coverage_gap import analyze_category_coverage_gap
from scripts.analyze_coco_category_coverage_gap import analyze_per_category_coverage_gap
from scripts.analyze_coco_score_iou_correlation import FIELDNAMES as SCORE_IOU_FIELDNAMES
from scripts.analyze_coco_score_iou_correlation import analyze_score_iou
from scripts.evaluate_coco_predictions import METRIC_NAMES
from scripts.evaluate_coco_predictions import evaluate_coco_predictions
from scripts.evaluate_coco_slices import FIELDNAMES as SLICE_FIELDNAMES
from scripts.evaluate_coco_slices import evaluate_coco_slices
from scripts.oracle_coco_category_candidates import PER_CATEGORY_FIELDS as CANDIDATE_PER_CATEGORY_FIELDS
from scripts.oracle_coco_category_candidates import SUMMARY_FIELDS as CANDIDATE_SUMMARY_FIELDS
from scripts.oracle_coco_category_candidates import oracle_category_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help=(
            "Output prefix without suffix, for example "
            "results/run_name. The script writes *_test_cocoeval.csv, etc."
        ),
    )
    parser.add_argument(
        "--candidate-score-mode",
        choices=("candidate", "group_max", "oracle_iou"),
        default="group_max",
    )
    parser.add_argument("--bbox-decimals", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = evaluate_run_artifacts(
        annotation_json=args.annotations,
        prediction_json=args.predictions,
        out_prefix=args.out_prefix,
        candidate_score_mode=args.candidate_score_mode,
        bbox_decimals=args.bbox_decimals,
    )
    print(f"saved_rfdetr_run_artifacts_prefix: {args.out_prefix}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


def evaluate_run_artifacts(
    *,
    annotation_json: Path,
    prediction_json: Path,
    out_prefix: Path,
    candidate_score_mode: str = "group_max",
    bbox_decimals: int = 3,
) -> dict[str, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(out_prefix)

    write_single_row(
        outputs["class_cocoeval"],
        METRIC_NAMES,
        evaluate_coco_predictions(annotation_json, prediction_json, class_agnostic=False),
    )
    write_single_row(
        outputs["loc_cocoeval"],
        METRIC_NAMES,
        evaluate_coco_predictions(annotation_json, prediction_json, class_agnostic=True),
    )
    write_rows(
        outputs["slices"],
        SLICE_FIELDNAMES,
        evaluate_coco_slices(annotation_json=annotation_json, prediction_json=prediction_json),
    )
    write_single_row(
        outputs["score_iou_loc"],
        SCORE_IOU_FIELDNAMES,
        analyze_score_iou(annotation_json, prediction_json, class_aware=False),
    )
    write_single_row(
        outputs["score_iou_classaware"],
        SCORE_IOU_FIELDNAMES,
        analyze_score_iou(annotation_json, prediction_json, class_aware=True),
    )
    write_rows(
        outputs["category_coverage"],
        COVERAGE_FIELDNAMES,
        analyze_category_coverage_gap(annotation_json=annotation_json, prediction_json=prediction_json),
    )
    write_rows(
        outputs["per_category_coverage"],
        PER_CATEGORY_FIELDNAMES,
        analyze_per_category_coverage_gap(annotation_json=annotation_json, prediction_json=prediction_json),
    )
    candidate_predictions, candidate_summary = oracle_category_candidates(
        annotation_json=annotation_json,
        prediction_json=prediction_json,
        bbox_decimals=bbox_decimals,
        score_mode=candidate_score_mode,
        out_per_category=outputs["candidate_oracle_per_category"],
    )
    outputs["candidate_oracle_predictions"].write_text(
        json.dumps(candidate_predictions, indent=2) + "\n",
        encoding="utf-8",
    )
    write_single_row(
        outputs["candidate_oracle_summary"],
        CANDIDATE_SUMMARY_FIELDS,
        candidate_summary,
    )
    write_single_row(
        outputs["candidate_oracle_cocoeval"],
        METRIC_NAMES,
        evaluate_coco_predictions(annotation_json, outputs["candidate_oracle_predictions"], class_agnostic=False),
    )
    return outputs


def output_paths(out_prefix: Path) -> dict[str, Path]:
    stem = str(out_prefix)
    return {
        "class_cocoeval": Path(f"{stem}_test_cocoeval.csv"),
        "loc_cocoeval": Path(f"{stem}_test_loc_cocoeval.csv"),
        "slices": Path(f"{stem}_test_slices.csv"),
        "score_iou_loc": Path(f"{stem}_score_iou_loc.csv"),
        "score_iou_classaware": Path(f"{stem}_score_iou_classaware.csv"),
        "category_coverage": Path(f"{stem}_category_coverage_gap.csv"),
        "per_category_coverage": Path(f"{stem}_per_category_coverage_gap.csv"),
        "candidate_oracle_predictions": Path(f"{stem}_candidate_oracle_predictions.json"),
        "candidate_oracle_summary": Path(f"{stem}_candidate_oracle_summary.csv"),
        "candidate_oracle_per_category": Path(f"{stem}_candidate_oracle_per_category.csv"),
        "candidate_oracle_cocoeval": Path(f"{stem}_candidate_oracle_groupmax_cocoeval.csv"),
    }


def write_single_row(path: Path, fieldnames: tuple[str, ...], row: dict[str, Any]) -> None:
    write_rows(path, fieldnames, [row])


def write_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
