"""Build a candidate-constrained category oracle for COCO predictions.

This diagnostic groups category-expanded predictions by image and bbox. For each
box group, it finds the nearest GT box. If the nearest GT category is present in
the group's candidate categories, the output keeps that category; otherwise the
group is dropped by default. This estimates whether the candidate category set is
itself sufficient, separate from category ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torchvision.ops import box_iou

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rescore_coco_predictions_by_oracle_iou import gt_boxes_by_image, xywh_to_xyxy


SUMMARY_FIELDS = (
    "groups",
    "emitted_predictions",
    "candidate_hits",
    "candidate_hit_rate",
    "mean_nearest_iou",
    "mean_hit_iou",
    "score_mode",
)

PER_CATEGORY_FIELDS = (
    "category_id",
    "category_name",
    "groups",
    "candidate_hits",
    "candidate_hit_rate",
    "mean_nearest_iou",
    "mean_hit_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-per-category", type=Path)
    parser.add_argument("--bbox-decimals", type=int, default=3)
    parser.add_argument("--score-mode", choices=("candidate", "group_max", "oracle_iou"), default="candidate")
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep the group's highest-score original prediction when the nearest GT category is absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, summary = oracle_category_candidates(
        annotation_json=args.annotations,
        prediction_json=args.predictions,
        bbox_decimals=args.bbox_decimals,
        score_mode=args.score_mode,
        keep_missing=args.keep_missing,
        out_per_category=args.out_per_category,
    )
    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.out_predictions.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)
    print(f"saved_candidate_oracle_predictions: {args.out_predictions}")
    print(f"saved_candidate_oracle_summary: {args.out_summary}")
    print(
        f"groups={summary['groups']} emitted={summary['emitted_predictions']} "
        f"hit_rate={summary['candidate_hit_rate']:.4f} mean_iou={summary['mean_nearest_iou']:.4f}"
    )


def oracle_category_candidates(
    *,
    annotation_json: Path,
    prediction_json: Path,
    bbox_decimals: int = 3,
    score_mode: str = "candidate",
    keep_missing: bool = False,
    out_per_category: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | int | str]]:
    annotations = json.loads(annotation_json.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_json.read_text(encoding="utf-8"))
    gt_by_image = gt_boxes_by_image(annotations)
    groups = group_predictions_by_image_box(predictions, bbox_decimals=bbox_decimals)
    category_names = {
        int(row["id"]): str(row.get("name", row["id"]))
        for row in annotations.get("categories", [])
    }

    output: list[dict[str, Any]] = []
    nearest_ious: list[float] = []
    hit_ious: list[float] = []
    candidate_hits = 0
    per_category: dict[int, dict[str, Any]] = {}
    for group in groups:
        nearest = nearest_gt_for_group(group, gt_by_image)
        if nearest is None:
            continue
        target_category, nearest_iou = nearest
        nearest_ious.append(nearest_iou)
        category_stats = per_category.setdefault(
            target_category,
            {
                "category_id": target_category,
                "category_name": category_names.get(target_category, str(target_category)),
                "groups": 0,
                "candidate_hits": 0,
                "nearest_ious": [],
                "hit_ious": [],
            },
        )
        category_stats["groups"] += 1
        category_stats["nearest_ious"].append(nearest_iou)
        target_rows = [row for row in group if int(row["category_id"]) == target_category]
        if target_rows:
            candidate_hits += 1
            hit_ious.append(nearest_iou)
            category_stats["candidate_hits"] += 1
            category_stats["hit_ious"].append(nearest_iou)
            best_target_row = max(target_rows, key=lambda row: float(row.get("score", 0.0)))
            row = dict(best_target_row)
            row["category_id"] = target_category
            row["score"] = oracle_score(
                group,
                best_target_row,
                nearest_iou=nearest_iou,
                score_mode=score_mode,
            )
            output.append(row)
        elif keep_missing:
            output.append(dict(max(group, key=lambda row: float(row.get("score", 0.0)))))

    output.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    total_groups = len(groups)
    summary = {
        "groups": total_groups,
        "emitted_predictions": len(output),
        "candidate_hits": candidate_hits,
        "candidate_hit_rate": 0.0 if total_groups == 0 else candidate_hits / total_groups,
        "mean_nearest_iou": mean(nearest_ious),
        "mean_hit_iou": mean(hit_ious),
        "score_mode": score_mode,
    }
    if out_per_category is not None:
        write_per_category_summary(per_category, out_per_category)
    return output, summary


def group_predictions_by_image_box(
    predictions: list[dict[str, Any]],
    *,
    bbox_decimals: int,
) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for prediction in predictions:
        bbox = tuple(round(float(value), bbox_decimals) for value in prediction["bbox"])
        key = (int(prediction["image_id"]), *bbox)
        groups.setdefault(key, []).append(prediction)
    return list(groups.values())


def nearest_gt_for_group(
    group: list[dict[str, Any]],
    gt_by_image: dict[int, list[dict[str, Any]]],
) -> tuple[int, float] | None:
    if not group:
        return None
    image_id = int(group[0]["image_id"])
    gt_rows = gt_by_image.get(image_id, [])
    if not gt_rows:
        return None
    gt_boxes = torch.tensor([row["box"] for row in gt_rows], dtype=torch.float32).reshape(-1, 4)
    pred_box = torch.tensor([xywh_to_xyxy(group[0]["bbox"])], dtype=torch.float32)
    ious = box_iou(pred_box, gt_boxes).squeeze(0)
    best_idx = int(torch.argmax(ious).item())
    return int(gt_rows[best_idx]["category_id"]), float(ious[best_idx].item())


def oracle_score(
    group: list[dict[str, Any]],
    target_row: dict[str, Any],
    *,
    nearest_iou: float,
    score_mode: str,
) -> float:
    if score_mode == "candidate":
        return float(target_row.get("score", 0.0))
    if score_mode == "group_max":
        return max(float(row.get("score", 0.0)) for row in group)
    if score_mode == "oracle_iou":
        return nearest_iou
    raise ValueError(f"unsupported score_mode: {score_mode}")


def write_per_category_summary(per_category: dict[int, dict[str, Any]], out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for stats in per_category.values():
        groups = int(stats["groups"])
        hits = int(stats["candidate_hits"])
        rows.append(
            {
                "category_id": int(stats["category_id"]),
                "category_name": str(stats["category_name"]),
                "groups": groups,
                "candidate_hits": hits,
                "candidate_hit_rate": 0.0 if groups == 0 else hits / groups,
                "mean_nearest_iou": mean(stats["nearest_ious"]),
                "mean_hit_iou": mean(stats["hit_ious"]),
            }
        )
    rows.sort(key=lambda row: (float(row["candidate_hit_rate"]), -int(row["groups"]), int(row["category_id"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PER_CATEGORY_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


if __name__ == "__main__":
    main()
