"""Fit a small held-out score calibrator for COCO prediction JSONs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_coco_score_iou_correlation import pearson, spearman, xyxy_iou
from scripts.rescore_coco_predictions_by_oracle_iou import gt_boxes_by_image, xywh_to_xyxy


SUMMARY_FIELDNAMES = (
    "train_annotations",
    "apply_annotations",
    "prediction_json",
    "train_predictions",
    "apply_predictions",
    "class_aware",
    "score_mode",
    "quality_alpha",
    "category_weight",
    "category_smoothing",
    "ridge",
    "train_target_mean",
    "train_pred_quality_mean",
    "train_pearson_quality_iou",
    "train_spearman_quality_iou",
    "apply_original_score_mean",
    "apply_calibrated_score_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--apply-annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--class-aware", action="store_true")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--quality-alpha", type=float, default=1.0)
    parser.add_argument(
        "--category-weight",
        type=float,
        default=0.5,
        help="Blend linear quality with per-category calibration quality.",
    )
    parser.add_argument(
        "--category-smoothing",
        type=float,
        default=5.0,
        help="Pseudo-count smoothing toward the global target mean for category quality.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("multiply", "replace"),
        default="multiply",
        help="Use predicted quality as a multiplier or as the full new score.",
    )
    parser.add_argument(
        "--include-all-images",
        action="store_true",
        help="Write all prediction images; otherwise write only images in --apply-annotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rescored, summary = calibrate_prediction_scores(
        train_annotation_json=args.train_annotations,
        apply_annotation_json=args.apply_annotations,
        prediction_json=args.predictions,
        class_aware=args.class_aware,
        ridge=args.ridge,
        score_mode=args.score_mode,
        quality_alpha=args.quality_alpha,
        category_weight=args.category_weight,
        category_smoothing=args.category_smoothing,
        include_all_images=args.include_all_images,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rescored), encoding="utf-8")
    print(f"saved_calibrated_predictions: {args.out}")
    print(
        f"train={summary['train_predictions']} apply={summary['apply_predictions']} "
        f"quality_corr={summary['train_spearman_quality_iou']:.4f}"
    )
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDNAMES), lineterminator="\n")
            writer.writeheader()
            writer.writerow(summary)
        print(f"saved_calibration_summary: {args.summary_out}")


def calibrate_prediction_scores(
    train_annotation_json: Path,
    apply_annotation_json: Path,
    prediction_json: Path,
    *,
    class_aware: bool = False,
    ridge: float = 1e-3,
    score_mode: str = "multiply",
    quality_alpha: float = 1.0,
    category_weight: float = 0.5,
    category_smoothing: float = 5.0,
    include_all_images: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float | int | str | bool]]:
    train_annotations = json.loads(train_annotation_json.read_text(encoding="utf-8"))
    apply_annotations = json.loads(apply_annotation_json.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_json.read_text(encoding="utf-8"))
    train_image_ids = image_ids(train_annotations)
    apply_image_ids = image_ids(apply_annotations)
    image_sizes = {**image_size_map(train_annotations), **image_size_map(apply_annotations)}
    train_gt_by_image = gt_boxes_by_image(train_annotations)
    rank_features = prediction_rank_features(predictions)

    train_indices = [index for index, row in enumerate(predictions) if int(row["image_id"]) in train_image_ids]
    train_predictions = [predictions[index] for index in train_indices]
    if not train_predictions:
        raise ValueError("No predictions overlap --train-annotations images")

    x_train = np.asarray(
        [
            prediction_features(row, image_sizes, rank_features=rank_features[index])
            for index, row in zip(train_indices, train_predictions, strict=True)
        ],
        dtype=np.float64,
    )
    y_train = np.asarray(
        [
            nearest_iou_target(row, train_gt_by_image, class_aware=class_aware)
            for row in train_predictions
        ],
        dtype=np.float64,
    )
    train_categories = [int(row["category_id"]) for row in train_predictions]
    model = fit_ridge_quality_model(
        x_train,
        y_train,
        train_categories,
        ridge=ridge,
        category_weight=category_weight,
        category_smoothing=category_smoothing,
    )
    train_quality = predict_quality(model, x_train, train_categories)

    output_rows = []
    apply_original_scores = []
    apply_calibrated_scores = []
    for index, prediction in enumerate(predictions):
        in_apply = int(prediction["image_id"]) in apply_image_ids
        if not include_all_images and not in_apply:
            continue
        row = dict(prediction)
        quality = float(
            predict_quality(
                model,
                np.asarray([prediction_features(row, image_sizes, rank_features=rank_features[index])], dtype=np.float64),
                [int(row["category_id"])],
            )[0]
        )
        original_score = float(row.get("score", 0.0))
        calibrated = calibrated_score(original_score, quality, score_mode=score_mode, quality_alpha=quality_alpha)
        row["score"] = calibrated
        row["calibrated_quality"] = quality
        row["original_score"] = original_score
        output_rows.append(row)
        if in_apply:
            apply_original_scores.append(original_score)
            apply_calibrated_scores.append(calibrated)

    summary: dict[str, float | int | str | bool] = {
        "train_annotations": str(train_annotation_json),
        "apply_annotations": str(apply_annotation_json),
        "prediction_json": str(prediction_json),
        "train_predictions": len(train_predictions),
        "apply_predictions": len([row for row in output_rows if int(row["image_id"]) in apply_image_ids]),
        "class_aware": class_aware,
        "score_mode": score_mode,
        "quality_alpha": quality_alpha,
        "category_weight": category_weight,
        "category_smoothing": category_smoothing,
        "ridge": ridge,
        "train_target_mean": safe_mean(y_train.tolist()),
        "train_pred_quality_mean": safe_mean(train_quality.tolist()),
        "train_pearson_quality_iou": pearson(train_quality.tolist(), y_train.tolist()),
        "train_spearman_quality_iou": spearman(train_quality.tolist(), y_train.tolist()),
        "apply_original_score_mean": safe_mean(apply_original_scores),
        "apply_calibrated_score_mean": safe_mean(apply_calibrated_scores),
    }
    return output_rows, summary


def image_ids(annotations: dict[str, Any]) -> set[int]:
    return {int(image["id"]) for image in annotations.get("images", [])}


def image_size_map(annotations: dict[str, Any]) -> dict[int, tuple[float, float]]:
    sizes = {}
    for image in annotations.get("images", []):
        width = float(image.get("width", 1.0) or 1.0)
        height = float(image.get("height", 1.0) or 1.0)
        sizes[int(image["id"])] = (max(width, 1.0), max(height, 1.0))
    return sizes


def prediction_features(
    prediction: dict[str, Any],
    image_sizes: dict[int, tuple[float, float]],
    *,
    rank_features: tuple[float, float] | None = None,
) -> list[float]:
    image_id = int(prediction["image_id"])
    image_width, image_height = image_sizes.get(image_id, (1.0, 1.0))
    x, y, width, height = [float(value) for value in prediction["bbox"]]
    width_norm = max(width, 0.0) / image_width
    height_norm = max(height, 0.0) / image_height
    cx = (x + 0.5 * width) / image_width
    cy = (y + 0.5 * height) / image_height
    area = max(width_norm * height_norm, 0.0)
    center_dx = abs(cx - 0.5)
    center_dy = abs(cy - 0.5)
    score = clip(float(prediction.get("score", 0.0)), 1e-6, 1.0 - 1e-6)
    aspect = math.log(max(width_norm, 1e-6) / max(height_norm, 1e-6))
    image_rank, category_rank = rank_features if rank_features is not None else (0.0, 0.0)
    return [
        1.0,
        score,
        math.log(score / (1.0 - score)),
        width_norm,
        height_norm,
        area,
        math.sqrt(area),
        cx,
        cy,
        center_dx,
        center_dy,
        math.sqrt(center_dx * center_dx + center_dy * center_dy),
        aspect,
        image_rank,
        category_rank,
    ]


def prediction_rank_features(predictions: list[dict[str, Any]]) -> list[tuple[float, float]]:
    image_groups: dict[int, list[int]] = {}
    image_category_groups: dict[tuple[int, int], list[int]] = {}
    for index, prediction in enumerate(predictions):
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        image_groups.setdefault(image_id, []).append(index)
        image_category_groups.setdefault((image_id, category_id), []).append(index)
    image_ranks = fractional_score_ranks(predictions, image_groups)
    category_ranks = fractional_score_ranks(predictions, image_category_groups)
    return list(zip(image_ranks, category_ranks, strict=True))


def fractional_score_ranks(
    predictions: list[dict[str, Any]],
    groups: dict[Any, list[int]],
) -> list[float]:
    ranks_out = [0.0] * len(predictions)
    for indices in groups.values():
        sorted_indices = sorted(indices, key=lambda index: float(predictions[index].get("score", 0.0)), reverse=True)
        denom = max(1, len(sorted_indices) - 1)
        for rank, index in enumerate(sorted_indices):
            ranks_out[index] = rank / denom
    return ranks_out


def nearest_iou_target(
    prediction: dict[str, Any],
    gt_by_image: dict[int, list[dict[str, Any]]],
    *,
    class_aware: bool,
) -> float:
    image_id = int(prediction["image_id"])
    category_id = int(prediction["category_id"])
    gt_rows = gt_by_image.get(image_id, [])
    if class_aware:
        gt_rows = [row for row in gt_rows if int(row["category_id"]) == category_id]
    pred_box = xywh_to_xyxy([float(value) for value in prediction["bbox"]])
    return max((xyxy_iou(pred_box, row["box"]) for row in gt_rows), default=0.0)


def fit_ridge_quality_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    categories: list[int],
    *,
    ridge: float,
    category_weight: float,
    category_smoothing: float,
) -> dict[str, Any]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std = np.where(std < 1e-6, 1.0, std)
    x_scaled = (x_train - mean) / std
    penalty = np.eye(x_scaled.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_scaled.T @ x_scaled + penalty, x_scaled.T @ y_train)
    global_mean = float(y_train.mean()) if y_train.size else 0.0
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for category, target in zip(categories, y_train.tolist(), strict=True):
        sums[category] = sums.get(category, 0.0) + float(target)
        counts[category] = counts.get(category, 0) + 1
    category_quality = {
        category: (sums[category] + category_smoothing * global_mean) / (counts[category] + category_smoothing)
        for category in sums
    }
    return {
        "mean": mean,
        "std": std,
        "beta": beta,
        "category_quality": category_quality,
        "global_quality": global_mean,
        "category_weight": clip(float(category_weight), 0.0, 1.0),
    }


def predict_quality(model: dict[str, Any], features: np.ndarray, categories: list[int] | None = None) -> np.ndarray:
    x_scaled = (features - model["mean"]) / model["std"]
    linear_quality = np.clip(x_scaled @ model["beta"], 0.0, 1.0)
    if categories is None:
        return linear_quality
    category_weight = float(model["category_weight"])
    if category_weight <= 0.0:
        return linear_quality
    category_quality_map = model["category_quality"]
    global_quality = float(model["global_quality"])
    category_quality = np.asarray(
        [float(category_quality_map.get(category, global_quality)) for category in categories],
        dtype=np.float64,
    )
    return np.clip((1.0 - category_weight) * linear_quality + category_weight * category_quality, 0.0, 1.0)


def calibrated_score(
    original_score: float,
    quality: float,
    *,
    score_mode: str,
    quality_alpha: float,
) -> float:
    quality = clip(quality, 0.0, 1.0)
    if score_mode == "replace":
        return quality
    if score_mode != "multiply":
        raise ValueError(f"Unsupported score mode: {score_mode}")
    return clip(original_score, 0.0, 1.0) * (quality**quality_alpha)


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
