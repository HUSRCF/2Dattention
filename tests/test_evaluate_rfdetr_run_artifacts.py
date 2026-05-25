from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts import evaluate_rfdetr_run_artifacts as module


def read_one_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    return rows[0]


def test_output_paths_use_standard_suffixes(tmp_path: Path) -> None:
    paths = module.output_paths(tmp_path / "run")

    assert paths["class_cocoeval"] == tmp_path / "run_test_cocoeval.csv"
    assert paths["loc_cocoeval"] == tmp_path / "run_test_loc_cocoeval.csv"
    assert paths["candidate_oracle_cocoeval"] == tmp_path / "run_candidate_oracle_groupmax_cocoeval.csv"


def test_output_paths_reflect_candidate_score_mode(tmp_path: Path) -> None:
    paths = module.output_paths(tmp_path / "run", candidate_score_mode="oracle_iou")

    assert paths["candidate_oracle_cocoeval"] == tmp_path / "run_candidate_oracle_oracleiou_cocoeval.csv"


def test_evaluate_run_artifacts_writes_standard_bundle(monkeypatch: Any, tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    predictions = tmp_path / "predictions.json"
    annotations.write_text("{}", encoding="utf-8")
    predictions.write_text("[]", encoding="utf-8")

    def fake_evaluate_coco_predictions(annotation_json: Path, prediction_json: Path, class_agnostic: bool = False) -> dict[str, float]:
        return {name: (0.5 if class_agnostic else 0.25) for name in module.METRIC_NAMES}

    def fake_evaluate_coco_slices(**_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "slice": "all",
                "images": 1,
                "annotations": 1,
                **{name: 0.1 for name in module.METRIC_NAMES},
            }
        ]

    def fake_analyze_score_iou(_annotation_json: Path, _prediction_json: Path, *, class_aware: bool = False) -> dict[str, float | int]:
        return {
            "predictions": 2,
            "images": 1,
            "mean_score": 0.7,
            "mean_nearest_iou": 0.6,
            "pearson_score_iou": 0.2,
            "spearman_score_iou": 0.3 if class_aware else 0.4,
            "top10_mean_iou": 0.8,
            "top50_mean_iou": 0.7,
            "top100_mean_iou": 0.6,
        }

    def fake_coverage(**_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "slice": "all",
                "top_k": 100,
                "iou_threshold": 0.5,
                "gt_count": 1,
                "global_topk_loc_recall": 1.0,
                "global_topk_class_recall": 0.5,
                "per_category_topk_class_recall": 0.5,
                "global_category_retention": 0.5,
                "ranking_gap": 0.0,
                "mean_global_loc_best_iou": 0.9,
                "mean_global_class_best_iou": 0.8,
                "mean_per_category_class_best_iou": 0.8,
            }
        ]

    def fake_per_category(**_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "category_id": 1,
                "category_name": "n00000001",
                "top_k": 100,
                "iou_threshold": 0.5,
                "gt_count": 1,
                "global_topk_loc_recall": 1.0,
                "global_topk_class_recall": 0.5,
                "per_category_topk_class_recall": 0.5,
                "global_category_retention": 0.5,
                "loc_to_class_gap": 0.5,
                "ranking_gap": 0.0,
                "mean_global_loc_best_iou": 0.9,
                "mean_global_class_best_iou": 0.8,
                "mean_per_category_class_best_iou": 0.8,
            }
        ]

    def fake_oracle_category_candidates(
        *,
        annotation_json: Path,
        prediction_json: Path,
        bbox_decimals: int,
        score_mode: str,
        out_per_category: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assert bbox_decimals == 3
        assert score_mode == "group_max"
        out_per_category.parent.mkdir(parents=True, exist_ok=True)
        with out_per_category.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(module.CANDIDATE_PER_CATEGORY_FIELDS))
            writer.writeheader()
            writer.writerow(
                {
                    "category_id": 1,
                    "category_name": "n00000001",
                    "groups": 1,
                    "candidate_hits": 1,
                    "candidate_hit_rate": 1.0,
                    "mean_nearest_iou": 0.9,
                    "mean_hit_iou": 0.9,
                }
            )
        return (
            [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.9}],
            {
                "groups": 1,
                "emitted_predictions": 1,
                "candidate_hits": 1,
                "candidate_hit_rate": 1.0,
                "mean_nearest_iou": 0.9,
                "mean_hit_iou": 0.9,
                "score_mode": "group_max",
            },
        )

    monkeypatch.setattr(module, "evaluate_coco_predictions", fake_evaluate_coco_predictions)
    monkeypatch.setattr(module, "evaluate_coco_slices", fake_evaluate_coco_slices)
    monkeypatch.setattr(module, "analyze_score_iou", fake_analyze_score_iou)
    monkeypatch.setattr(module, "analyze_category_coverage_gap", fake_coverage)
    monkeypatch.setattr(module, "analyze_per_category_coverage_gap", fake_per_category)
    monkeypatch.setattr(module, "oracle_category_candidates", fake_oracle_category_candidates)

    outputs = module.evaluate_run_artifacts(
        annotation_json=annotations,
        prediction_json=predictions,
        out_prefix=tmp_path / "eval" / "run",
    )

    assert set(outputs) == set(module.output_paths(tmp_path / "eval" / "run"))
    assert read_one_row(outputs["class_cocoeval"])["ap50"] == "0.25"
    assert read_one_row(outputs["loc_cocoeval"])["ap50"] == "0.5"
    assert read_one_row(outputs["score_iou_classaware"])["spearman_score_iou"] == "0.3"
    assert read_one_row(outputs["candidate_oracle_summary"])["candidate_hit_rate"] == "1.0"
    assert json.loads(outputs["candidate_oracle_predictions"].read_text(encoding="utf-8"))[0]["score"] == 0.9
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["annotations"] == str(annotations)
    assert manifest["predictions"] == str(predictions)
    assert manifest["candidate_score_mode"] == "group_max"
    assert len(manifest["annotations_sha256"]) == 64
    assert len(manifest["predictions_sha256"]) == 64
