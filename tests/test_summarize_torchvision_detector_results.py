from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_torchvision_detector_results import summarize_csv, summarize_predictions


def test_summarize_torchvision_detector_csv(tmp_path: Path) -> None:
    path = tmp_path / "run.csv"
    path.write_text(
        "\n".join(
            [
                "step,loss,eval_iou,eval_ap50,eval_ap50_class",
                "0,0.0000,0.100,0.010,0.000",
                "1,2.0000,0.200,0.005,0.002",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_csv(path)

    assert summary["rows"] == 2
    assert summary["final_step"] == 1
    assert summary["final_loss"] == 2.0
    assert summary["final_iou"] == 0.2
    assert summary["final_ap50"] == 0.005
    assert summary["best_ap50"] == 0.01
    assert summary["best_iou"] == 0.2


def test_summarize_torchvision_detector_predictions(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 2, "bbox": [1, 2, 3, 4], "score": 0.25},
                {"image_id": 1, "category_id": 3, "bbox": [2, 3, 4, 5], "score": 0.75},
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_predictions(path)

    assert summary == {"predictions": 2, "images": 1, "mean_score": 0.5, "max_score": 0.75}


def test_summarize_torchvision_detector_empty_predictions(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text("[]", encoding="utf-8")

    summary = summarize_predictions(path)

    assert summary == {"predictions": 0, "images": 0, "mean_score": 0.0, "max_score": 0.0}
