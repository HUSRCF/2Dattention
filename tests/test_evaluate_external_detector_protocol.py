from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_external_detector_protocol import evaluate_external_detector_protocol


pytest.importorskip("pycocotools")


def test_evaluate_external_detector_protocol_writes_stage_gate_artifacts(tmp_path: Path) -> None:
    annotations = tmp_path / "ann.json"
    predictions = tmp_path / "pred.json"
    runner_csv = tmp_path / "runner.csv"
    out_prefix = tmp_path / "stage_gate"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "center.JPEG", "width": 100, "height": 100},
                    {"id": 2, "file_name": "offcenter.JPEG", "width": 100, "height": 100},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [40, 40, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    },
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 1, "bbox": [40, 40, 20, 20], "score": 0.99},
                {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.98},
            ]
        ),
        encoding="utf-8",
    )
    runner_csv.write_text(
        "\n".join(
            [
                "step,loss,eval_iou,eval_ap50,eval_ap50_class",
                "0,0.0000,0.100,0.010,0.000",
                "1,2.0000,0.200,0.050,0.020",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = evaluate_external_detector_protocol(
        annotations=annotations,
        predictions=predictions,
        runner_csv=runner_csv,
        out_prefix=out_prefix,
        name="unit",
    )

    assert artifacts["cocoeval_csv"].exists()
    assert artifacts["slices_csv"].exists()
    assert artifacts["summary_csv"].exists()
    summary = list(csv.DictReader(artifacts["summary_csv"].open(encoding="utf-8")))[0]
    assert summary["name"] == "unit"
    assert summary["runner_final_step"] == "1"
    assert float(summary["coco_ap50"]) > 0.99
    assert float(summary["offcenter_ap50"]) > 0.99
