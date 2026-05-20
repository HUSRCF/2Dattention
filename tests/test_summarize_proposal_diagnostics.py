from __future__ import annotations

from pathlib import Path

from scripts.summarize_proposal_diagnostics import summarize_proposal_diagnostics


def test_summarize_proposal_diagnostics_merges_recall_and_ap(tmp_path: Path) -> None:
    recall_csv = tmp_path / "recall.csv"
    ap_csv = tmp_path / "ap.csv"
    recall_csv.write_text(
        "\n".join(
            [
                "slice,top_k,iou_threshold,gt_count,recall,mean_best_iou",
                "all,50,0.5,10,0.4,0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ap_csv.write_text(
        "\n".join(
            [
                "slice,images,annotations,ap,ap50,ap75,ap_small,ap_medium,ap_large,ar1,ar10,ar100",
                "all,5,10,0.1,0.2,0.05,0,0,0,0,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = summarize_proposal_diagnostics(
        recall_inputs=[("rpn96", "rpn_recall", recall_csv)],
        ap_inputs=[("rpn96", ap_csv)],
    )

    assert rows[0]["source_type"] == "rpn_recall"
    assert rows[0]["recall"] == "0.4"
    assert rows[1]["source_type"] == "rpn_objectness_ap"
    assert rows[1]["ap50"] == "0.2"
