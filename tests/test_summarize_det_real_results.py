from __future__ import annotations

from pathlib import Path

from scripts.summarize_det_real_results import (
    load_final_rows,
    paired_delta,
    render_markdown,
    summarize_by_model,
)


def test_summarize_det_real_uses_last_step_per_seed(tmp_path: Path) -> None:
    csv_path = tmp_path / "det.csv"
    csv_path.write_text(
        "\n".join(
            [
                "model,run_seed,step,eval_iou,eval_ap50",
                "base,1,100,0.10,0.20",
                "base,1,200,0.30,0.40",
                "base,2,200,0.50,0.60",
                "quality,1,200,0.40,0.70",
                "quality,2,200,0.70,0.50",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_final_rows(csv_path)
    assert len(rows) == 4

    summaries = summarize_by_model(rows, ("eval_iou", "eval_ap50"))
    assert round(summaries["base"]["eval_iou"].mean, 3) == 0.400
    assert round(summaries["quality"]["eval_iou"].mean, 3) == 0.550

    deltas = paired_delta(rows, "base", ("eval_iou", "eval_ap50"))
    assert deltas["quality"]["eval_iou"] == (0.15, 2, 2)
    assert round(deltas["quality"]["eval_ap50"][0], 3) == 0.100
    assert deltas["quality"]["eval_ap50"][1:] == (1, 2)

    markdown = render_markdown(summaries, ("eval_iou", "eval_ap50"), "base", deltas)
    assert "`quality`" in markdown
    assert "+0.150 (2/2)" in markdown
