from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


DEFAULT_METRICS = (
    "eval_iou",
    "eval_ap50",
    "eval_ap50_class",
    "eval_ap75",
    "eval_ap50_q2",
    "eval_ap50_q_fixed",
    "eval_ap50_q_fixed_alpha",
    "eval_ap75_q_fixed",
    "eval_small_ap50_q_fixed",
    "eval_medium_ap50_q_fixed",
    "eval_large_ap50_q_fixed",
    "eval_center_ap50_q_fixed",
    "eval_offcenter_ap50_q_fixed",
    "objectness_ece50",
    "objectness_ece75",
    "combined_ece50",
    "combined_ece75",
)


@dataclass(frozen=True)
class SummaryCell:
    mean: float
    std: float
    count: int


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_final_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load the last-step row for every model/seed pair."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        model = row.get("model", "")
        seed = row.get("run_seed", "")
        step = parse_float(row.get("step")) or 0.0
        key = (model, seed)
        previous = latest.get(key)
        previous_step = parse_float(previous.get("step")) if previous else None
        if previous is None or previous_step is None or step >= previous_step:
            latest[key] = row

    return sorted(latest.values(), key=lambda item: (item.get("model", ""), item.get("run_seed", "")))


def summarize_by_model(rows: list[dict[str, str]], metrics: tuple[str, ...]) -> dict[str, dict[str, SummaryCell]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model = row.get("model", "")
        for metric in metrics:
            value = parse_float(row.get(metric))
            if value is not None:
                grouped[model][metric].append(value)

    summaries: dict[str, dict[str, SummaryCell]] = {}
    for model, metric_values in grouped.items():
        summaries[model] = {}
        for metric, values in metric_values.items():
            summaries[model][metric] = SummaryCell(
                mean=mean(values),
                std=stdev(values) if len(values) > 1 else 0.0,
                count=len(values),
            )
    return summaries


def paired_delta(
    rows: list[dict[str, str]], reference_model: str, metrics: tuple[str, ...]
) -> dict[str, dict[str, tuple[float, int, int]]]:
    """Return mean paired delta, wins, and pair count against the reference model."""
    by_model_seed = {(row.get("model", ""), row.get("run_seed", "")): row for row in rows}
    reference_seeds = {
        seed for model, seed in by_model_seed if model == reference_model
    }
    models = sorted({model for model, _ in by_model_seed if model != reference_model})

    output: dict[str, dict[str, tuple[float, int, int]]] = {}
    for model in models:
        output[model] = {}
        shared_seeds = sorted(seed for seed in reference_seeds if (model, seed) in by_model_seed)
        for metric in metrics:
            deltas: list[float] = []
            wins = 0
            for seed in shared_seeds:
                value = parse_float(by_model_seed[(model, seed)].get(metric))
                ref_value = parse_float(by_model_seed[(reference_model, seed)].get(metric))
                if value is None or ref_value is None:
                    continue
                delta = value - ref_value
                deltas.append(delta)
                wins += int(delta > 0)
            if deltas:
                output[model][metric] = (mean(deltas), wins, len(deltas))
    return output


def format_cell(cell: SummaryCell | None) -> str:
    if cell is None:
        return "-"
    return f"{cell.mean:.3f} +/- {cell.std:.3f}"


def render_markdown(
    summaries: dict[str, dict[str, SummaryCell]],
    metrics: tuple[str, ...],
    reference_model: str | None = None,
    deltas: dict[str, dict[str, tuple[float, int, int]]] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("| Model | " + " | ".join(metrics) + " |")
    lines.append("|---|" + "|".join("---:" for _ in metrics) + "|")
    for model in sorted(summaries):
        cells = [format_cell(summaries[model].get(metric)) for metric in metrics]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")

    if reference_model and deltas:
        lines.append("")
        lines.append(f"Paired delta vs `{reference_model}`:")
        lines.append("")
        lines.append("| Model | " + " | ".join(metrics) + " |")
        lines.append("|---|" + "|".join("---:" for _ in metrics) + "|")
        for model in sorted(deltas):
            cells = []
            for metric in metrics:
                item = deltas[model].get(metric)
                if item is None:
                    cells.append("-")
                else:
                    delta, wins, count = item
                    cells.append(f"{delta:+.3f} ({wins}/{count})")
            lines.append(f"| `{model}` | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def infer_metrics(rows: list[dict[str, str]], requested: list[str] | None) -> tuple[str, ...]:
    if requested:
        return tuple(requested)
    available = set(rows[0]) if rows else set()
    return tuple(metric for metric in DEFAULT_METRICS if metric in available)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real DET CSV final-step metrics.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV written by scripts/train_det_real.py")
    parser.add_argument("--reference-model", default="", help="Optional model for paired deltas.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Metric columns to summarize.")
    args = parser.parse_args()

    rows = load_final_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    metrics = infer_metrics(rows, args.metrics)
    summaries = summarize_by_model(rows, metrics)
    deltas = (
        paired_delta(rows, args.reference_model, metrics)
        if args.reference_model
        else None
    )
    print(render_markdown(summaries, metrics, args.reference_model or None, deltas))


if __name__ == "__main__":
    main()
