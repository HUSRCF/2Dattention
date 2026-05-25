"""Update an RF-DETR mainline summary from a standard artifact prefix."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rank_rfdetr_stage_gate_rows import rank_rows
from scripts.rank_rfdetr_stage_gate_rows import write_rows as write_rank_rows
from scripts.summarize_rfdetr_mainline import FIELDNAMES
from scripts.summarize_rfdetr_mainline import summarize_entry


DEFAULT_RANK_METRICS = "class_ap50,offcenter_ap50,small_ap50,loc_ap50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--artifact-prefix", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--candidate-score-mode",
        choices=("candidate", "group_max", "oracle_iou"),
        default="group_max",
        help="Candidate-oracle COCO suffix used by the artifact bundle.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--append-only",
        action="store_true",
        help="Append even if a row with the same setting exists. Default replaces by setting.",
    )
    parser.add_argument("--high-support-min-groups", type=int, default=20)
    parser.add_argument("--rank-out", type=Path)
    parser.add_argument("--rank-metrics", default=DEFAULT_RANK_METRICS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = summarize_standard_artifacts(
        setting=args.setting,
        protocol=args.protocol,
        artifact_prefix=args.artifact_prefix,
        notes=args.notes,
        candidate_score_mode=args.candidate_score_mode,
        high_support_min_groups=args.high_support_min_groups,
    )
    rows = update_summary_rows(read_rows(args.base_summary), row, replace=not args.append_only)
    write_summary(args.out, rows)
    print(f"saved_rfdetr_mainline_summary: {args.out}")
    print(
        f"{row['setting']}: class_ap50={row['class_ap50']} "
        f"offcenter_ap50={row['offcenter_ap50']} small_ap50={row['small_ap50']} "
        f"loc_ap50={row['loc_ap50']} cand_hit={row['candidate_hit_rate']}"
    )
    if args.rank_out is not None:
        metrics = [metric.strip() for metric in args.rank_metrics.split(",") if metric.strip()]
        ranked = rank_rows(summary_paths=[args.out], metrics=metrics)
        args.rank_out.parent.mkdir(parents=True, exist_ok=True)
        write_rank_rows(args.rank_out, ranked, metrics)
        print(f"saved_rfdetr_stage_gate_rank: {args.rank_out}")


def summarize_standard_artifacts(
    *,
    setting: str,
    protocol: str,
    artifact_prefix: Path,
    notes: str = "",
    candidate_score_mode: str = "group_max",
    high_support_min_groups: int = 20,
) -> dict[str, str]:
    return summarize_entry(
        build_standard_entry(
            setting=setting,
            protocol=protocol,
            artifact_prefix=artifact_prefix,
            notes=notes,
            candidate_score_mode=candidate_score_mode,
        ),
        high_support_min_groups=high_support_min_groups,
    )


def build_standard_entry(
    *,
    setting: str,
    protocol: str,
    artifact_prefix: Path,
    notes: str = "",
    candidate_score_mode: str = "group_max",
) -> str:
    prefix = str(artifact_prefix)
    candidate_score_suffix = candidate_score_mode.replace("_", "")
    fields = [
        setting,
        f"protocol={protocol}",
        f"class={prefix}_test_cocoeval.csv",
        f"loc={prefix}_test_loc_cocoeval.csv",
        f"slices={prefix}_test_slices.csv",
        f"candidate={prefix}_candidate_oracle_summary.csv",
        f"candidate_per_category={prefix}_candidate_oracle_per_category.csv",
        f"candidate_oracle={prefix}_candidate_oracle_{candidate_score_suffix}_cocoeval.csv",
        f"coverage={prefix}_category_coverage_gap.csv",
        f"score_loc={prefix}_score_iou_loc.csv",
        f"score_class={prefix}_score_iou_classaware.csv",
    ]
    if notes:
        fields.append(f"notes={notes}")
    return ",".join(fields)


def update_summary_rows(
    existing_rows: list[dict[str, str]],
    new_row: dict[str, str],
    *,
    replace: bool = True,
) -> list[dict[str, str]]:
    if not replace:
        return [*existing_rows, new_row]
    rows = []
    replaced = False
    for row in existing_rows:
        if row.get("setting") == new_row["setting"]:
            if row.get("protocol", "") != new_row.get("protocol", ""):
                raise ValueError(
                    "refusing to replace row with matching setting but different protocol: "
                    f"{new_row['setting']!r} existing={row.get('protocol', '')!r} "
                    f"new={new_row.get('protocol', '')!r}; use --append-only or a unique setting"
                )
            rows.append(new_row)
            replaced = True
        else:
            rows.append(row)
    if not replaced:
        rows.append(new_row)
    return rows


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


if __name__ == "__main__":
    main()
