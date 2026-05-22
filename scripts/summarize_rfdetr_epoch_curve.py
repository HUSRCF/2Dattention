"""Summarize direct RF-DETR Small epoch curves from the detector-side table."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FIELDNAMES = (
    "seed",
    "epoch",
    "setting",
    "class_ap",
    "class_ap50",
    "class_ap75",
    "loc_ap50",
    "offcenter_ap50",
    "center_ap50",
    "small_ap50",
    "medium_ap50",
    "large_ap50",
)


SETTING_RE = re.compile(r"^full_small_(?:(?:resume)?(?P<epoch>\d+)ep)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/rfdetr_indtest_seed43_detector_side_summary.csv"),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_epoch_curve(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_epoch_curve: {args.out}")
    for row in rows:
        print(
            f"seed{row['seed']} ep{row['epoch']}: "
            f"ap50={row['class_ap50']} ap75={row['class_ap75']} "
            f"loc_ap50={row['loc_ap50']}"
        )


def summarize_epoch_curve(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    out_rows: list[dict[str, str]] = []
    for row in source_rows:
        setting = row["setting"]
        match = SETTING_RE.match(setting)
        if not match:
            continue
        out_rows.append(
            {
                "seed": match.group("seed"),
                "epoch": match.group("epoch"),
                "setting": setting,
                "class_ap": row["class_ap"],
                "class_ap50": row["class_ap50"],
                "class_ap75": row["class_ap75"],
                "loc_ap50": row["loc_ap50"],
                "offcenter_ap50": row["offcenter_ap50"],
                "center_ap50": row["center_ap50"],
                "small_ap50": row["small_ap50"],
                "medium_ap50": row["medium_ap50"],
                "large_ap50": row["large_ap50"],
            }
        )

    out_rows.sort(key=lambda row: (int(row["seed"]), int(row["epoch"])))
    return out_rows


if __name__ == "__main__":
    main()
