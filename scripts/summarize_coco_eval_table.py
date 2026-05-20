"""Build a compact table from named COCOeval CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_coco_predictions import METRIC_NAMES


FIELDNAMES = ("name", "source", *METRIC_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Named input in the form name=/path/to/cocoeval.csv.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [summarize_entry(entry) for entry in args.entry]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_summary: {args.out}")
    for row in rows:
        print(f"{row['name']}: ap={row['ap']:.4f} ap50={row['ap50']:.4f} ap75={row['ap75']:.4f}")


def summarize_entry(entry: str) -> dict[str, float | str]:
    name, path = parse_entry(entry)
    metrics = read_cocoeval_csv(path)
    return {"name": name, "source": str(path), **metrics}


def parse_entry(entry: str) -> tuple[str, Path]:
    if "=" not in entry:
        raise ValueError(f"entry must use name=path format: {entry}")
    name, path_text = entry.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"entry name is empty: {entry}")
    path = Path(path_text).expanduser()
    return name, path


def read_cocoeval_csv(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one COCOeval row in {path}, found {len(rows)}")
    row = rows[0]
    missing = [name for name in METRIC_NAMES if name not in row]
    if missing:
        raise KeyError(f"COCOeval CSV {path} missing columns: {missing}")
    return {name: float(row[name]) for name in METRIC_NAMES}


if __name__ == "__main__":
    main()
