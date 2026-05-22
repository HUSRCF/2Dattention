"""Join per-category coverage diagnostics with COCO split category counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument(
        "--annotation-entry",
        action="append",
        required=True,
        help="Named COCO annotation JSON in the form split=/path/to/_annotations.coco.json.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.coverage.open(newline="", encoding="utf-8")))
    count_columns = []
    counts_by_split = {}
    names_by_category = {}
    for entry in args.annotation_entry:
        split, path_text = parse_entry(entry)
        path = Path(path_text)
        count_columns.append(f"{split}_gt_count")
        counts, names = count_categories(path)
        counts_by_split[split] = counts
        names_by_category.update(names)
    fieldnames = list(rows[0].keys()) if rows else ["category_id", "category_name"]
    for column in count_columns:
        if column not in fieldnames:
            fieldnames.append(column)
    merged = []
    for row in rows:
        category_id = int(row["category_id"])
        row = dict(row)
        row["category_name"] = row.get("category_name") or names_by_category.get(category_id, str(category_id))
        for split, counts in counts_by_split.items():
            row[f"{split}_gt_count"] = counts.get(category_id, 0)
        merged.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)
    print(f"saved_merged_per_category_coverage: {args.out}")


def parse_entry(entry: str) -> tuple[str, str]:
    if "=" not in entry:
        raise ValueError(f"expected split=path entry, got {entry!r}")
    split, path = entry.split("=", 1)
    if not split:
        raise ValueError(f"empty split name in entry {entry!r}")
    return split, path


def count_categories(path: Path) -> tuple[dict[int, int], dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in data.get("categories", [])
    }
    counts: dict[int, int] = {}
    for annotation in data.get("annotations", []):
        category_id = int(annotation["category_id"])
        counts[category_id] = counts.get(category_id, 0) + 1
    return counts, names


if __name__ == "__main__":
    main()
