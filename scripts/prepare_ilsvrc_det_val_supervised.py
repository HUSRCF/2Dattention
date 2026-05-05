"""Prepare supervised helpers from ILSVRC2013 detection validation data."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObjectBox:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class Annotation:
    image_id: str
    width: int
    height: int
    boxes: tuple[ObjectBox, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument(
        "--anno-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_bbox_val/ILSVRC2013_DET_bbox_val"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_val_supervised"),
    )
    parser.add_argument(
        "--top-classes",
        type=int,
        default=20,
        help="number of most frequent single-label classes to link; <=0 keeps all classes",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=200,
        help="max symlinked images per class; <=0 keeps all matching images",
    )
    parser.add_argument(
        "--imagefolder-name",
        default="single_label_imagefolder",
        help="subdirectory name for the symlinked ImageFolder dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remove the target ImageFolder directory before recreating symlinks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.anno_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_root / "det_val_manifest.csv"
    write_detection_manifest(manifest_path, annotations, args.image_root)

    single_label = [
        annotation for annotation in annotations if len({box.label for box in annotation.boxes}) == 1
    ]
    label_counts = Counter(annotation.boxes[0].label for annotation in single_label)
    if args.top_classes <= 0:
        selected_labels = sorted(label_counts)
    else:
        selected_labels = [label for label, _ in label_counts.most_common(args.top_classes)]
    imagefolder_root = args.out_root / args.imagefolder_name
    linked = build_imagefolder_subset(
        imagefolder_root=imagefolder_root,
        image_root=args.image_root,
        annotations=single_label,
        selected_labels=set(selected_labels),
        max_per_class=args.max_per_class,
        overwrite=args.overwrite,
    )

    labels_path = args.out_root / f"{args.imagefolder_name}_classes.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "single_label_images", "linked_images"])
        for label in selected_labels:
            writer.writerow([label, label_counts[label], linked[label]])

    print("annotations:", len(annotations))
    print("manifest:", manifest_path)
    print("single_label_images:", len(single_label))
    print("imagefolder:", imagefolder_root)
    print("classes:", len(selected_labels))
    print("linked_images:", sum(linked.values()))


def load_annotations(anno_root: Path) -> list[Annotation]:
    annotations = []
    for xml_path in sorted(anno_root.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = required_text(root, "filename")
        size = root.find("size")
        if size is None:
            raise ValueError(f"missing size in {xml_path}")
        width = int(required_text(size, "width"))
        height = int(required_text(size, "height"))
        boxes = []
        for obj in root.findall("object"):
            label = required_text(obj, "name")
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"missing bndbox in {xml_path}")
            boxes.append(
                ObjectBox(
                    label=label,
                    xmin=int(required_text(box, "xmin")),
                    ymin=int(required_text(box, "ymin")),
                    xmax=int(required_text(box, "xmax")),
                    ymax=int(required_text(box, "ymax")),
                )
            )
        annotations.append(
            Annotation(
                image_id=filename,
                width=width,
                height=height,
                boxes=tuple(boxes),
            )
        )
    return annotations


def required_text(root: ET.Element, tag: str) -> str:
    child = root.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"missing XML tag: {tag}")
    return child.text.strip()


def write_detection_manifest(
    path: Path,
    annotations: list[Annotation],
    image_root: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_id",
                "image_path",
                "width",
                "height",
                "label",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
        )
        for annotation in annotations:
            image_path = image_root / f"{annotation.image_id}.JPEG"
            for box in annotation.boxes:
                writer.writerow(
                    [
                        annotation.image_id,
                        image_path,
                        annotation.width,
                        annotation.height,
                        box.label,
                        box.xmin,
                        box.ymin,
                        box.xmax,
                        box.ymax,
                    ]
                )


def build_imagefolder_subset(
    imagefolder_root: Path,
    image_root: Path,
    annotations: list[Annotation],
    selected_labels: set[str],
    max_per_class: int,
    overwrite: bool,
) -> Counter[str]:
    if overwrite and imagefolder_root.exists():
        shutil.rmtree(imagefolder_root)
    imagefolder_root.mkdir(parents=True, exist_ok=True)
    linked: Counter[str] = Counter()

    for annotation in annotations:
        label = annotation.boxes[0].label
        if label not in selected_labels:
            continue
        if max_per_class > 0 and linked[label] >= max_per_class:
            continue

        source = (image_root / f"{annotation.image_id}.JPEG").resolve()
        if not source.exists():
            continue
        class_dir = imagefolder_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        target = class_dir / source.name
        if not target.exists():
            os.symlink(source, target)
        linked[label] += 1

    return linked


if __name__ == "__main__":
    sys.exit(main())
