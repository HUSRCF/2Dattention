from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.prepare_rfdetr_dataset import prepare_rfdetr_split, stable_image_name


def test_stable_image_name_prefixes_image_id() -> None:
    assert stable_image_name(7, "sub/ILSVRC2012 val 7.JPEG") == "000000000007_ILSVRC2012_val_7.JPEG"


def test_prepare_rfdetr_split_materializes_images_and_rewrites_names(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (20, 10), "white").save(image_root / "sample.JPEG")
    annotation_json = tmp_path / "ann.json"
    annotation_json.write_text(
        json.dumps(
            {
                "images": [{"id": 3, "file_name": "sample.JPEG", "width": 20, "height": 10}],
                "annotations": [
                    {"id": 1, "image_id": 3, "category_id": 1, "bbox": [1, 2, 3, 4], "area": 12}
                ],
                "categories": [{"id": 1, "name": "object"}],
            }
        ),
        encoding="utf-8",
    )

    out = prepare_rfdetr_split(
        annotation_json=annotation_json,
        image_root=image_root,
        split_dir=tmp_path / "rfdetr" / "train",
        link_mode="copy",
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert out.name == "_annotations.coco.json"
    assert data["images"][0]["file_name"] == "000000000003_sample.JPEG"
    assert (out.parent / "000000000003_sample.JPEG").exists()
