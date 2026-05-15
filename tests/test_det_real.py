from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from scripts.train_det_real import (
    RealDetDataset,
    build_label_map,
    det_collate,
    filter_samples,
    load_real_det_samples,
)


def test_real_det_dataset_parses_xml_and_normalizes_boxes(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    anno_root = tmp_path / "annos"
    image_root.mkdir()
    anno_root.mkdir()
    Image.new("RGB", (100, 50), "white").save(image_root / "sample.JPEG")
    (anno_root / "sample.xml").write_text(
        """
        <annotation>
          <filename>sample</filename>
          <size><width>100</width><height>50</height></size>
          <object>
            <name>class_a</name>
            <bndbox><xmin>10</xmin><ymin>5</ymin><xmax>60</xmax><ymax>25</ymax></bndbox>
          </object>
          <object>
            <name>class_b</name>
            <bndbox><xmin>70</xmin><ymin>10</ymin><xmax>90</xmax><ymax>40</ymax></bndbox>
          </object>
        </annotation>
        """,
        encoding="utf-8",
    )

    samples = load_real_det_samples(anno_root, image_root)
    label_to_id = build_label_map(samples, top_classes=0)
    filtered = filter_samples(samples, set(label_to_id), max_samples=0)
    dataset = RealDetDataset(filtered, label_to_id, image_size=32, max_objects=2)
    image, target = dataset[0]

    assert image.shape == (3, 32, 32)
    assert target["labels"].shape == (2,)
    assert target["boxes"].shape == (2, 4)
    assert torch.allclose(target["boxes"][0], torch.tensor([0.35, 0.30, 0.50, 0.40]))

    images, targets = det_collate([dataset[0]])
    assert images.shape == (1, 3, 32, 32)
    assert len(targets) == 1
