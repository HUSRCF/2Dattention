from __future__ import annotations

import torch

from pathlib import Path

from PIL import Image

from scripts.relabel_coco_predictions_by_torchvision_imagenet import (
    CocoImageDataset,
    choose_category,
    largest_category_by_image,
)


def test_choose_category_maps_highest_synset_to_coco_category() -> None:
    probs = torch.tensor([0.1, 0.8, 0.2])

    category = choose_category(
        probs,
        index_to_synset={0: "n0", 1: "n1", 2: "n2"},
        category_id_by_synset={"n1": 7, "n2": 9},
        restrict_to_annotation_categories=False,
        allowed_category_ids={7, 9},
    )

    assert category == 7


def test_largest_category_by_image() -> None:
    data = {
        "annotations": [
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 3, 3], "area": 9},
            {"image_id": 1, "category_id": 4, "bbox": [0, 0, 5, 5], "area": 25},
        ]
    }

    assert largest_category_by_image(data) == {1: 4}


def test_coco_image_dataset_can_return_source_image_id(tmp_path: Path) -> None:
    Image.new("RGB", (16, 16), "white").save(tmp_path / "crop.jpg")
    data = {
        "images": [{"id": 1, "source_image_id": 99, "file_name": "crop.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 3, 3], "area": 9}],
    }

    dataset = CocoImageDataset(data=data, image_root=tmp_path, use_source_image_id=True)
    _, image_id, label = dataset[0]

    assert int(image_id.item()) == 99
    assert int(label.item()) == 2
