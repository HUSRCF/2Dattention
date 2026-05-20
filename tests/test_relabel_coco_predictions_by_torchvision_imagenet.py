from __future__ import annotations

import torch

from pathlib import Path

from PIL import Image

from scripts.relabel_coco_predictions_by_torchvision_imagenet import (
    CocoImageDataset,
    MODEL_CHOICES,
    choose_category,
    largest_category_by_image,
    ranked_mapped_categories,
    relabel_predictions,
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


def test_ranked_mapped_categories_skips_unmapped_imagenet_classes() -> None:
    probs = torch.tensor([0.9, 0.8, 0.7, 0.6])

    categories = ranked_mapped_categories(
        probs,
        index_to_synset={0: "unmapped", 1: "n1", 2: "n2", 3: "n3"},
        category_id_by_synset={"n1": 7, "n2": 9, "n3": 11},
        restrict_to_annotation_categories=True,
        allowed_category_ids={7, 11},
        limit=2,
    )

    assert categories == [7, 11]


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


def test_coco_image_dataset_uses_injected_preprocess(tmp_path: Path) -> None:
    Image.new("RGB", (16, 16), "white").save(tmp_path / "image.jpg")
    data = {
        "images": [{"id": 1, "file_name": "image.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 3, 3], "area": 9}],
    }

    def preprocess(image: Image.Image) -> torch.Tensor:
        assert image.size == (16, 16)
        return torch.ones(3, 5, 7)

    dataset = CocoImageDataset(data=data, image_root=tmp_path, preprocess=preprocess)
    image, _, _ = dataset[0]

    assert image.shape == (3, 5, 7)
    assert float(image.sum()) == 105.0


def test_model_choices_include_stronger_teachers() -> None:
    assert "resnet18" in MODEL_CHOICES
    assert "resnet50" in MODEL_CHOICES
    assert "convnext_tiny" in MODEL_CHOICES


def test_relabel_predictions_can_expand_topk_categories_with_score_multiplication() -> None:
    predictions = [{"image_id": 1, "category_id": 99, "bbox": [1, 2, 3, 4], "score": 0.8}]

    relabeled = relabel_predictions(
        predictions,
        {1: [(7, 0.5), (9, 0.25)]},
        score_mode="multiply",
    )

    assert [row["category_id"] for row in relabeled] == [7, 9]
    assert [row["score"] for row in relabeled] == [0.4, 0.2]
    assert all(row["bbox"] == [1, 2, 3, 4] for row in relabeled)
