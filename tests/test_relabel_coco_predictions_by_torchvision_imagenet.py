from __future__ import annotations

import torch

from scripts.relabel_coco_predictions_by_torchvision_imagenet import choose_category, largest_category_by_image


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
