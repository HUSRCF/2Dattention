from __future__ import annotations

from scripts.build_coco_hard_category_oversample import build_oversampled_train


def test_build_oversampled_train_avoids_reserved_valid_test_image_ids() -> None:
    data = {
        "images": [
            {"id": 10, "file_name": "a.jpg"},
            {"id": 11, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 5},
            {"id": 2, "image_id": 11, "category_id": 1},
        ],
        "categories": [{"id": 5, "name": "hard"}],
    }

    oversampled, _summary = build_oversampled_train(
        data,
        hard_categories={5},
        target_hard_boxes=3,
        max_repeat=3,
        reserved_image_ids={10, 11, 12, 13},
    )

    image_ids = [int(image["id"]) for image in oversampled["images"]]
    assert image_ids == [10, 14, 15, 11]
    assert len(image_ids) == len(set(image_ids))
