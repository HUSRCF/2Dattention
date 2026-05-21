from __future__ import annotations

from scripts.split_coco_by_images import split_coco_by_images, subset_coco


def test_split_coco_by_images_creates_disjoint_image_sets() -> None:
    data = {
        "images": [{"id": image_id, "file_name": f"{image_id}.jpg"} for image_id in range(6)],
        "annotations": [
            {"id": image_id, "image_id": image_id, "category_id": image_id % 2 + 1}
            for image_id in range(6)
        ],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
    }

    a_data, b_data = split_coco_by_images(data, a_fraction=0.5, seed=7)
    a_ids = {int(image["id"]) for image in a_data["images"]}
    b_ids = {int(image["id"]) for image in b_data["images"]}

    assert a_ids
    assert b_ids
    assert a_ids.isdisjoint(b_ids)
    assert a_ids | b_ids == set(range(6))
    assert {int(row["image_id"]) for row in a_data["annotations"]} == a_ids
    assert {int(row["image_id"]) for row in b_data["annotations"]} == b_ids


def test_subset_coco_drops_unused_categories() -> None:
    data = {
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 2}],
        "categories": [{"id": 1, "name": "unused"}, {"id": 2, "name": "kept"}],
    }

    subset = subset_coco(data, {1})

    assert [category["id"] for category in subset["categories"]] == [2]


def test_subset_coco_can_preserve_all_categories() -> None:
    data = {
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 2}],
        "categories": [{"id": 1, "name": "unused"}, {"id": 2, "name": "kept"}],
    }

    subset = subset_coco(data, {1}, keep_all_categories=True)

    assert [category["id"] for category in subset["categories"]] == [1, 2]
