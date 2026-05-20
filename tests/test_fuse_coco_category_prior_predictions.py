from __future__ import annotations

from scripts.fuse_coco_category_prior_predictions import fuse_category_prior_predictions


def test_fuse_category_prior_predictions_uses_common_image_box_category() -> None:
    primary = [
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.25},
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
    ]
    secondary = [
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 1.0},
    ]

    fused = fuse_category_prior_predictions(
        primary_predictions=primary,
        secondary_predictions=secondary,
        mode="geomean",
    )

    assert fused == [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.5}]


def test_fuse_category_prior_predictions_can_keep_unmatched_primary() -> None:
    primary = [{"image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.25}]

    fused = fuse_category_prior_predictions(
        primary_predictions=primary,
        secondary_predictions=[],
        mode="geomean",
        keep_unmatched_primary=True,
    )

    assert fused == primary
