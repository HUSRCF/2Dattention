import pytest

from scripts.build_coco_pseudolabel_dataset import build_pseudo_coco


def _mini_coco():
    return {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 99,
                "image_id": 1,
                "category_id": 1,
                "bbox": [1, 1, 2, 2],
                "area": 4,
                "iscrowd": 0,
            }
        ],
    }


def test_build_pseudo_coco_filters_and_topk():
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [2, 2, 3, 3], "score": 0.8},
        {"image_id": 1, "category_id": 1, "bbox": [3, 3, 3, 3], "score": 0.7},
        {"image_id": 2, "category_id": 1, "bbox": [1, 1, 3, 3], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [1, 1, 3, 3], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3], "score": 0.1},
        {"image_id": 1, "category_id": 1, "bbox": [1, 1, -1, 3], "score": 0.9},
    ]

    out_coco, drop_stats = build_pseudo_coco(
        _mini_coco(),
        predictions,
        min_score=0.2,
        topk_per_image=2,
        min_area=1.0,
        include_gt=False,
    )

    assert [ann["teacher_score"] for ann in out_coco["annotations"]] == [0.9, 0.8]
    assert drop_stats["raw_predictions"] == 7
    assert drop_stats["dropped_image_id"] == 1
    assert drop_stats["dropped_category_id"] == 1
    assert drop_stats["dropped_score"] == 1
    assert drop_stats["dropped_bbox"] == 1
    assert drop_stats["kept_before_topk"] == 3
    assert drop_stats["dropped_topk"] == 1


def test_build_pseudo_coco_include_gt_rewrites_annotation_ids():
    out_coco, _ = build_pseudo_coco(
        _mini_coco(),
        [{"image_id": 1, "category_id": 1, "bbox": [4, 4, 2, 2], "score": 0.9}],
        min_score=0.2,
        topk_per_image=1,
        min_area=1.0,
        include_gt=True,
    )

    assert [ann["id"] for ann in out_coco["annotations"]] == [1, 2]
    assert out_coco["annotations"][0]["bbox"] == [1, 1, 2, 2]
    assert out_coco["annotations"][1]["teacher_score"] == 0.9


def test_build_pseudo_coco_rejects_invalid_topk_and_area():
    with pytest.raises(ValueError, match="topk_per_image"):
        build_pseudo_coco(_mini_coco(), [], min_score=0.2, topk_per_image=0, min_area=1.0, include_gt=False)
    with pytest.raises(ValueError, match="min_area"):
        build_pseudo_coco(_mini_coco(), [], min_score=0.2, topk_per_image=1, min_area=-1.0, include_gt=False)
