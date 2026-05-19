from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import torch

from scripts.train_torchvision_coco_detector import (
    CocoDetectionLite,
    ap_at_iou,
    build_model,
    count_trainable_parameters,
    collate_detection,
    mean_best_iou,
    set_trainable_parts,
    voc_ap,
)


def test_coco_detection_lite_scales_boxes(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (100, 50), "white").save(image_root / "sample.JPEG")
    annotations = {
        "images": [{"id": 1, "file_name": "sample.JPEG", "width": 100, "height": 50}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10, 5, 40, 20],
                "area": 800,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 2, "name": "class_a"}],
    }
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(annotations), encoding="utf-8")

    dataset = CocoDetectionLite(json_path, image_root=image_root, image_size=64)
    image, target = dataset[0]

    assert image.shape == (3, 64, 64)
    assert torch.allclose(target["boxes"][0], torch.tensor([6.4, 6.4, 32.0, 32.0]))
    assert target["labels"].tolist() == [2]


def test_torchvision_detector_metrics_match_greedy_ap() -> None:
    gt_by_image = {
        1: (
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            torch.tensor([1, 2]),
        )
    }
    predictions = [
        {"image_id": 1, "box": torch.tensor([0.0, 0.0, 10.0, 10.0]), "label": 1, "score": 0.9},
        {"image_id": 1, "box": torch.tensor([20.0, 20.0, 30.0, 30.0]), "label": 2, "score": 0.8},
        {"image_id": 1, "box": torch.tensor([40.0, 40.0, 50.0, 50.0]), "label": 1, "score": 0.7},
    ]

    assert mean_best_iou(predictions, gt_by_image) == 1.0
    assert ap_at_iou(predictions, gt_by_image, iou_threshold=0.5, class_aware=False) == 1.0
    assert ap_at_iou(predictions, gt_by_image, iou_threshold=0.5, class_aware=True) == 1.0


def test_torchvision_detector_metrics_penalize_wrong_class() -> None:
    gt_by_image = {1: (torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([2]))}
    predictions = [
        {"image_id": 1, "box": torch.tensor([0.0, 0.0, 10.0, 10.0]), "label": 1, "score": 0.9},
    ]

    assert ap_at_iou(predictions, gt_by_image, iou_threshold=0.5, class_aware=False) == 1.0
    assert ap_at_iou(predictions, gt_by_image, iou_threshold=0.5, class_aware=True) == 0.0


def test_torchvision_detector_mean_iou_handles_images_without_predictions() -> None:
    gt_by_image = {
        1: (torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([1])),
        2: (torch.tensor([[20.0, 20.0, 30.0, 30.0]]), torch.tensor([1])),
    }
    predictions = [
        {"image_id": 1, "box": torch.tensor([0.0, 0.0, 10.0, 10.0]), "label": 1, "score": 0.9},
    ]

    assert mean_best_iou(predictions, gt_by_image) == 0.5


def test_collate_detection_keeps_lists() -> None:
    batch = [
        (
            torch.zeros(3, 4, 4),
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.ones(1, dtype=torch.int64),
                "image_id": torch.tensor([1]),
            },
        )
    ]
    images, targets = collate_detection(batch)

    assert isinstance(images, list)
    assert isinstance(targets, list)
    assert images[0].shape == (3, 4, 4)


def test_build_model_replaces_predictor_for_requested_class_count() -> None:
    model = build_model(num_classes=7, image_size=64, weights="none")

    assert model.roi_heads.box_predictor.cls_score.out_features == 7
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 28


def test_build_model_can_load_local_checkpoint_before_replacing_head(tmp_path: Path) -> None:
    source = build_model(num_classes=91, image_size=64, weights="none")
    checkpoint_path = tmp_path / "model.pt"
    torch.save(source.state_dict(), checkpoint_path)

    model = build_model(num_classes=5, image_size=64, weights="none", weights_file=checkpoint_path)

    assert model.roi_heads.box_predictor.cls_score.out_features == 5
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 20


def test_set_trainable_parts_can_freeze_to_box_predictor() -> None:
    model = build_model(num_classes=5, image_size=64, weights="none")
    set_trainable_parts(model, "box_predictor")

    assert count_trainable_parameters(model) > 0
    assert all(parameter.requires_grad for parameter in model.roi_heads.box_predictor.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.rpn.parameters())


def test_set_trainable_parts_can_freeze_to_roi_heads() -> None:
    model = build_model(num_classes=5, image_size=64, weights="none")
    set_trainable_parts(model, "roi_heads")

    assert all(parameter.requires_grad for parameter in model.roi_heads.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.rpn.parameters())


def test_voc_ap_handles_empty_curve() -> None:
    assert voc_ap(torch.tensor([]), torch.tensor([])) == 0.0
