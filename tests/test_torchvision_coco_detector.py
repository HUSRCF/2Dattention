from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from PIL import Image
import pytest
import torch

from scripts.train_torchvision_coco_detector import (
    CocoDetectionLite,
    ap_at_iou,
    build_model,
    checkpoint_args,
    count_trainable_parameters,
    collate_detection,
    coco_category_id_by_label,
    ensure_same_category_mapping,
    load_checkpoint,
    max_category_id,
    mean_best_iou,
    save_checkpoint,
    scale_xyxy_to_original,
    set_trainable_parts,
    write_predictions,
    xyxy_to_xywh,
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
    assert torch.allclose(target["orig_size"], torch.tensor([50.0, 100.0]))
    assert torch.allclose(target["resized_size"], torch.tensor([64.0, 64.0]))
    assert target["labels"].tolist() == [1]


def test_coco_detection_lite_remaps_sparse_category_ids(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (32, 32), "white").save(image_root / "sample.JPEG")
    annotations = {
        "images": [{"id": 1, "file_name": "sample.JPEG", "width": 32, "height": 32}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 42, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0}
        ],
        "categories": [{"id": 42, "name": "class_sparse"}],
    }
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(annotations), encoding="utf-8")

    dataset = CocoDetectionLite(json_path, image_root=image_root, image_size=32)
    _, target = dataset[0]

    assert target["labels"].tolist() == [1]
    assert max_category_id(dataset) == 1
    assert coco_category_id_by_label(dataset) == {1: 42}


def test_ensure_same_category_mapping_rejects_mismatched_jsons(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (32, 32), "white").save(image_root / "sample.JPEG")

    def write_ann(path: Path, category_id: int) -> None:
        path.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "sample.JPEG", "width": 32, "height": 32}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": category_id,
                            "bbox": [1, 2, 3, 4],
                            "area": 12,
                            "iscrowd": 0,
                        }
                    ],
                    "categories": [{"id": category_id, "name": f"class_{category_id}"}],
                }
            ),
            encoding="utf-8",
        )

    ann_a = tmp_path / "a.json"
    ann_b = tmp_path / "b.json"
    write_ann(ann_a, 42)
    write_ann(ann_b, 99)
    dataset_a = CocoDetectionLite(ann_a, image_root=image_root, image_size=32)
    dataset_b = CocoDetectionLite(ann_b, image_root=image_root, image_size=32)

    with pytest.raises(ValueError, match="category mappings differ"):
        ensure_same_category_mapping(dataset_a, dataset_b)


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


def test_set_trainable_parts_can_freeze_to_rpn() -> None:
    model = build_model(num_classes=5, image_size=64, weights="none")
    set_trainable_parts(model, "rpn")

    assert all(parameter.requires_grad for parameter in model.rpn.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.roi_heads.parameters())


def test_voc_ap_handles_empty_curve() -> None:
    assert voc_ap(torch.tensor([]), torch.tensor([])) == 0.0


def test_write_predictions_uses_coco_detection_format(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    write_predictions(
        path,
        [
            {
                "image_id": 3,
                "label": 2,
                "box": torch.tensor([1.0, 2.0, 6.0, 8.0]),
                "coco_box": torch.tensor([10.0, 20.0, 60.0, 80.0]),
                "score": 0.75,
            }
        ],
    )

    records = json.loads(path.read_text(encoding="utf-8"))
    assert records == [{"image_id": 3, "category_id": 2, "bbox": [10.0, 20.0, 50.0, 60.0], "score": 0.75}]


def test_xyxy_to_xywh_clamps_negative_size() -> None:
    assert xyxy_to_xywh(torch.tensor([5.0, 6.0, 2.0, 4.0])) == [5.0, 6.0, 0.0, 0.0]


def test_scale_xyxy_to_original_inverts_dataset_resize() -> None:
    box = torch.tensor([6.4, 6.4, 32.0, 32.0])
    scaled = scale_xyxy_to_original(
        box,
        orig_size=torch.tensor([50.0, 100.0]),
        resized_size=torch.tensor([64.0, 64.0]),
    )

    assert torch.allclose(scaled, torch.tensor([10.0, 5.0, 50.0, 25.0]))


def test_checkpoint_roundtrip_restores_model_weights(tmp_path: Path) -> None:
    model = build_model(num_classes=5, image_size=64, weights="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, args=type("Args", (), {"foo": "bar"})())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
            break

    loaded_step = load_checkpoint(path, model, optimizer)
    reloaded = torch.load(path, map_location="cpu")
    first_name, first_weight = next(iter(model.state_dict().items()))
    assert torch.allclose(first_weight.cpu(), reloaded["model"][first_name])
    assert loaded_step == 0


def test_checkpoint_roundtrip_restores_global_step(tmp_path: Path) -> None:
    model = build_model(num_classes=5, image_size=64, weights="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, args=type("Args", (), {"foo": "bar"})(), step=17)
    loaded_step = load_checkpoint(path, model, optimizer)

    assert loaded_step == 17


def test_checkpoint_args_are_safe_scalars(tmp_path: Path) -> None:
    args = Namespace(path=tmp_path / "x", count=3, name="run", flag=True)

    values = checkpoint_args(args)

    assert values == {"path": str(tmp_path / "x"), "count": 3, "name": "run", "flag": True}
