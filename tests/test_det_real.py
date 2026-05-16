from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from scripts.train_det_real import (
    RealDetDataset,
    build_label_map,
    binary_auc,
    det_collate,
    duplicate_predictions_per_gt,
    filter_samples,
    load_real_det_samples,
    matcher_aware_quality_classification_loss,
    objectness_logits,
    objectness_ap50_for_image,
    oracle_query_mask_logits,
    pearson_corr,
    quality_score_multipliers,
    quality_head_loss_scale,
    query_quality_head_loss,
    query_ranking_diagnostics,
    score_iou_calibration_loss,
    set_quality_head_only_trainable,
)
from attention2d.detection import DetectionCriterion
from attention2d.detection import TinyAnchorRegionDETR


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


def test_real_det_ranking_diagnostics_capture_high_score_false_positive() -> None:
    pred_logits = torch.tensor(
        [
            [4.0, -2.0, -3.0],
            [3.0, -2.0, -3.0],
            [-2.0, 4.0, -3.0],
        ]
    )
    pred_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.40, 0.40],
            [0.52, 0.50, 0.40, 0.40],
            [0.10, 0.10, 0.10, 0.10],
        ]
    )
    target_boxes = torch.tensor([[0.50, 0.50, 0.40, 0.40]])
    target_labels = torch.tensor([0])

    diagnostics = query_ranking_diagnostics(pred_logits, pred_boxes, target_boxes, target_labels)

    assert torch.allclose(diagnostics["matched_assignment_class_correct"], torch.tensor([1.0]))
    assert torch.allclose(diagnostics["tp50_class_correct"], torch.tensor([1.0]))
    assert float(diagnostics["objectness_auc"]) < 0.5
    assert float(diagnostics["topk_fp_rate"]) == 0.0
    assert float(diagnostics["duplicate_per_gt"]) == 1.0
    assert diagnostics["matched_query_counts"].shape == (3,)


def test_real_det_scalar_diagnostics() -> None:
    assert torch.allclose(
        pearson_corr(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])),
        torch.tensor(1.0),
    )
    assert torch.allclose(
        binary_auc(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([True, True, False])),
        torch.tensor(1.0),
    )
    iou_matrix = torch.tensor([[0.6, 0.1], [0.7, 0.2], [0.0, 0.8]])
    assert torch.allclose(duplicate_predictions_per_gt(iou_matrix, threshold=0.5), torch.tensor(0.5))


def test_real_det_ranking_diagnostics_handle_empty_targets() -> None:
    diagnostics = query_ranking_diagnostics(
        pred_logits=torch.zeros(2, 3),
        pred_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
        target_boxes=torch.zeros(0, 4),
        target_labels=torch.zeros(0, dtype=torch.long),
    )

    assert diagnostics["matched_assignment_class_correct"].numel() == 0
    assert diagnostics["tp50_class_correct"].numel() == 0
    assert float(diagnostics["topk_fp_rate"]) == 0.0
    assert float(diagnostics["duplicate_per_gt"]) == 0.0


def test_real_det_oracle_query_mask_logits_from_targets() -> None:
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.50, 0.50]]),
        },
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        },
    ]
    logits = oracle_query_mask_logits(
        targets=targets,
        feature_height=8,
        feature_width=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert logits.shape == (2, 8, 8)
    assert float(logits.max()) == 8.0
    assert float(logits.min()) == -8.0
    assert int((logits[0] > 0).sum().item()) > int((logits[1] > 0).sum().item())


def test_real_det_score_iou_calibration_loss_backpropagates_to_logits_only() -> None:
    pred_logits = torch.tensor(
        [[[2.0, -1.0], [-1.0, 2.0]]],
        requires_grad=True,
    )
    pred_boxes = torch.tensor(
        [[[0.50, 0.50, 0.40, 0.40], [0.10, 0.10, 0.10, 0.10]]],
        requires_grad=True,
    )
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.40, 0.40]]),
        }
    ]
    loss = score_iou_calibration_loss(
        {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
        targets,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert pred_logits.grad is not None
    assert float(pred_logits.grad.abs().sum()) > 0.0
    assert pred_boxes.grad is None


def test_real_det_objectness_logits_are_foreground_vs_background() -> None:
    logits = torch.tensor(
        [
            [3.0, -3.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    values = objectness_logits(logits)

    assert values.shape == (2,)
    assert float(values[0]) > 0.0
    assert float(values[1]) < 0.0


def test_real_det_matcher_aware_quality_loss_uses_class_logits_only() -> None:
    pred_logits = torch.tensor(
        [[[2.0, -2.0, -1.0], [-2.0, 2.0, -1.0]]],
        requires_grad=True,
    )
    pred_boxes = torch.tensor(
        [[[0.50, 0.50, 0.40, 0.40], [0.10, 0.10, 0.10, 0.10]]],
        requires_grad=True,
    )
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.40, 0.40]]),
        }
    ]
    criterion = DetectionCriterion(num_classes=2)
    loss = matcher_aware_quality_classification_loss(
        {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
        targets,
        criterion,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert pred_logits.grad is not None
    assert float(pred_logits.grad.abs().sum()) > 0.0
    assert pred_boxes.grad is None


def test_real_det_query_quality_head_loss_uses_quality_logits_only() -> None:
    pred_logits = torch.tensor(
        [[[2.0, -2.0, -1.0], [-2.0, 2.0, -1.0]]],
        requires_grad=True,
    )
    pred_boxes = torch.tensor(
        [[[0.50, 0.50, 0.40, 0.40], [0.10, 0.10, 0.10, 0.10]]],
        requires_grad=True,
    )
    pred_quality_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.40, 0.40]]),
        }
    ]
    criterion = DetectionCriterion(num_classes=2)
    loss = query_quality_head_loss(
        {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_quality_logits": pred_quality_logits,
        },
        targets,
        criterion,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert pred_quality_logits.grad is not None
    assert float(pred_quality_logits.grad.abs().sum()) > 0.0
    assert pred_logits.grad is None
    assert pred_boxes.grad is None


def test_real_det_quality_score_can_rescue_ap_ranking() -> None:
    target_boxes = torch.tensor([[0.50, 0.50, 0.40, 0.40]])
    pred_boxes = torch.tensor(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.50, 0.50, 0.40, 0.40],
        ]
    )
    pred_logits = torch.tensor(
        [
            [4.0, -2.0],
            [2.0, -2.0],
        ]
    )
    quality_logits = torch.tensor([-4.0, 4.0])

    base_ap = objectness_ap50_for_image(pred_logits, pred_boxes, target_boxes)
    quality_scores = quality_score_multipliers(quality_logits)
    quality_ap = objectness_ap50_for_image(
        pred_logits,
        pred_boxes,
        target_boxes,
        score_multiplier=quality_scores[1.0],
    )

    assert float(base_ap) < 0.6
    assert float(quality_ap) > 0.99


def test_real_det_quality_head_loss_scale_supports_late_start_and_warmup() -> None:
    assert quality_head_loss_scale(step=50, start_step=100, warmup_steps=0) == 0.0
    assert quality_head_loss_scale(step=100, start_step=100, warmup_steps=0) == 1.0
    assert quality_head_loss_scale(step=100, start_step=100, warmup_steps=50) == 0.02
    assert quality_head_loss_scale(step=124, start_step=100, warmup_steps=50) == 0.5
    assert quality_head_loss_scale(step=200, start_step=100, warmup_steps=50) == 1.0


def test_real_det_set_quality_head_only_trainable_freezes_detector() -> None:
    model = TinyAnchorRegionDETR(embed_dim=16, num_classes=2, num_queries=4)
    set_quality_head_only_trainable(model)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("head.quality_head.") for name in trainable)
