from pathlib import Path

import torch

from scripts.compare_bbox_center_regression_models import baseline_l2, bbox_center_target
from scripts.compare_bbox_heatmap_models import (
    argmax_coords,
    center_to_cell_index,
    gaussian_heatmap,
    softargmax_coords,
)
from scripts.compare_bbox_mask_models import (
    bbox_mask_target,
    mask_centroid_from_probs,
    mask_metrics_from_probs,
)
from scripts.compare_bbox_probe_models import BBoxSample, bbox_probe_label, num_probe_classes


def make_sample(xmin: int, ymin: int, xmax: int, ymax: int) -> BBoxSample:
    return BBoxSample(
        image_id="sample",
        image_path=Path("sample.JPEG"),
        width=100,
        height=100,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )


def test_quadrant4_labels() -> None:
    assert bbox_probe_label(make_sample(0, 0, 20, 20), "quadrant4") == 0
    assert bbox_probe_label(make_sample(80, 0, 100, 20), "quadrant4") == 1
    assert bbox_probe_label(make_sample(0, 80, 20, 100), "quadrant4") == 2
    assert bbox_probe_label(make_sample(80, 80, 100, 100), "quadrant4") == 3


def test_grid9_and_size3_labels() -> None:
    assert bbox_probe_label(make_sample(0, 0, 20, 20), "grid9") == 0
    assert bbox_probe_label(make_sample(80, 80, 100, 100), "grid9") == 8
    assert bbox_probe_label(make_sample(0, 0, 20, 20), "size3") == 0
    assert bbox_probe_label(make_sample(0, 0, 50, 50), "size3") == 1
    assert bbox_probe_label(make_sample(0, 0, 80, 80), "size3") == 2


def test_num_probe_classes() -> None:
    assert num_probe_classes("quadrant4") == 4
    assert num_probe_classes("grid9") == 9
    assert num_probe_classes("size3") == 3


def test_bbox_center_target_and_baseline_l2() -> None:
    sample = make_sample(20, 30, 60, 90)
    assert bbox_center_target(sample) == (0.4, 0.6)
    targets = torch.tensor([[0.4, 0.6], [0.5, 0.5]])
    assert baseline_l2(targets, torch.tensor([0.5, 0.5])) > 0.0


def test_heatmap_helpers() -> None:
    center = torch.tensor([0.5, 0.5])
    heatmap = gaussian_heatmap(center, size=5, sigma=1.0)
    assert torch.isclose(heatmap.sum(), torch.tensor(1.0))
    probs = heatmap.flatten().unsqueeze(0)
    argmax = argmax_coords(probs, size=5)
    softargmax = softargmax_coords(probs, size=5)
    assert torch.allclose(argmax, torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(softargmax, torch.tensor([[0.5, 0.5]]), atol=1e-4)
    assert int(center_to_cell_index(center.unsqueeze(0), size=5)[0]) == 12


def test_bbox_mask_helpers() -> None:
    sample = make_sample(25, 25, 75, 75)
    mask = bbox_mask_target(sample, size=4)
    assert torch.isclose(mask.sum(), torch.tensor(4.0))
    probs = mask.unsqueeze(0)
    centroid = mask_centroid_from_probs(probs, size=4)
    assert torch.allclose(centroid, torch.tensor([[0.5, 0.5]]))
    metrics = mask_metrics_from_probs(
        probs=probs,
        centers=torch.tensor([[0.5, 0.5]]),
        targets=probs,
        threshold=0.5,
        mask_size=4,
    )
    assert metrics["iou"] > 0.99
    assert metrics["dice"] > 0.99
    assert metrics["cell_balanced_acc"] > 0.99
