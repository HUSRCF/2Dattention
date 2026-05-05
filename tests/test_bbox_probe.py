from pathlib import Path

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
