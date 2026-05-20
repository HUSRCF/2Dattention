from __future__ import annotations

import json

import pytest

from scripts.train_rfdetr_coco import (
    build_rfdetr_model,
    detect_rfdetr_dataset_num_classes,
    main,
    rfdetr_availability_report,
    validate_rfdetr_dataset,
)


def test_validate_rfdetr_dataset_requires_standard_splits(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train"):
        validate_rfdetr_dataset(tmp_path)


def test_detect_rfdetr_dataset_num_classes_reads_train_categories(tmp_path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [],
                "annotations": [],
                "categories": [{"id": 7, "name": "a"}, {"id": 12, "name": "b"}],
            }
        ),
        encoding="utf-8",
    )

    assert detect_rfdetr_dataset_num_classes(tmp_path) == 2


def test_detect_rfdetr_dataset_num_classes_rejects_empty_categories(tmp_path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "_annotations.coco.json").write_text(
        json.dumps({"images": [], "annotations": [], "categories": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no categories"):
        detect_rfdetr_dataset_num_classes(tmp_path)


def test_main_rejects_num_classes_that_do_not_match_train_split(monkeypatch, tmp_path) -> None:
    for split in ("train", "valid", "test"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [],
                    "annotations": [],
                    "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_rfdetr_coco.py",
            "--dataset-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--num-classes",
            "3",
            "--check-only",
        ],
    )

    with pytest.raises(ValueError, match="does not match train split category count 2"):
        main()


def test_build_rfdetr_model_missing_package_message(monkeypatch) -> None:
    monkeypatch.setattr("scripts.train_rfdetr_coco.importlib.util.find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match="pip install rfdetr"):
        build_rfdetr_model("nano")


def test_build_rfdetr_model_reports_broken_transitive_import(monkeypatch) -> None:
    import importlib.machinery

    monkeypatch.setattr(
        "scripts.train_rfdetr_coco.importlib.util.find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None),
    )

    def fake_import_module(name: str):
        if name == "rfdetr":
            raise ImportError("protobuf mismatch")
        raise AssertionError(name)

    monkeypatch.setattr("scripts.train_rfdetr_coco.importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="transitive dependency.*protobuf mismatch"):
        build_rfdetr_model("nano")


def test_rfdetr_availability_report_lists_known_classes(monkeypatch) -> None:
    import types

    fake_module = types.SimpleNamespace(RFDETRNano=object, RFDETRBase=object)
    monkeypatch.setattr("scripts.train_rfdetr_coco.import_rfdetr_module", lambda: fake_module)

    report = rfdetr_availability_report()

    assert report["classes"] == ["RFDETRNano", "RFDETRBase"]


def test_build_rfdetr_model_passes_optional_constructor_kwargs(monkeypatch, tmp_path) -> None:
    import types

    seen_kwargs = {}

    class FakeNano:
        def __init__(self, **kwargs) -> None:
            seen_kwargs.update(kwargs)

    fake_module = types.SimpleNamespace(RFDETRNano=FakeNano)
    monkeypatch.setattr("scripts.train_rfdetr_coco.import_rfdetr_module", lambda: fake_module)

    model = build_rfdetr_model("nano", pretrain_weights=tmp_path / "weights.pth", num_classes=200, device="mps")

    assert isinstance(model, FakeNano)
    assert seen_kwargs == {"pretrain_weights": str(tmp_path / "weights.pth"), "num_classes": 200, "device": "mps"}
