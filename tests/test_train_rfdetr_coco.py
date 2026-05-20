from __future__ import annotations

import pytest

from scripts.train_rfdetr_coco import build_rfdetr_model, validate_rfdetr_dataset


def test_validate_rfdetr_dataset_requires_standard_splits(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train"):
        validate_rfdetr_dataset(tmp_path)


def test_build_rfdetr_model_missing_package_message(monkeypatch) -> None:
    import importlib

    real_import_module = importlib.import_module

    def fake_import(name: str):
        if name == "rfdetr":
            raise ImportError("missing")
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="pip install rfdetr"):
        build_rfdetr_model("nano")
