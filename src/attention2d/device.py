"""Device selection helpers for local Apple Silicon torch trials."""

from __future__ import annotations

import torch


def get_best_device() -> torch.device:
    """Prefer Apple MPS when PyTorch can access it, otherwise use CPU."""

    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
