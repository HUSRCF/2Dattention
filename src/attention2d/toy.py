"""Synthetic 2D tasks for early information-flow checks."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_aligned_pair_batch(
    batch_size: int,
    image_size: int = 32,
    grid_size: int = 8,
    noise_std: float = 0.03,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate images with two markers and label whether they share row/column.

    Class 1: the red and green markers are aligned on the same row or column.
    Class 0: the markers are not aligned on either axis.
    """

    if image_size % grid_size != 0:
        raise ValueError("image_size must be divisible by grid_size")
    if grid_size < 3:
        raise ValueError("grid_size must be >= 3")

    patch = image_size // grid_size
    images = noise_std * torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        generator=generator,
    )
    labels = torch.zeros(batch_size, dtype=torch.long)

    for idx in range(batch_size):
        positive = idx % 2 == 0
        labels[idx] = int(positive)

        y1 = _randint(grid_size, generator)
        x1 = _randint(grid_size, generator)

        if positive:
            if _randint(2, generator) == 0:
                y2 = y1
                x2 = _sample_different(x1, grid_size, generator)
            else:
                y2 = _sample_different(y1, grid_size, generator)
                x2 = x1
        else:
            y2 = _sample_different(y1, grid_size, generator)
            x2 = _sample_different(x1, grid_size, generator)

        _paint_marker(images[idx], channel=0, y=y1, x=x1, patch=patch)
        _paint_marker(images[idx], channel=1, y=y2, x=x2, patch=patch)

    order = torch.randperm(batch_size, generator=generator)
    images = images[order]
    labels = labels[order]

    if device is not None:
        images = images.to(device)
        labels = labels.to(device)
    return images, labels


def sample_oriented_pair_batch(
    batch_size: int,
    image_size: int = 32,
    grid_size: int = 8,
    noise_std: float = 0.03,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate a local 2D orientation task.

    Class 1: green marker is one cell to the right of red.
    Class 0: green marker is one cell below red.
    """

    if image_size % grid_size != 0:
        raise ValueError("image_size must be divisible by grid_size")
    if grid_size < 3:
        raise ValueError("grid_size must be >= 3")

    patch = image_size // grid_size
    images = noise_std * torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        generator=generator,
    )
    labels = torch.zeros(batch_size, dtype=torch.long)

    for idx in range(batch_size):
        positive = idx % 2 == 0
        labels[idx] = int(positive)
        y1 = _randint(grid_size - 1, generator)
        x1 = _randint(grid_size - 1, generator)
        if positive:
            y2, x2 = y1, x1 + 1
        else:
            y2, x2 = y1 + 1, x1

        _paint_marker(images[idx], channel=0, y=y1, x=x1, patch=patch)
        _paint_marker(images[idx], channel=1, y=y2, x=x2, patch=patch)

    order = torch.randperm(batch_size, generator=generator)
    images = images[order]
    labels = labels[order]

    if device is not None:
        images = images.to(device)
        labels = labels.to(device)
    return images, labels


def sample_distractor_aligned_pair_batch(
    batch_size: int,
    image_size: int = 32,
    grid_size: int = 8,
    noise_std: float = 0.05,
    distractors: int = 8,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Aligned-pair task with extra clutter markers.

    Class 1: the target red and green markers share row or column.
    Class 0: the target red and green markers share neither axis.
    Blue/gray distractors add non-target local evidence.
    """

    if image_size % grid_size != 0:
        raise ValueError("image_size must be divisible by grid_size")
    if grid_size < 4:
        raise ValueError("grid_size must be >= 4")

    patch = image_size // grid_size
    images = noise_std * torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        generator=generator,
    )
    labels = torch.zeros(batch_size, dtype=torch.long)

    for idx in range(batch_size):
        used: set[tuple[int, int]] = set()
        positive = idx % 2 == 0
        labels[idx] = int(positive)

        y1, x1 = _sample_unused_position(grid_size, used, generator)
        if positive:
            if _randint(2, generator) == 0:
                y2 = y1
                x2 = _sample_unused_axis_value(x1, grid_size, generator)
            else:
                y2 = _sample_unused_axis_value(y1, grid_size, generator)
                x2 = x1
        else:
            y2 = _sample_unused_axis_value(y1, grid_size, generator)
            x2 = _sample_unused_axis_value(x1, grid_size, generator)

        used.add((y2, x2))
        _paint_marker(images[idx], channel=0, y=y1, x=x1, patch=patch)
        _paint_marker(images[idx], channel=1, y=y2, x=x2, patch=patch)

        for _ in range(distractors):
            yd, xd = _sample_unused_position(grid_size, used, generator)
            if _randint(2, generator) == 0:
                _paint_marker(images[idx], channel=2, y=yd, x=xd, patch=patch)
            else:
                _paint_gray_marker(images[idx], y=yd, x=xd, patch=patch)

    order = torch.randperm(batch_size, generator=generator)
    images = images[order]
    labels = labels[order]

    if device is not None:
        images = images.to(device)
        labels = labels.to(device)
    return images, labels


def _randint(high: int, generator: torch.Generator | None) -> int:
    return int(torch.randint(high, size=(), generator=generator).item())


def _sample_different(
    current: int,
    high: int,
    generator: torch.Generator | None,
) -> int:
    value = _randint(high - 1, generator)
    if value >= current:
        value += 1
    return value


def _paint_marker(image: Tensor, channel: int, y: int, x: int, patch: int) -> None:
    y0 = y * patch
    x0 = x * patch
    image[channel, y0 : y0 + patch, x0 : x0 + patch] = 1.0


def _paint_gray_marker(image: Tensor, y: int, x: int, patch: int) -> None:
    y0 = y * patch
    x0 = x * patch
    image[:, y0 : y0 + patch, x0 : x0 + patch] = 0.45


def _sample_unused_position(
    grid_size: int,
    used: set[tuple[int, int]],
    generator: torch.Generator | None,
) -> tuple[int, int]:
    for _ in range(grid_size * grid_size * 2):
        position = (_randint(grid_size, generator), _randint(grid_size, generator))
        if position not in used:
            used.add(position)
            return position
    raise RuntimeError("could not sample an unused grid position")


def _sample_unused_axis_value(
    current: int,
    high: int,
    generator: torch.Generator | None,
) -> int:
    return _sample_different(current, high, generator)
