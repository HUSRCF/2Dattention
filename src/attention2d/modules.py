"""Core building blocks for the 2D prefill + lattice AttnRes demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Offset2D:
    """A 2D memory displacement in patch-grid coordinates."""

    dy: int
    dx: int


def default_offsets(include_dilated: bool = True) -> tuple[Offset2D, ...]:
    """Return a compact 2D support set for lattice memory reads."""

    local = [Offset2D(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    if not include_dilated:
        return tuple(local)
    dilated = [
        Offset2D(-2, -2),
        Offset2D(-2, 2),
        Offset2D(2, -2),
        Offset2D(2, 2),
    ]
    return tuple(local + dilated)


def shift_2d(x: Tensor, offset: Offset2D) -> Tensor:
    """Shift without wraparound so each location reads a displaced 2D neighbor."""

    if x.ndim != 4:
        raise ValueError(f"expected [B, C, H, W], got shape {tuple(x.shape)}")

    dy, dx = offset.dy, offset.dx
    if dy == 0 and dx == 0:
        return x

    batch, channels, height, width = x.shape
    out = x.new_zeros(batch, channels, height, width)

    src_y0 = max(0, dy)
    src_y1 = height + min(0, dy)
    src_x0 = max(0, dx)
    src_x1 = width + min(0, dx)

    dst_y0 = max(0, -dy)
    dst_y1 = height - max(0, dy)
    dst_x0 = max(0, -dx)
    dst_x1 = width - max(0, dx)

    out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = x[
        :, :, src_y0:src_y1, src_x0:src_x1
    ]
    return out


class PatchEmbed2D(nn.Module):
    """Patchify an image while preserving a 2D feature lattice."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class Coordinate2DEncoding(nn.Module):
    """Project normalized 2D coordinates into the feature lattice."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(2, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B, C, H, W], got shape {tuple(x.shape)}")
        batch, _, height, width = x.shape
        ys = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((yy, xx), dim=0).expand(batch, -1, -1, -1)
        return x + self.proj(coords)


class Local2DUpdate(nn.Module):
    """Lightweight local message-passing block used during prefill."""

    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.norm = nn.GroupNorm(1, dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.depthwise(self.norm(x))
        y = self.pointwise(y)
        return x + y


class SpatialPrefill2D(nn.Module):
    """Build a full-image 2D memory list before later selective reads."""

    def __init__(self, dim: int, rounds: int = 2) -> None:
        super().__init__()
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        self.rounds = rounds
        self.updates = nn.ModuleList(Local2DUpdate(dim) for _ in range(rounds))

    def forward(self, x: Tensor) -> list[Tensor]:
        memories = [x]
        state = x
        for update in self.updates:
            state = update(state)
            memories.append(state)
        return memories


class LatticeMemoryRead(nn.Module):
    """AttnRes-style read over previous 2D memories and spatial offsets."""

    def __init__(
        self,
        dim: int,
        offsets: Iterable[Offset2D] | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.offsets = tuple(offsets or default_offsets())
        if not self.offsets:
            raise ValueError("at least one offset is required")

        self.key_norm = nn.GroupNorm(1, dim)
        self.query = nn.Parameter(torch.zeros(dim))
        self.offset_bias = nn.Parameter(torch.zeros(len(self.offsets)))
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, memories: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        if not memories:
            raise ValueError("memories must contain at least one feature map")

        reference_shape = memories[0].shape
        if len(reference_shape) != 4:
            raise ValueError("memory tensors must have shape [B, C, H, W]")
        for idx, memory in enumerate(memories):
            if memory.shape != reference_shape:
                raise ValueError(
                    "all memory tensors must share shape "
                    f"{tuple(reference_shape)}, got {tuple(memory.shape)} at {idx}"
                )

        candidates: list[Tensor] = []
        logits: list[Tensor] = []
        query = self.query.view(1, self.dim, 1, 1)

        for memory in memories:
            for offset_idx, offset in enumerate(self.offsets):
                shifted = shift_2d(memory, offset)
                key = self.key_norm(shifted)
                logit = (key * query).sum(dim=1)
                logit = logit + self.offset_bias[offset_idx]
                candidates.append(shifted)
                logits.append(logit)

        candidate_tensor = torch.stack(candidates, dim=1)
        logit_tensor = torch.stack(logits, dim=1)
        routing = F.softmax(logit_tensor, dim=1)
        fused = (candidate_tensor * routing.unsqueeze(2)).sum(dim=1)
        return self.value_proj(fused), routing


class GroupedLatticeMemoryRead(nn.Module):
    """Read independent channel subspaces from the same memory candidates.

    Each group owns its query, routing distribution, and value projection. The
    groups therefore cannot select different sources through a shared channel
    scalar, while the concatenated output keeps the original ``[B, C, H, W]``
    interface.
    """

    def __init__(
        self,
        dim: int,
        groups: int = 2,
        offsets: Iterable[Offset2D] | None = None,
    ) -> None:
        super().__init__()
        if groups < 2:
            raise ValueError("groups must be >= 2")
        if dim % groups:
            raise ValueError(f"dim={dim} must be divisible by groups={groups}")
        self.dim = dim
        self.groups = groups
        group_dim = dim // groups
        self.reads = nn.ModuleList(
            LatticeMemoryRead(dim=group_dim, offsets=offsets)
            for _ in range(groups)
        )

    def forward(self, memories: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        if not memories:
            raise ValueError("memories must contain at least one feature map")
        outputs: list[Tensor] = []
        routings: list[Tensor] = []
        for group_idx, reader in enumerate(self.reads):
            start = group_idx * reader.dim
            stop = start + reader.dim
            group_memories = [memory[:, start:stop] for memory in memories]
            output, routing = reader(group_memories)
            outputs.append(output)
            routings.append(routing)
        return torch.cat(outputs, dim=1), torch.stack(routings, dim=1), torch.stack(outputs, dim=1)


class BlockDiagonalLatticeMemoryRead(nn.Module):
    """Shared candidate routing with block-diagonal value projections."""

    def __init__(self, dim: int, groups: int = 2, offsets: Iterable[Offset2D] | None = None) -> None:
        super().__init__()
        if groups < 2:
            raise ValueError("groups must be >= 2")
        if dim % groups:
            raise ValueError(f"dim={dim} must be divisible by groups={groups}")
        self.dim = dim
        self.groups = groups
        self.offsets = tuple(offsets or default_offsets())
        self.key_norm = nn.GroupNorm(1, dim)
        self.query = nn.Parameter(torch.zeros(dim))
        self.offset_bias = nn.Parameter(torch.zeros(len(self.offsets)))
        group_dim = dim // groups
        self.value_projs = nn.ModuleList(
            nn.Conv2d(group_dim, group_dim, kernel_size=1) for _ in range(groups)
        )

    def forward(self, memories: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        if not memories:
            raise ValueError("memories must contain at least one feature map")
        reference_shape = memories[0].shape
        if len(reference_shape) != 4:
            raise ValueError("memory tensors must have shape [B, C, H, W]")
        for idx, memory in enumerate(memories):
            if memory.shape != reference_shape:
                raise ValueError(f"all memory tensors must share shape {tuple(reference_shape)}, got {tuple(memory.shape)} at {idx}")
        query = self.query.view(1, self.dim, 1, 1)
        candidates: list[Tensor] = []
        logits: list[Tensor] = []
        for memory in memories:
            for offset_idx, offset in enumerate(self.offsets):
                shifted = shift_2d(memory, offset)
                logit = (self.key_norm(shifted) * query).sum(dim=1) + self.offset_bias[offset_idx]
                candidates.append(shifted)
                logits.append(logit)
        candidate_tensor = torch.stack(candidates, dim=1)
        routing = F.softmax(torch.stack(logits, dim=1), dim=1)
        fused = (candidate_tensor * routing.unsqueeze(2)).sum(dim=1)
        outputs = []
        private = []
        group_dim = self.dim // self.groups
        for group_idx, value_proj in enumerate(self.value_projs):
            start = group_idx * group_dim
            output = value_proj(fused[:, start : start + group_dim])
            outputs.append(output)
            private.append(output)
        return torch.cat(outputs, dim=1), routing, torch.stack(private, dim=1)


def cross_group_decorrelation_loss(expert_outputs: Tensor, eps: float = 1e-6) -> Tensor:
    """Penalize linear redundancy between private expert outputs.

    ``expert_outputs`` must have shape ``[B, E, D, H, W]``. Statistics are
    computed over corresponding batch/spatial samples and do not constrain
    channels within one expert.
    """

    if expert_outputs.ndim != 5:
        raise ValueError("expert_outputs must have shape [B, E, D, H, W]")
    _, groups, _, _, _ = expert_outputs.shape
    if groups < 2:
        return expert_outputs.new_zeros(())
    flattened = expert_outputs.permute(1, 0, 3, 4, 2).reshape(groups, -1, expert_outputs.shape[2])
    normalized = flattened - flattened.mean(dim=1, keepdim=True)
    normalized = normalized / normalized.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    losses = []
    for left in range(groups):
        for right in range(left + 1, groups):
            covariance = normalized[left].transpose(0, 1) @ normalized[right]
            covariance = covariance / max(normalized.shape[1] - 1, 1)
            losses.append(covariance.square().mean())
    return torch.stack(losses).mean()


def variance_floor_loss(expert_outputs: Tensor, floor: float = 1.0, eps: float = 1e-6) -> Tensor:
    """Keep private expert dimensions from collapsing to constants."""

    if expert_outputs.ndim != 5:
        raise ValueError("expert_outputs must have shape [B, E, D, H, W]")
    flattened = expert_outputs.permute(1, 0, 3, 4, 2).reshape(
        expert_outputs.shape[1], -1, expert_outputs.shape[2]
    )
    std = flattened.std(dim=1, unbiased=False)
    return F.relu(float(floor) - torch.sqrt(std.square() + eps)).mean()


class AxisAnchorMemoryRead(nn.Module):
    """Read row, column, and global anchor summaries from the memory pool."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.key_norm = nn.GroupNorm(1, dim)
        self.query = nn.Parameter(torch.zeros(dim))
        self.source_bias = nn.Parameter(torch.zeros(3))
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, memories: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        if not memories:
            raise ValueError("memories must contain at least one feature map")

        reference_shape = memories[0].shape
        if len(reference_shape) != 4:
            raise ValueError("memory tensors must have shape [B, C, H, W]")
        for idx, memory in enumerate(memories):
            if memory.shape != reference_shape:
                raise ValueError(
                    "all memory tensors must share shape "
                    f"{tuple(reference_shape)}, got {tuple(memory.shape)} at {idx}"
                )

        _, _, height, width = reference_shape
        candidates: list[Tensor] = []
        logits: list[Tensor] = []
        query = self.query.view(1, self.dim, 1, 1)

        for memory in memories:
            row_anchor = memory.mean(dim=3, keepdim=True).expand(-1, -1, -1, width)
            col_anchor = memory.mean(dim=2, keepdim=True).expand(-1, -1, height, -1)
            global_anchor = memory.mean(dim=(2, 3), keepdim=True).expand(
                -1,
                -1,
                height,
                width,
            )

            for source_idx, candidate in enumerate(
                (row_anchor, col_anchor, global_anchor)
            ):
                key = self.key_norm(candidate)
                logit = (key * query).sum(dim=1)
                logit = logit + self.source_bias[source_idx]
                candidates.append(candidate)
                logits.append(logit)

        candidate_tensor = torch.stack(candidates, dim=1)
        logit_tensor = torch.stack(logits, dim=1)
        routing = F.softmax(logit_tensor, dim=1)
        fused = (candidate_tensor * routing.unsqueeze(2)).sum(dim=1)
        return self.value_proj(fused), routing


class SemanticGraphMemoryRead(nn.Module):
    """Read feature-similar nonlocal neighbors through a dense top-k graph."""

    def __init__(self, dim: int, k: int = 4) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1")
        self.dim = dim
        self.k = k
        self.key_norm = nn.GroupNorm(1, dim)
        self.query = nn.Parameter(torch.zeros(dim))
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, memories: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        if not memories:
            raise ValueError("memories must contain at least one feature map")

        reference_shape = memories[0].shape
        if len(reference_shape) != 4:
            raise ValueError("memory tensors must have shape [B, C, H, W]")
        for idx, memory in enumerate(memories):
            if memory.shape != reference_shape:
                raise ValueError(
                    "all memory tensors must share shape "
                    f"{tuple(reference_shape)}, got {tuple(memory.shape)} at {idx}"
                )

        batch, channels, height, width = reference_shape
        num_nodes = height * width
        k = min(self.k, max(1, num_nodes - 1))
        candidates: list[Tensor] = []
        logits: list[Tensor] = []
        query = self.query.view(1, channels, 1, 1)

        for memory in memories:
            normalized = self.key_norm(memory).flatten(2).transpose(1, 2)
            normalized = F.normalize(normalized, dim=-1)
            values = memory.flatten(2).transpose(1, 2)

            similarity = normalized @ normalized.transpose(1, 2)
            eye = torch.eye(num_nodes, device=memory.device, dtype=torch.bool)
            similarity = similarity.masked_fill(eye.unsqueeze(0), -torch.inf)
            top_values, top_indices = similarity.topk(k=k, dim=-1)

            gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, channels)
            expanded_values = values.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            neighbors = torch.gather(expanded_values, dim=2, index=gather_indices)
            neighbor_weights = F.softmax(top_values, dim=-1).unsqueeze(-1)
            graph_feature = (neighbors * neighbor_weights).sum(dim=2)
            graph_feature = graph_feature.transpose(1, 2).reshape(
                batch,
                channels,
                height,
                width,
            )

            key = self.key_norm(graph_feature)
            logits.append((key * query).sum(dim=1))
            candidates.append(graph_feature)

        candidate_tensor = torch.stack(candidates, dim=1)
        logit_tensor = torch.stack(logits, dim=1)
        routing = F.softmax(logit_tensor, dim=1)
        fused = (candidate_tensor * routing.unsqueeze(2)).sum(dim=1)
        return self.value_proj(fused), routing
