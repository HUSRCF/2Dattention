"""Tiny end-to-end model for validating the proposed information flow."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .modules import (
    AxisAnchorMemoryRead,
    Coordinate2DEncoding,
    LatticeMemoryRead,
    PatchEmbed2D,
    SemanticGraphMemoryRead,
    SpatialPrefill2D,
    default_offsets,
)


class MemoryReadBlock(nn.Module):
    """A small refinement block that appends its output to the memory pool."""

    def __init__(self, dim: int, gate_init: float = 1e-3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.read = LatticeMemoryRead(dim=dim, offsets=default_offsets())
        self.read_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.mix = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, memories: list[Tensor]) -> tuple[Tensor, Tensor]:
        readout, routing = self.read(memories)
        self._record_memory_stats(readout=readout, state=memories[-1])
        state = memories[-1] + self.read_gate * readout
        state = state + self.mix(self.norm(state))
        return state, routing

    def _record_memory_stats(self, readout: Tensor, state: Tensor) -> None:
        read_norm = readout.detach().norm(dim=1).mean()
        state_norm = state.detach().norm(dim=1).mean().clamp_min(1e-8)
        self.last_gate = float(self.read_gate.detach().cpu())
        self.last_read_norm = float(read_norm.cpu())
        self.last_state_norm = float(state_norm.cpu())
        self.last_scaled_read_ratio = float(
            (self.read_gate.detach().abs().cpu() * read_norm.cpu()) / state_norm.cpu()
        )


class AnchorMemoryReadBlock(nn.Module):
    """Combine local lattice reads with row/column/global anchor reads."""

    def __init__(self, dim: int, gate_init: float = 1e-3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.lattice_read = LatticeMemoryRead(dim=dim, offsets=default_offsets())
        self.anchor_read = AxisAnchorMemoryRead(dim=dim)
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.read_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.mix = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, memories: list[Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        lattice, lattice_routing = self.lattice_read(memories)
        anchor, anchor_routing = self.anchor_read(memories)
        readout = self.fuse(torch.cat((lattice, anchor), dim=1))
        self._record_memory_stats(readout=readout, state=memories[-1])
        state = memories[-1] + self.read_gate * readout
        state = state + self.mix(self.norm(state))
        return state, {
            "lattice": lattice_routing,
            "anchor": anchor_routing,
        }

    def _record_memory_stats(self, readout: Tensor, state: Tensor) -> None:
        read_norm = readout.detach().norm(dim=1).mean()
        state_norm = state.detach().norm(dim=1).mean().clamp_min(1e-8)
        self.last_gate = float(self.read_gate.detach().cpu())
        self.last_read_norm = float(read_norm.cpu())
        self.last_state_norm = float(state_norm.cpu())
        self.last_scaled_read_ratio = float(
            (self.read_gate.detach().abs().cpu() * read_norm.cpu()) / state_norm.cpu()
        )


class AnchorOnlyMemoryReadBlock(nn.Module):
    """Use row/column/global anchor reads without lattice offset candidates."""

    def __init__(self, dim: int, gate_init: float = 1e-3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.anchor_read = AxisAnchorMemoryRead(dim=dim)
        self.read_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.mix = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, memories: list[Tensor]) -> tuple[Tensor, Tensor]:
        readout, routing = self.anchor_read(memories)
        self._record_memory_stats(readout=readout, state=memories[-1])
        state = memories[-1] + self.read_gate * readout
        state = state + self.mix(self.norm(state))
        return state, routing

    def _record_memory_stats(self, readout: Tensor, state: Tensor) -> None:
        read_norm = readout.detach().norm(dim=1).mean()
        state_norm = state.detach().norm(dim=1).mean().clamp_min(1e-8)
        self.last_gate = float(self.read_gate.detach().cpu())
        self.last_read_norm = float(read_norm.cpu())
        self.last_state_norm = float(state_norm.cpu())
        self.last_scaled_read_ratio = float(
            (self.read_gate.detach().abs().cpu() * read_norm.cpu()) / state_norm.cpu()
        )


class SemanticGraphMemoryReadBlock(nn.Module):
    """Combine lattice, anchor, and semantic graph memory reads."""

    def __init__(self, dim: int, graph_k: int = 4, gate_init: float = 1e-3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.lattice_read = LatticeMemoryRead(dim=dim, offsets=default_offsets())
        self.anchor_read = AxisAnchorMemoryRead(dim=dim)
        self.graph_read = SemanticGraphMemoryRead(dim=dim, k=graph_k)
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.read_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.mix = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, memories: list[Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        lattice, lattice_routing = self.lattice_read(memories)
        anchor, anchor_routing = self.anchor_read(memories)
        graph, graph_routing = self.graph_read(memories)
        readout = self.fuse(torch.cat((lattice, anchor, graph), dim=1))
        self._record_memory_stats(readout=readout, state=memories[-1])
        state = memories[-1] + self.read_gate * readout
        state = state + self.mix(self.norm(state))
        return state, {
            "lattice": lattice_routing,
            "anchor": anchor_routing,
            "graph": graph_routing,
        }

    def _record_memory_stats(self, readout: Tensor, state: Tensor) -> None:
        read_norm = readout.detach().norm(dim=1).mean()
        state_norm = state.detach().norm(dim=1).mean().clamp_min(1e-8)
        self.last_gate = float(self.read_gate.detach().cpu())
        self.last_read_norm = float(read_norm.cpu())
        self.last_state_norm = float(state_norm.cpu())
        self.last_scaled_read_ratio = float(
            (self.read_gate.detach().abs().cpu() * read_norm.cpu()) / state_norm.cpu()
        )


class TinyPrefillLatticeAttnRes(nn.Module):
    """Minimal image model with 2D prefill and spatial-depth memory reads."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        patch_size: int = 4,
        num_classes: int = 10,
        prefill_rounds: int = 2,
        read_blocks: int = 2,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        self.prefill = SpatialPrefill2D(dim=embed_dim, rounds=prefill_rounds)
        self.read_blocks = nn.ModuleList(
            MemoryReadBlock(embed_dim, gate_init=gate_init) for _ in range(read_blocks)
        )
        self.head = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        memories = self.prefill(self.coord_encoding(self.patch_embed(x)))
        routing_maps: list[Tensor] = []

        for block in self.read_blocks:
            state, routing = block(memories)
            memories.append(state)
            routing_maps.append(routing)

        logits = self.head(memories[-1])
        return {
            "logits": logits,
            "memories": memories,
            "routing_maps": routing_maps,
        }


class LocalMixBlock(nn.Module):
    """Local refinement block without any memory read path."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.mix = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, state: Tensor) -> Tensor:
        return state + self.mix(self.norm(state))


class TinyPrefillLocalMixClassifier(nn.Module):
    """Prefill backbone plus local mix blocks, with no AttnRes memory reads."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        patch_size: int = 4,
        num_classes: int = 10,
        prefill_rounds: int = 2,
        read_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        self.prefill = SpatialPrefill2D(dim=embed_dim, rounds=prefill_rounds)
        self.blocks = nn.ModuleList(LocalMixBlock(embed_dim) for _ in range(read_blocks))
        self.head = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        memories = self.prefill(self.coord_encoding(self.patch_embed(x)))
        state = memories[-1]
        for block in self.blocks:
            state = block(state)
            memories.append(state)
        logits = self.head(state)
        return {
            "logits": logits,
            "memories": memories,
        }


class XAttnResReadBlock(nn.Module):
    """XAttnRes-style history read without explicit 2D offset candidates."""

    def __init__(self, dim: int, gate_init: float = 1e-3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.query = nn.Parameter(torch.zeros(dim))
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.read_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.mix = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, memories: list[Tensor]) -> tuple[Tensor, Tensor]:
        query = self.query.view(1, -1, 1, 1)
        logits = []
        for memory in memories:
            logits.append((self.norm(memory) * query).sum(dim=1))
        routing = nn.functional.softmax(torch.stack(logits, dim=1), dim=1)
        stacked = torch.stack(memories, dim=1)
        readout = (stacked * routing.unsqueeze(2)).sum(dim=1)
        projected = self.value_proj(readout)
        self._record_memory_stats(readout=projected, state=memories[-1])
        state = memories[-1] + self.read_gate * projected
        state = state + self.mix(self.norm(state))
        return state, routing

    def _record_memory_stats(self, readout: Tensor, state: Tensor) -> None:
        read_norm = readout.detach().norm(dim=1).mean()
        state_norm = state.detach().norm(dim=1).mean().clamp_min(1e-8)
        self.last_gate = float(self.read_gate.detach().cpu())
        self.last_read_norm = float(read_norm.cpu())
        self.last_state_norm = float(state_norm.cpu())
        self.last_scaled_read_ratio = float(
            (self.read_gate.detach().abs().cpu() * read_norm.cpu()) / state_norm.cpu()
        )


class TinyXAttnResClassifier(nn.Module):
    """Cross-stage/history attention residual baseline without lattice offsets."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        patch_size: int = 4,
        num_classes: int = 10,
        prefill_rounds: int = 2,
        read_blocks: int = 2,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        self.prefill = SpatialPrefill2D(dim=embed_dim, rounds=prefill_rounds)
        self.read_blocks = nn.ModuleList(
            XAttnResReadBlock(embed_dim, gate_init=gate_init)
            for _ in range(read_blocks)
        )
        self.head = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        memories = self.prefill(self.coord_encoding(self.patch_embed(x)))
        routing_maps: list[Tensor] = []
        for block in self.read_blocks:
            state, routing = block(memories)
            memories.append(state)
            routing_maps.append(routing)

        logits = self.head(memories[-1])
        return {
            "logits": logits,
            "memories": memories,
            "routing_maps": routing_maps,
        }


class TinyAnchorPrefillLatticeAttnRes(nn.Module):
    """Tiny model with extra row/column/global anchor memory reads."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        patch_size: int = 4,
        num_classes: int = 10,
        prefill_rounds: int = 2,
        read_blocks: int = 2,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        self.prefill = SpatialPrefill2D(dim=embed_dim, rounds=prefill_rounds)
        self.read_blocks = nn.ModuleList(
            AnchorMemoryReadBlock(embed_dim, gate_init=gate_init)
            for _ in range(read_blocks)
        )
        self.head = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor] | list[dict[str, Tensor]]]:
        memories = self.prefill(self.coord_encoding(self.patch_embed(x)))
        routing_maps: list[dict[str, Tensor]] = []

        for block in self.read_blocks:
            state, routing = block(memories)
            memories.append(state)
            routing_maps.append(routing)

        logits = self.head(memories[-1])
        return {
            "logits": logits,
            "memories": memories,
            "routing_maps": routing_maps,
        }


class TinyAnchorOnlyAttnResClassifier(nn.Module):
    """Tiny model with prefill plus anchor reads, but no lattice read branch."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        patch_size: int = 4,
        num_classes: int = 10,
        prefill_rounds: int = 2,
        read_blocks: int = 2,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        self.prefill = SpatialPrefill2D(dim=embed_dim, rounds=prefill_rounds)
        self.read_blocks = nn.ModuleList(
            AnchorOnlyMemoryReadBlock(embed_dim, gate_init=gate_init)
            for _ in range(read_blocks)
        )
        self.head = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        memories = self.prefill(self.coord_encoding(self.patch_embed(x)))
        routing_maps: list[Tensor] = []

        for block in self.read_blocks:
            state, routing = block(memories)
            memories.append(state)
            routing_maps.append(routing)

        logits = self.head(memories[-1])
        return {
            "logits": logits,
            "memories": memories,
            "routing_maps": routing_maps,
        }


class TinyGraphPrefillLatticeAttnRes(nn.Module):
    """Tiny model with lattice, anchor, and semantic graph memory reads."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 48,
        patch_size: int = 4,
        num_classes: int = 10,
        prefill_rounds: int = 2,
        read_blocks: int = 2,
        graph_k: int = 4,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        self.prefill = SpatialPrefill2D(dim=embed_dim, rounds=prefill_rounds)
        self.read_blocks = nn.ModuleList(
            SemanticGraphMemoryReadBlock(
                embed_dim,
                graph_k=graph_k,
                gate_init=gate_init,
            )
            for _ in range(read_blocks)
        )
        self.head = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self,
        x: Tensor,
    ) -> dict[str, Tensor | list[Tensor] | list[dict[str, Tensor]]]:
        memories = self.prefill(self.coord_encoding(self.patch_embed(x)))
        routing_maps: list[dict[str, Tensor]] = []

        for block in self.read_blocks:
            state, routing = block(memories)
            memories.append(state)
            routing_maps.append(routing)

        logits = self.head(memories[-1])
        return {
            "logits": logits,
            "memories": memories,
            "routing_maps": routing_maps,
        }
