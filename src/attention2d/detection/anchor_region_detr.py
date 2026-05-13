"""Tiny DETR-style detector using 2D local or anchor-region features."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..modules import Coordinate2DEncoding, PatchEmbed2D
from ..model import AnchorOnlyMemoryReadBlock, LocalMixBlock
from .heads import DetectionHead, LearnedObjectQueries


class TinyAnchorRegionDETR(nn.Module):
    """Small detector scaffold for testing feature/query anchor controls."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        patch_size: int = 4,
        num_classes: int = 3,
        num_queries: int = 8,
        local_blocks: int = 2,
        feature_mode: str = "anchor",
        query_init: str = "learned",
        query_refine: str = "none",
    ) -> None:
        super().__init__()
        if feature_mode not in {"local", "anchor"}:
            raise ValueError("feature_mode must be 'local' or 'anchor'")
        if query_init not in {"learned", "anchor", "anchor_detached"}:
            raise ValueError("query_init must be 'learned', 'anchor', or 'anchor_detached'")
        if query_refine not in {"none", "mask_pool", "mask_bias"}:
            raise ValueError("query_refine must be 'none', 'mask_pool', or 'mask_bias'")
        self.feature_mode = feature_mode
        self.query_init = query_init
        self.query_refine = query_refine
        self.num_queries = num_queries
        self.patch_embed = PatchEmbed2D(in_channels, embed_dim, patch_size)
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        self.local_blocks = nn.ModuleList(LocalMixBlock(embed_dim) for _ in range(local_blocks))
        self.anchor_block = AnchorOnlyMemoryReadBlock(embed_dim)
        self.learned_queries = LearnedObjectQueries(num_queries, embed_dim)
        if query_refine in {"mask_pool", "mask_bias"}:
            self.query_mask_head = nn.Conv2d(embed_dim, 1, kernel_size=1)
            self.query_mask_gate = nn.Parameter(torch.tensor(0.1))
        if query_refine == "mask_pool":
            self.query_mask_proj = nn.Linear(embed_dim, embed_dim)
        self.query_decoder = SimpleCrossAttentionDecoder(embed_dim)
        self.head = DetectionHead(embed_dim, num_classes=num_classes)

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        state = self.coord_encoding(self.patch_embed(x))
        memories = [state]
        for block in self.local_blocks:
            state = block(state)
            memories.append(state)

        routing_maps: list[Tensor] = []
        if self.feature_mode == "anchor":
            spatial_state, anchor_routing = self.anchor_block(memories)
            memories.append(spatial_state)
            routing_maps.append(anchor_routing)
        else:
            spatial_state = state

        spatial_tokens = spatial_state.flatten(2).transpose(1, 2)
        if self.query_init == "anchor":
            queries = anchor_queries_from_state(spatial_state, self.num_queries)
        elif self.query_init == "anchor_detached":
            queries = anchor_queries_from_state(spatial_state.detach(), self.num_queries)
        else:
            queries = self.learned_queries(x.shape[0])
        query_mask_logits = None
        if self.query_refine == "mask_pool":
            query_mask_logits = self.query_mask_head(spatial_state).squeeze(1)
            region = mask_pooled_region(spatial_state, query_mask_logits)
            queries = queries + self.query_mask_gate * self.query_mask_proj(region).unsqueeze(1)
        attention_bias = None
        if self.query_refine == "mask_bias":
            query_mask_logits = self.query_mask_head(spatial_state).squeeze(1)
            attention_bias = mask_attention_bias(query_mask_logits, self.query_mask_gate)
        decoded = self.query_decoder(queries, spatial_tokens, attention_bias=attention_bias)
        outputs = self.head(decoded)
        if query_mask_logits is not None:
            outputs["query_mask_logits"] = query_mask_logits
        outputs.update(
            {
                "spatial_features": spatial_state,
                "memories": memories,
                "routing_maps": routing_maps,
            }
        )
        return outputs


class SimpleCrossAttentionDecoder(nn.Module):
    """MPS-friendly one-block cross-attention decoder for tiny detection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.scale = dim**-0.5

    def forward(self, queries: Tensor, memory: Tensor, attention_bias: Tensor | None = None) -> Tensor:
        norm_queries = self.query_norm(queries)
        norm_memory = self.memory_norm(memory)
        q = self.query_proj(norm_queries)
        k = self.key_proj(norm_memory)
        v = self.value_proj(norm_memory)
        attention = torch.bmm(q, k.transpose(1, 2)) * self.scale
        if attention_bias is not None:
            attention = attention + attention_bias
        weights = attention.softmax(dim=-1)
        readout = torch.bmm(weights, v)
        decoded = queries + self.out_proj(readout)
        return decoded + self.mlp(decoded)


def anchor_queries_from_state(state: Tensor, num_queries: int) -> Tensor:
    """Create deterministic row/column/global query seeds from a 2D state."""

    batch, channels, height, width = state.shape
    row_tokens = state.mean(dim=3).transpose(1, 2)
    col_tokens = state.mean(dim=2).transpose(1, 2)
    global_token = state.mean(dim=(2, 3)).unsqueeze(1)
    tokens = torch.cat((row_tokens, col_tokens, global_token), dim=1)
    if tokens.shape[1] >= num_queries:
        positions = torch.linspace(
            0,
            tokens.shape[1] - 1,
            num_queries,
            device=state.device,
        ).round().long()
        return tokens[:, positions]
    repeats = (num_queries + tokens.shape[1] - 1) // tokens.shape[1]
    return tokens.repeat(1, repeats, 1)[:, :num_queries]


def mask_pooled_region(state: Tensor, mask_logits: Tensor) -> Tensor:
    """Pool a foreground region summary from a 2D state using soft mask logits."""

    weights = mask_logits.sigmoid().unsqueeze(1)
    numerator = (state * weights).sum(dim=(2, 3))
    denominator = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    return numerator / denominator


def mask_attention_bias(mask_logits: Tensor, gate: Tensor) -> Tensor:
    """Convert dense foreground logits into a soft attention bias over spatial tokens."""

    bias = torch.nn.functional.logsigmoid(mask_logits).flatten(1).unsqueeze(1)
    return gate * bias
