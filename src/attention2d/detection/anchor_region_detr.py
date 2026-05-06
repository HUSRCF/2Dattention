"""Tiny DETR-style detector using 2D local or anchor-region features."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..modules import Coordinate2DEncoding, PatchEmbed2D
from ..model import AnchorOnlyMemoryReadBlock, LocalMixBlock
from .heads import DetectionHead, LearnedObjectQueries


class TinyAnchorRegionDETR(nn.Module):
    """Small detector scaffold for testing anchor-region query initialization."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        patch_size: int = 4,
        num_classes: int = 3,
        num_queries: int = 8,
        local_blocks: int = 2,
        query_init: str = "learned",
    ) -> None:
        super().__init__()
        if query_init not in {"learned", "anchor"}:
            raise ValueError("query_init must be 'learned' or 'anchor'")
        self.query_init = query_init
        self.num_queries = num_queries
        self.patch_embed = PatchEmbed2D(in_channels, embed_dim, patch_size)
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        self.local_blocks = nn.ModuleList(LocalMixBlock(embed_dim) for _ in range(local_blocks))
        self.anchor_block = AnchorOnlyMemoryReadBlock(embed_dim)
        self.learned_queries = LearnedObjectQueries(num_queries, embed_dim)
        self.query_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
        )
        self.head = DetectionHead(embed_dim, num_classes=num_classes)

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        state = self.coord_encoding(self.patch_embed(x))
        memories = [state]
        for block in self.local_blocks:
            state = block(state)
            memories.append(state)
        anchor_state, anchor_routing = self.anchor_block(memories)
        memories.append(anchor_state)

        spatial_tokens = anchor_state.flatten(2).transpose(1, 2)
        if self.query_init == "anchor":
            queries = anchor_queries_from_state(anchor_state, self.num_queries)
        else:
            queries = self.learned_queries(x.shape[0])
        decoded = self.query_decoder(tgt=queries, memory=spatial_tokens)
        outputs = self.head(decoded)
        outputs.update(
            {
                "spatial_features": anchor_state,
                "memories": memories,
                "routing_maps": [anchor_routing],
            }
        )
        return outputs


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
