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
        query_mask_gate_init: float = 0.1,
        quality_mode: str = "query",
        class_mode: str = "query",
    ) -> None:
        super().__init__()
        if feature_mode not in {"local", "anchor"}:
            raise ValueError("feature_mode must be 'local' or 'anchor'")
        if query_init not in {
            "learned",
            "anchor",
            "anchor_detached",
            "anchor_residual",
            "anchor_residual_detached",
            "grid",
            "grid_residual",
            "grid_residual_detached",
            "mask_proposal",
            "mask_proposal_nms",
            "mask_proposal_residual",
            "mask_proposal_residual_nms",
            "mask_proposal_oracle",
            "mask_proposal_oracle_nms",
        }:
            raise ValueError(
                "query_init must be a supported learned, anchor, residual-anchor, "
                "grid-anchor, mask-proposal, or oracle mask-proposal mode"
            )
        proposal_refine_modes = {
            "proposal_decode2",
            "proposal_reinject",
            "proposal_persistent",
            "proposal_late_persistent",
            "proposal_persistent_stopgrad",
        }
        if query_refine not in {
            "none",
            "mask_pool",
            "mask_bias",
            "query_mask",
            "query_mask_refine",
            *proposal_refine_modes,
        }:
            raise ValueError(
                "query_refine must be 'none', 'mask_pool', 'mask_bias', 'query_mask', "
                "'query_mask_refine', "
                "'proposal_decode2', 'proposal_reinject', 'proposal_persistent', "
                "'proposal_late_persistent', or 'proposal_persistent_stopgrad'"
            )
        self.feature_mode = feature_mode
        self.query_init = query_init
        self.query_refine = query_refine
        self.num_queries = num_queries
        self.register_buffer("query_mask_gate_scale", torch.tensor(1.0))
        self.patch_embed = PatchEmbed2D(in_channels, embed_dim, patch_size)
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        self.local_blocks = nn.ModuleList(LocalMixBlock(embed_dim) for _ in range(local_blocks))
        self.anchor_block = AnchorOnlyMemoryReadBlock(embed_dim)
        self.learned_queries = LearnedObjectQueries(num_queries, embed_dim)
        if query_init in {
            "mask_proposal",
            "mask_proposal_nms",
            "mask_proposal_residual",
            "mask_proposal_residual_nms",
        } or query_refine in {"mask_pool", "mask_bias"}:
            self.query_mask_head = nn.Conv2d(embed_dim, 1, kernel_size=1)
        if query_init in {
            "anchor_residual",
            "anchor_residual_detached",
            "grid_residual",
            "grid_residual_detached",
            "mask_proposal_residual",
            "mask_proposal_residual_nms",
        }:
            self.anchor_query_gate = nn.Parameter(torch.tensor(float(query_mask_gate_init)))
        if query_refine in {"mask_pool", "mask_bias"}:
            self.query_mask_gate = nn.Parameter(torch.tensor(float(query_mask_gate_init)))
        if query_refine == "mask_pool":
            self.query_mask_proj = nn.Linear(embed_dim, embed_dim)
        if query_refine in {"query_mask", "query_mask_refine"}:
            self.query_mask_query_proj = nn.Linear(embed_dim, embed_dim)
            self.query_mask_feature_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        if query_refine == "query_mask_refine":
            self.query_box_refine_gate = nn.Parameter(torch.tensor(float(query_mask_gate_init)))
        if query_refine in proposal_refine_modes:
            self.proposal_state_gate = nn.Parameter(torch.tensor(float(query_mask_gate_init)))
        if query_refine in {"proposal_persistent", "proposal_late_persistent", "proposal_persistent_stopgrad"}:
            self.proposal_state_update = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        self.query_decoder = SimpleCrossAttentionDecoder(embed_dim)
        self.head = DetectionHead(
            embed_dim,
            num_classes=num_classes,
            quality_mode=quality_mode,
            class_mode=class_mode,
        )

    def set_query_mask_gate_scale(self, scale: float) -> None:
        """Set a runtime multiplier for mask-conditioned query refinement."""

        self.query_mask_gate_scale.fill_(float(scale))

    def forward(
        self,
        x: Tensor,
        query_mask_logits_override: Tensor | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
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
        query_mask_logits = None
        query_proposal_indices = None
        proposal_queries = None
        if self.query_init in {
            "mask_proposal",
            "mask_proposal_nms",
            "mask_proposal_residual",
            "mask_proposal_residual_nms",
            "mask_proposal_oracle",
            "mask_proposal_oracle_nms",
        }:
            if self.query_init in {"mask_proposal_oracle", "mask_proposal_oracle_nms"}:
                if query_mask_logits_override is None:
                    raise ValueError("oracle mask-proposal query init requires query_mask_logits_override")
                query_mask_logits = query_mask_logits_override.to(
                    device=spatial_state.device,
                    dtype=spatial_state.dtype,
                )
            else:
                query_mask_logits = self.query_mask_head(spatial_state).squeeze(1)
            proposal_suppression_radius = (
                2
                if self.query_init
                in {"mask_proposal_nms", "mask_proposal_residual_nms", "mask_proposal_oracle_nms"}
                else 0
            )
            proposal_queries, query_proposal_indices = mask_proposal_queries_from_state(
                spatial_state,
                query_mask_logits,
                self.num_queries,
                suppression_radius=proposal_suppression_radius,
            )
            if self.query_init in {"mask_proposal_residual", "mask_proposal_residual_nms"}:
                queries = self.learned_queries(x.shape[0]) + self.anchor_query_gate * proposal_queries
            else:
                queries = proposal_queries
        elif self.query_init == "anchor":
            queries = anchor_queries_from_state(spatial_state, self.num_queries)
        elif self.query_init == "anchor_detached":
            queries = anchor_queries_from_state(spatial_state.detach(), self.num_queries)
        elif self.query_init == "anchor_residual":
            queries = self.learned_queries(x.shape[0]) + self.anchor_query_gate * anchor_queries_from_state(
                spatial_state,
                self.num_queries,
            )
        elif self.query_init == "anchor_residual_detached":
            queries = self.learned_queries(x.shape[0]) + self.anchor_query_gate * anchor_queries_from_state(
                spatial_state.detach(),
                self.num_queries,
            )
        elif self.query_init == "grid":
            queries = grid_queries_from_state(spatial_state, self.num_queries)
        elif self.query_init == "grid_residual":
            queries = self.learned_queries(x.shape[0]) + self.anchor_query_gate * grid_queries_from_state(
                spatial_state,
                self.num_queries,
            )
        elif self.query_init == "grid_residual_detached":
            queries = self.learned_queries(x.shape[0]) + self.anchor_query_gate * grid_queries_from_state(
                spatial_state.detach(),
                self.num_queries,
            )
        else:
            queries = self.learned_queries(x.shape[0])
        if self.query_refine == "mask_pool":
            query_mask_logits = self.query_mask_head(spatial_state).squeeze(1)
            region = mask_pooled_region(spatial_state, query_mask_logits)
            gate = self.query_mask_gate * self.query_mask_gate_scale
            queries = queries + gate * self.query_mask_proj(region).unsqueeze(1)
        attention_bias = None
        if self.query_refine == "mask_bias":
            query_mask_logits = self.query_mask_head(spatial_state).squeeze(1)
            gate = self.query_mask_gate * self.query_mask_gate_scale
            attention_bias = mask_attention_bias(query_mask_logits, gate)
        if self.query_refine in {
            "proposal_decode2",
            "proposal_reinject",
            "proposal_persistent",
            "proposal_late_persistent",
            "proposal_persistent_stopgrad",
        }:
            if proposal_queries is None:
                raise ValueError("proposal-state refinement requires mask-proposal query initialization")
            decoded = proposal_state_decode(
                decoder=self.query_decoder,
                queries=queries,
                proposal_queries=proposal_queries,
                spatial_tokens=spatial_tokens,
                mode=self.query_refine,
                gate=self.proposal_state_gate * self.query_mask_gate_scale,
                proposal_update=self.proposal_state_update
                if self.query_refine
                in {"proposal_persistent", "proposal_late_persistent", "proposal_persistent_stopgrad"}
                else None,
            )
        else:
            decoded = self.query_decoder(queries, spatial_tokens, attention_bias=attention_bias)
        outputs = self.head(decoded)
        if self.query_refine in {"query_mask", "query_mask_refine"}:
            query_mask_logits_per_query = query_conditioned_mask_logits(
                decoded,
                spatial_state,
                self.query_mask_query_proj,
                self.query_mask_feature_proj,
            )
            outputs["query_mask_logits_per_query"] = query_mask_logits_per_query
            if self.query_refine == "query_mask_refine":
                raw_boxes = outputs["pred_boxes"]
                mask_boxes = query_mask_boxes_from_logits(query_mask_logits_per_query)
                gate = self.query_box_refine_gate * self.query_mask_gate_scale
                outputs["pred_boxes_raw"] = raw_boxes
                outputs["pred_boxes_mask"] = mask_boxes
                outputs["pred_boxes"] = (raw_boxes + gate * (mask_boxes - raw_boxes)).clamp(0.0, 1.0)
        if query_mask_logits is not None:
            outputs["query_mask_logits"] = query_mask_logits
        if query_proposal_indices is not None:
            outputs["query_proposal_indices"] = query_proposal_indices
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


def grid_queries_from_state(state: Tensor, num_queries: int) -> Tensor:
    """Create query seeds from a coarse 2D grid over the feature lattice."""

    batch, channels, height, width = state.shape
    cols = max(1, int(num_queries**0.5))
    while cols > 1 and num_queries % cols != 0:
        cols -= 1
    rows = (num_queries + cols - 1) // cols
    ys = torch.linspace(0, height - 1, rows, device=state.device).round().long()
    xs = torch.linspace(0, width - 1, cols, device=state.device).round().long()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    flat_y = yy.flatten()[:num_queries]
    flat_x = xx.flatten()[:num_queries]
    if flat_y.numel() < num_queries:
        repeats = (num_queries + flat_y.numel() - 1) // flat_y.numel()
        flat_y = flat_y.repeat(repeats)[:num_queries]
        flat_x = flat_x.repeat(repeats)[:num_queries]
    tokens = state.permute(0, 2, 3, 1)
    return tokens[:, flat_y, flat_x, :].reshape(batch, num_queries, channels)


def mask_proposal_queries_from_state(
    state: Tensor,
    mask_logits: Tensor,
    num_queries: int,
    suppression_radius: int = 0,
) -> tuple[Tensor, Tensor]:
    """Use top-k foreground mask cells as object-query feature seeds."""

    tokens = state.flatten(2).transpose(1, 2)
    scores = mask_logits.flatten(1)
    if scores.shape[1] >= num_queries:
        if suppression_radius > 0:
            indices = spatially_suppressed_topk_indices(mask_logits, num_queries, suppression_radius)
        else:
            indices = scores.topk(num_queries, dim=1).indices
    else:
        repeats = (num_queries + scores.shape[1] - 1) // scores.shape[1]
        indices = scores.topk(scores.shape[1], dim=1).indices.repeat(1, repeats)[:, :num_queries]
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return tokens.gather(dim=1, index=gather_indices), indices


def spatially_suppressed_topk_indices(mask_logits: Tensor, num_queries: int, radius: int) -> Tensor:
    """Greedily select high-mask cells while suppressing nearby spatial neighbors."""

    batch, height, width = mask_logits.shape
    scores = mask_logits.flatten(1).clone()
    grid_y = torch.arange(height, device=mask_logits.device).view(1, height, 1)
    grid_x = torch.arange(width, device=mask_logits.device).view(1, 1, width)
    selected = []
    for _ in range(num_queries):
        top = scores.argmax(dim=1)
        selected.append(top)
        y = (top // width).view(batch, 1, 1)
        x = (top % width).view(batch, 1, 1)
        suppress = (grid_y - y).abs().maximum((grid_x - x).abs()) <= radius
        scores = scores.masked_fill(suppress.flatten(1), torch.finfo(scores.dtype).min)
    return torch.stack(selected, dim=1)


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


def query_conditioned_mask_logits(
    queries: Tensor,
    state: Tensor,
    query_proj: nn.Linear,
    feature_proj: nn.Conv2d,
) -> Tensor:
    """Predict one dense spatial mask per object query."""

    projected_queries = query_proj(queries)
    projected_features = feature_proj(state)
    logits = torch.einsum("bqc,bchw->bqhw", projected_queries, projected_features)
    return logits * (queries.shape[-1] ** -0.5)


def query_mask_boxes_from_logits(mask_logits: Tensor) -> Tensor:
    """Convert query-specific mask logits into differentiable cxcywh boxes."""

    batch, queries, height, width = mask_logits.shape
    probs = mask_logits.sigmoid()
    weights = probs.flatten(2)
    mass = weights.sum(dim=-1).clamp_min(1e-6)
    xs = torch.linspace(
        0.5 / width,
        1.0 - 0.5 / width,
        width,
        device=mask_logits.device,
        dtype=mask_logits.dtype,
    )
    ys = torch.linspace(
        0.5 / height,
        1.0 - 0.5 / height,
        height,
        device=mask_logits.device,
        dtype=mask_logits.dtype,
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat_x = grid_x.flatten().view(1, 1, -1)
    flat_y = grid_y.flatten().view(1, 1, -1)
    cx = (weights * flat_x).sum(dim=-1) / mass
    cy = (weights * flat_y).sum(dim=-1) / mass
    abs_dev_x = (weights * (flat_x - cx.unsqueeze(-1)).abs()).sum(dim=-1) / mass
    abs_dev_y = (weights * (flat_y - cy.unsqueeze(-1)).abs()).sum(dim=-1) / mass
    box_w = (4.0 * abs_dev_x).clamp(1e-4, 1.0)
    box_h = (4.0 * abs_dev_y).clamp(1e-4, 1.0)
    return torch.stack((cx, cy, box_w, box_h), dim=-1).view(batch, queries, 4)


def proposal_state_decode(
    decoder: SimpleCrossAttentionDecoder,
    queries: Tensor,
    proposal_queries: Tensor,
    spatial_tokens: Tensor,
    mode: str,
    gate: Tensor,
    proposal_update: nn.Module | None = None,
) -> Tensor:
    """Decode queries while controlling how proposal state is consumed."""

    if mode == "proposal_decode2":
        decoded = decoder(queries, spatial_tokens)
        return decoder(decoded, spatial_tokens)
    if mode == "proposal_reinject":
        decoded = decoder(queries, spatial_tokens)
        return decoder(decoded + gate * proposal_queries, spatial_tokens)
    if mode in {"proposal_persistent", "proposal_persistent_stopgrad"}:
        if proposal_update is None:
            raise ValueError(f"{mode} requires proposal_update")
        proposal_state = proposal_queries
        decoded = queries
        for _ in range(2):
            decoded = decoder(decoded + gate * proposal_state, spatial_tokens)
            update_input = decoded.detach() if mode == "proposal_persistent_stopgrad" else decoded
            proposal_state = proposal_state + gate * proposal_update(update_input)
        return decoded
    if mode == "proposal_late_persistent":
        if proposal_update is None:
            raise ValueError("proposal_late_persistent requires proposal_update")
        decoded = decoder(queries, spatial_tokens)
        proposal_state = proposal_queries + gate * proposal_update(decoded.detach())
        decoded = decoder(decoded + gate * proposal_state, spatial_tokens)
        return decoded
    raise ValueError(f"unsupported proposal state mode: {mode}")
