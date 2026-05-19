"""Detection heads and query utilities."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Simple feed-forward network used for box regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = []
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out_dim = output_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_layers - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DetectionHead(nn.Module):
    """Class, box, and query-quality heads for object-query features."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        quality_mode: str = "query",
        class_mode: str = "query",
    ) -> None:
        super().__init__()
        if quality_mode not in {"query", "box"}:
            raise ValueError("quality_mode must be 'query' or 'box'")
        if class_mode not in {"query", "box"}:
            raise ValueError("class_mode must be 'query' or 'box'")
        self.quality_mode = quality_mode
        self.class_mode = class_mode
        class_dim = dim + 4 if class_mode == "box" else dim
        self.class_head = nn.Linear(class_dim, num_classes + 1)
        self.box_head = MLP(dim, dim, 4, num_layers=3)
        quality_dim = dim + 4 if quality_mode == "box" else dim
        self.quality_head = nn.Linear(quality_dim, 1)

    def forward(self, queries: Tensor) -> dict[str, Tensor]:
        pred_box_logits = self.box_head(queries)
        pred_boxes = pred_box_logits.sigmoid()
        class_input = queries
        if self.class_mode == "box":
            class_input = torch.cat([queries, pred_boxes.detach()], dim=-1)
        quality_input = queries
        if self.quality_mode == "box":
            quality_input = torch.cat([queries, pred_boxes.detach()], dim=-1)
        return {
            "pred_logits": self.class_head(class_input),
            "pred_boxes": pred_boxes,
            "pred_boxes_logits": pred_box_logits,
            "pred_quality_logits": self.quality_head(quality_input).squeeze(-1),
        }


class LearnedObjectQueries(nn.Module):
    """Learned object query embeddings with batch expansion."""

    def __init__(self, num_queries: int, dim: int) -> None:
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

    def forward(self, batch_size: int) -> Tensor:
        return self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
