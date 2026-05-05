"""Small baselines for synthetic 2D relation checks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .modules import Coordinate2DEncoding, PatchEmbed2D


class ConvOnlyClassifier(nn.Module):
    """A compact 2D convolutional baseline with the same patch lattice input."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        patch_size: int = 4,
        num_classes: int = 2,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(in_channels, embed_dim, patch_size)
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        blocks = []
        for _ in range(depth):
            blocks.extend(
                [
                    nn.GroupNorm(1, embed_dim),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.coord_encoding(self.patch_embed(x))
        x = x + self.blocks(x)
        return self.head(x)


class SequenceTransformerClassifier(nn.Module):
    """ViT-ish flattened-token baseline for the same patch lattice."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        patch_size: int = 4,
        num_classes: int = 2,
        depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2D(in_channels, embed_dim, patch_size)
        self.coord_encoding = Coordinate2DEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.coord_encoding(self.patch_embed(x))
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


class TinyViTClassifier(nn.Module):
    """Small ViT baseline with CLS token and learned position embedding."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        patch_size: int = 4,
        image_size: int = 32,
        num_classes: int = 2,
        depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbed2D(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.encoder(tokens)
        return self.head(tokens[:, 0])


class MultiCLSViTClassifier(nn.Module):
    """ViT baseline with multiple global tokens but no spatial memory read."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 32,
        patch_size: int = 4,
        image_size: int = 32,
        num_classes: int = 2,
        depth: int = 2,
        num_heads: int = 4,
        global_tokens: int = 4,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if global_tokens < 1:
            raise ValueError("global_tokens must be >= 1")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbed2D(in_channels, embed_dim, patch_size)
        self.global_tokens = nn.Parameter(torch.zeros(1, global_tokens, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + global_tokens, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        global_tokens = self.global_tokens.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((global_tokens, tokens), dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.encoder(tokens)
        pooled = tokens[:, : self.global_tokens.shape[1]].mean(dim=1)
        return self.head(pooled)
