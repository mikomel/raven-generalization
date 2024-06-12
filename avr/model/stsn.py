"""
Source: https://github.com/shanka123/stsn
"""

from typing import Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from avr.model.neural_blocks import DiscretePositionEmbedding


class SoftPositionEmbedding(nn.Module):
    def __init__(self, hidden_size: int, height: int, width: int):
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size)
        grid = self.build_grid(height, width)
        self.grid = nn.Parameter(grid, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.embedding(self.grid)
        return x + grid

    @staticmethod
    def build_grid(height: int, width: int) -> torch.Tensor:
        ranges = [
            np.linspace(0.0, 1.0, num=height),
            np.linspace(0.0, 1.0, num=width),
        ]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [height, width, -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        grid = np.concatenate([grid, 1.0 - grid], axis=-1)
        return torch.from_numpy(grid)


class SlotAttention(nn.Module):
    def __init__(
        self, num_slots: int, dim: int, num_iterations: int = 3, eps: float = 1e-8
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.eps = eps
        self.scale = dim**-0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))

        self.to_queries = nn.Linear(dim, dim)
        self.to_keys = nn.Linear(dim, dim)
        self.to_values = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

        self.residual = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(
        self, x: torch.Tensor, num_slots: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_features, feature_dim = x.shape
        num_slots = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(batch_size, num_slots, -1)
        sigma = self.slots_sigma.expand(batch_size, num_slots, -1)
        slots = torch.normal(mu, sigma)

        x = self.norm_input(x)
        keys, values = self.to_keys(x), self.to_values(x)
        for _ in range(self.num_iterations):
            previous_slots = slots

            slots = self.norm_slots(slots)
            q = self.to_queries(slots)

            dots = torch.einsum("bid,bjd->bij", q, keys) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", values, attn)

            slots = self.gru(
                updates.reshape(-1, feature_dim),
                previous_slots.reshape(-1, feature_dim),
            )
            slots = slots.reshape(batch_size, -1, feature_dim)
            slots = slots + self.residual(slots)

        return slots, attn


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(
        self,
        height: int = 80,
        width: int = 80,
        num_slots: int = 9,
        num_iterations: int = 3,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.num_slots = num_slots
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, feature_dim, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 5, padding=2),
            nn.ReLU(inplace=True),
            Rearrange("b c h w -> b h w c"),
            SoftPositionEmbedding(feature_dim, height, width),
            Rearrange("b h w c -> b (h w) c"),
            nn.LayerNorm([height * width, feature_dim]),
        )
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.num_decoder_out_channels = 2
        self.decoder = nn.Sequential(
            SoftPositionEmbedding(feature_dim, height, width),
            Rearrange("b h w c -> b c h w"),
            nn.ConvTranspose2d(feature_dim, feature_dim, 5, stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_dim, feature_dim, 5, stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_dim, feature_dim, 5, stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_dim, self.num_decoder_out_channels, 3, stride=(1, 1), padding=1
            ),
            Rearrange("b c h w -> b h w c"),
        )
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=feature_dim,
            num_iterations=num_iterations,
            eps=1e-8,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        x = self.encoder(x)
        x = self.mlp(x)

        slots, attn = self.slot_attention(x)

        slots_reshaped = slots.view(batch_size * self.num_slots, 1, 1, self.feature_dim)
        slots_reshaped = slots_reshaped.repeat(1, self.height, self.width, 1)
        x = self.decoder(slots_reshaped)

        x = x.view(
            batch_size,
            self.num_slots,
            self.height,
            self.width,
            self.num_decoder_out_channels,
        )
        reconstruction, mask = x.split([1, 1], dim=-1)

        mask = self.softmax(mask)
        combined_reconstruction = (reconstruction * mask).sum(dim=1)
        combined_reconstruction = combined_reconstruction.permute(0, 3, 1, 2)

        reconstruction = reconstruction.permute(0, 1, 4, 2, 3)
        mask = mask.permute(0, 1, 4, 2, 3)
        attn = attn.reshape(batch_size, -1, 1, self.height, self.width)

        return combined_reconstruction, reconstruction, mask, slots, attn


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.model(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, dim_head: int = 64, dropout: float = 0.1
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.num_heads = num_heads
        self.scale = dim_head**-0.5

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.softmax(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                num_heads=num_heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        pool: str = "cls",
        dim_head: int = 32,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        embedding_size: int = 128,
    ):
        super().__init__()
        assert pool in (
            "mean",
            "cls",
        ), f"pool has to be one of ('mean', 'cls'), but was: {pool}"
        self.cls_token = nn.Parameter(
            torch.randn(
                dim,
            )
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, num_heads, dim_head, mlp_dim, dropout
        )
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp = nn.Linear(dim, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cls_token = self.cls_token.unsqueeze(0).unsqueeze(1)
        cls_token = cls_token.repeat(batch_size, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp(x)
        return x


class ContextNorm(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        sigma = (x.var(1, keepdim=True) + self.eps).sqrt()
        x = (x - mu) / sigma
        x = x * self.gamma + self.beta
        return x


class SlotTransformerScoringNetwork(nn.Module):
    def __init__(
        self,
        use_context_norm: bool = True,
        feature_dim: int = 32,
        vit_depth: int = 6,
        vit_num_heads: int = 8,
        vit_mlp_dim: int = 512,
        num_rows: int = 3,
        num_cols: int = 3,
        embedding_size: int = 128,
    ):
        super(SlotTransformerScoringNetwork, self).__init__()
        # TODO: have a dict for position embeddings that support various structures
        self.position_embedding = DiscretePositionEmbedding(
            num_rows, num_cols, feature_dim
        )
        self.context_norm = (
            ContextNorm(feature_dim) if use_context_norm else nn.Identity()
        )
        self.transformer = VisionTransformer(
            dim=feature_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_dim=vit_mlp_dim,
            embedding_size=embedding_size,
        )

    def forward(self, context: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        batch_size, num_context_panels, num_slots, feature_dim = context.shape
        _, num_answer_panels, _, _ = answers.shape

        position_embedding = self.position_embedding()
        position_embedding = position_embedding.view(
            1, num_context_panels + 1, 1, feature_dim
        )
        position_embedding = position_embedding.repeat(batch_size, 1, num_slots, 1)
        position_embedding = position_embedding.view(
            batch_size, (num_context_panels + 1) * num_slots, feature_dim
        )

        scores = []
        for d in range(num_answer_panels):
            x = torch.cat([context, answers[:, d].unsqueeze(1)], dim=1)
            x = x.view(batch_size, (num_context_panels + 1) * num_slots, feature_dim)
            x = self.context_norm(x)
            x = x + position_embedding
            score = self.transformer(x)
            scores.append(score)
        scores = torch.stack(scores, dim=1)
        return scores


class STSN(nn.Module):
    def __init__(
        self,
        height: int = 80,
        width: int = 80,
        num_rows: int = 3,
        num_cols: int = 3,
        feature_dim: int = 32,
        encoder_num_slots: int = 9,
        encoder_num_iterations: int = 3,
        transformer_use_context_norm: bool = True,
        vit_depth: int = 6,
        vit_num_heads: int = 8,
        vit_mlp_dim: int = 512,
        embedding_size: int = 128,
    ):
        super().__init__()
        self.encoder = SlotAttentionAutoEncoder(
            height,
            width,
            encoder_num_slots,
            encoder_num_iterations,
            feature_dim,
        )
        self.scoring_network = SlotTransformerScoringNetwork(
            transformer_use_context_norm,
            feature_dim,
            vit_depth,
            vit_num_heads,
            vit_mlp_dim,
            num_rows,
            num_cols,
            embedding_size,
        )

    def forward(
        self, context: torch.Tensor, answers: torch.Tensor, **kwargs
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, num_context_panels, num_channels, height, width = context.shape
        _, num_answer_panels, _, _, _ = answers.shape
        x = torch.cat([context, answers], dim=1)
        x = x.view(
            batch_size * (num_context_panels + num_answer_panels),
            num_channels,
            height,
            width,
        )
        combined_reconstruction, reconstruction, mask, slots, attn = self.encoder(x)
        combined_reconstruction = combined_reconstruction.view(
            batch_size,
            num_context_panels + num_answer_panels,
            num_channels,
            height,
            width,
        )
        reconstruction = reconstruction.view(
            batch_size,
            num_context_panels + num_answer_panels,
            self.encoder.num_slots,
            num_channels,
            height,
            width,
        )
        mask = mask.view(
            batch_size,
            num_context_panels + num_answer_panels,
            self.encoder.num_slots,
            num_channels,
            height,
            width,
        )
        slots = slots.view(
            batch_size,
            num_context_panels + num_answer_panels,
            self.encoder.num_slots,
            self.encoder.feature_dim,
        )
        attn = attn.view(
            batch_size,
            num_context_panels + num_answer_panels,
            self.encoder.num_slots,
            num_channels,
            height,
            width,
        )

        x_context = slots[:, :num_context_panels]
        x_answers = slots[:, num_context_panels:]
        y_hat = self.scoring_network(x_context, x_answers)

        return combined_reconstruction, reconstruction, mask, slots, attn, y_hat
