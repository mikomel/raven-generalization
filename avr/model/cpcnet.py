"""
Source: https://github.com/Yang-Yuan/CPCNet
"""

from typing import Tuple

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn


class EntryEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels_1: int, out_channels_2: int):
        super(EntryEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels_1, momentum=0.9, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels_2, momentum=0.9, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, num_channels, height, width = x.shape
        x = x.view(batch_size * num_panels, num_channels, height, width)
        x = self.model(x)
        _, out_channels, out_height, out_width = x.shape
        x = x.view(batch_size, num_panels, out_channels, out_height, out_width)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_channels, momentum=0.9, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(num_channels, momentum=0.9, eps=1e-5)
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.residual(x)
        z = self.relu2(x + y)
        return z


class SymmetricContrastBlock(nn.Module):
    def __init__(self, num_features: int):
        super(SymmetricContrastBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features)
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_a = self.model(x_a)
        y_b = self.model(x_b)
        x_a = x_a - y_b
        x_b = x_b - y_a
        return x_a, x_b


class CPCBlock(nn.Module):
    def __init__(
            self,
            num_rows: int = 3,
            num_cols: int = 3,
            num_channels: int = 64,
            height: int = 10,
            width: int = 10,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
    ):
        super(CPCBlock, self).__init__()
        self.pathway_a = nn.Sequential(
            Rearrange('b r c d h w -> (b h w) d r c'),
            ResidualBlock(num_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            Rearrange('(b h w) d r c -> (b r c) d h w', r=num_rows, c=num_cols, h=height, w=width),
            ResidualBlock(num_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            Rearrange('(b r c) d h w -> (b r c h w) d', r=num_rows, c=num_cols, h=height, w=width),
        )
        self.pathway_b = nn.Sequential(
            Rearrange('b r c d h w -> (b r c) d h w'),
            ResidualBlock(num_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            Rearrange('(b r c) d h w -> (b h w) d r c', r=num_rows, c=num_cols, h=height, w=width),
            ResidualBlock(num_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            Rearrange('(b h w) d r c -> (b r c h w) d', r=num_rows, c=num_cols, h=height, w=width),
        )
        self.contrast_block = SymmetricContrastBlock(num_channels)
        self.rearrange = Rearrange('(b r c h w) d -> b r c d h w', r=num_rows, c=num_cols, h=height, w=width)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_a = self.pathway_a(x_a)
        x_b = self.pathway_b(x_b)
        x_a, x_b = self.contrast_block(x_a, x_b)
        x_a = self.rearrange(x_a)
        x_b = self.rearrange(x_b)
        return x_a, x_b


class CPCNet(nn.Module):
    def __init__(
            self,
            image_size: int = 80,
            num_rows: int = 3,
            num_cols: int = 3,
            num_channels: int = 64,
            embedding_size: int = 128
    ):
        super(CPCNet, self).__init__()
        self.encoder = EntryEncoder(in_channels=1, out_channels_1=num_channels // 2, out_channels_2=num_channels)
        height = width = image_size // 8
        self.split_to_rows_cols = Rearrange('b (r c) d h w -> b r c d h w', r=num_rows, c=num_cols)
        self.cpc_blocks = nn.ModuleList(
            [
                CPCBlock(num_rows=num_rows, num_cols=num_cols, num_channels=num_channels, height=height, width=width)
                for _ in range(4)
            ] + [
                CPCBlock(num_rows=num_rows, num_cols=num_cols, num_channels=num_channels, height=height, width=width, kernel_size=5, padding=2),
            ]
        )
        dim = num_rows * num_cols * height * width
        self.mlp_a = nn.Sequential(
            Reduce('b r c d h w -> b r c h w', 'mean'),
            Rearrange('b r c h w -> b (r c h w)'),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_size),
        )
        self.mlp_b = nn.Sequential(
            Reduce('b r c d h w -> b r c h w', 'mean'),
            Rearrange('b r c h w -> b (r c h w)'),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_size),
        )

    def forward(self, context: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        num_context_panels = context.size(1)
        num_answer_panels = answers.size(1)
        x = torch.cat([context, answers], dim=1)
        batch_size = x.size(0)

        x = self.encoder(x)

        x = torch.cat([
            x[:, :num_context_panels, :, :, :].unsqueeze(dim=1).repeat(1, num_answer_panels, 1, 1, 1, 1),
            x[:, num_context_panels:, :, :, :].unsqueeze(dim=2)
        ], dim=2)
        x = x.flatten(0, 1)
        x = self.split_to_rows_cols(x)

        x_a, x_b = x, x
        for block in self.cpc_blocks:
            x_a, x_b = block(x, x)

        x_a = self.mlp_a(x_a)
        x_b = self.mlp_b(x_b)

        # Original implementation treats both x_a and x_b as model predictions and computes a joint loss
        # as CCE(x_a, y) + CCE(x_b, y). For simplicity, in this implementation we will only rely on their sum.
        x = x_a + x_b

        x = x.view(batch_size, num_answer_panels, -1)
        return x
