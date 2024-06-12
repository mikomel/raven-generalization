import torch
from einops.layers.torch import Rearrange
from torch import nn

from avr.model.neural_blocks import ConvBnRelu, FeedForwardResidualBlock, Scattering


class SCL(nn.Module):
    def __init__(
            self,
            num_hidden_channels: int = 32,
            embedding_size: int = 128,
            ff_dim: int = 80,
            image_size: int = 160,
            num_local_scattering_groups: int = 10,
            num_global_scattering_groups: int = 80,
            num_context_panels: int = 9):
        super(SCL, self).__init__()
        assert ff_dim % num_local_scattering_groups == 0
        assert ff_dim % num_global_scattering_groups == 0
        local_scattering_group_size = ff_dim // num_local_scattering_groups
        global_scattering_group_size = (num_local_scattering_groups * 8) // num_global_scattering_groups
        c = num_hidden_channels
        d = embedding_size
        conv_dimension = (40 * (image_size // 80)) ** 2

        self.num_global_scattering_groups = num_global_scattering_groups
        self.global_scattering_group_size = global_scattering_group_size

        self.model_local = nn.Sequential(
            ConvBnRelu(1, c // 2, kernel_size=3, stride=2, padding=1),
            ConvBnRelu(c // 2, c // 2, kernel_size=3, padding=1),
            ConvBnRelu(c // 2, c, kernel_size=3, padding=1),
            ConvBnRelu(c, c, kernel_size=3, padding=1),

            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(conv_dimension, ff_dim),
            nn.ReLU(inplace=True),
            FeedForwardResidualBlock(ff_dim),

            Scattering(num_groups=num_local_scattering_groups),
            nn.Linear(c * local_scattering_group_size, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, 8),

            nn.Flatten(start_dim=-2, end_dim=-1),
            FeedForwardResidualBlock(num_local_scattering_groups * 8))

        self.model_global = nn.Sequential(
            Scattering(num_groups=num_global_scattering_groups),
            Rearrange('b ng d -> (b ng) d'),
            nn.Linear(num_context_panels * global_scattering_group_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5),
            Rearrange('(b ng) d -> b (ng d)', ng=num_global_scattering_groups),
            FeedForwardResidualBlock(5 * num_global_scattering_groups),
            nn.Linear(5 * num_global_scattering_groups, d))

    def forward(self, context: torch.Tensor, answers: torch.Tensor, **kwargs) -> torch.Tensor:
        x = torch.cat([context, answers], dim=1)
        num_context_panels = context.size(1)
        num_answer_panels = answers.size(1)
        batch_size, num_panels, num_channels, height, width = x.shape

        x = x.view((batch_size * num_panels), num_channels, height, width)
        x = self.model_local(x)
        x = x.view(batch_size, num_panels, -1)

        x = torch.cat([
            x[:, :num_context_panels, :].unsqueeze(dim=1).repeat(1, num_answer_panels, 1, 1),
            x[:, num_context_panels:, :].unsqueeze(dim=2)
        ], dim=2)
        x = x.view((batch_size * num_answer_panels), (num_context_panels + 1), -1)

        x = self.model_global(x)
        x = x.view(batch_size, num_answer_panels, -1)
        return x
