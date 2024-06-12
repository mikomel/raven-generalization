import torch
from torch import nn


class BottleneckFactory:
    @staticmethod
    def create(method: str, ratio: float, input_dim: int = -1) -> nn.Module:
        if ratio == 1.0:
            return nn.Identity()
        if method == "maxpool":
            if ratio == 0.5:
                return nn.MaxPool1d(4, 2, 1)
            elif ratio == 0.25:
                return nn.MaxPool1d(6, 4, 1)
            elif ratio == 0.125:
                return nn.MaxPool1d(10, 8, 1)
            else:
                raise ValueError(f"Unsupported ratio: {ratio} for method: {method}")
        elif method == "avgpool":
            if ratio == 0.5:
                return nn.AvgPool1d(4, 2, 1)
            elif ratio == 0.25:
                return nn.AvgPool1d(6, 4, 1)
            elif ratio == 0.125:
                return nn.AvgPool1d(10, 8, 1)
            else:
                raise ValueError(f"Unsupported ratio: {ratio} for method: {method}")
        elif method == "linear":
            if input_dim == -1:
                raise ValueError(f"input_dim is required for method: {method}")
            return nn.Linear(input_dim, int(input_dim * ratio))
        elif method == "conv":
            if input_dim == -1:
                raise ValueError(f"input_dim is required for method: {method}")
            if ratio == 0.5:
                return WeightSharingSeparableConv1d(kernel_size=4, stride=2, padding=1)
            elif ratio == 0.25:
                return WeightSharingSeparableConv1d(kernel_size=6, stride=4, padding=1)
            elif ratio == 0.125:
                return WeightSharingSeparableConv1d(kernel_size=10, stride=8, padding=1)
        else:
            raise ValueError(f"Unsupported method: {method}")


class WeightSharingSeparableConv1d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, feature_dim = x.shape
        x = x.view(batch_size * num_channels, 1, feature_dim)
        x = self.conv(x)
        x = x.view(batch_size, num_channels, -1)
        return x
