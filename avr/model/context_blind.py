from abc import ABC, abstractmethod

import timm
import torch
from torch import nn


class WildResNet(nn.Module, ABC):
    SUPPORTED_RESNET_NAMES = ("resnet18", "resnet34", "resnet50", "resnet101")

    def __init__(
        self,
        embedding_size: int = 128,
        num_answers: int = 8,
        num_input_channels: int = 1,
        resnet_name: str = "resnet18",
    ):
        super().__init__()
        assert (
            resnet_name in self.SUPPORTED_RESNET_NAMES
        ), f"resnet_name {resnet_name} is unsupported, choose one of: {self.SUPPORTED_RESNET_NAMES}"
        self.embedding_size = embedding_size
        self.num_answers = num_answers
        num_classes = num_answers * embedding_size
        self.resnet = timm.create_model(resnet_name, num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(
            self.get_num_input_panels() * num_input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    @abstractmethod
    def get_num_input_panels(self) -> int:
        pass

    @abstractmethod
    def combine_panels(
        self, context: torch.Tensor, answers: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(self, context: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        x = self.combine_panels(context, answers)
        batch_size, num_panels, num_channels, height, width = x.shape
        x = x.view(batch_size, num_panels * num_channels, height, width)
        x = self.resnet(x)
        x = x.view(batch_size, self.num_answers, self.embedding_size)
        return x


class ContextBlindWildResNet(WildResNet):
    def get_num_input_panels(self) -> int:
        return self.num_answers

    def combine_panels(
        self, context: torch.Tensor, answers: torch.Tensor
    ) -> torch.Tensor:
        return answers
