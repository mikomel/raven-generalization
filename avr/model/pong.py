from functools import partial
from typing import Callable, List

import torch
from einops.layers.torch import Rearrange
from torch import nn

from avr.model.bottleneck import BottleneckFactory
from avr.model.neural_blocks import (
    ParallelSum,
    arrange_for_ravens_matrix,
    FeedForwardResidualBlock,
    DiscretePositionEmbedding,
    ParallelApplyAndConcatenate,
    BatchBroadcast,
    SharedGroupConv1d,
    SplitAttention1d,
    ParallelMapReduce,
    Sum,
    Stack,
    RowPairSharedGroupConv1d,
)


class Pong(nn.Module):
    def __init__(
            self,
            image_size: int = 80,
            embedding_size: int = 128,
            num_rows: int = 3,
            num_cols: int = 3,
            # panel encoder
            panel_encoder_num_input_channels: int = 1,
            panel_encoder_num_hidden_channels: int = 32,
            panel_encoder_kernel_size: int = 7,
            panel_encoder_stride: int = 2,
            panel_encoder_padding: int = 3,
            panel_encoder_activation_fn: Callable = partial(nn.ReLU, inplace=True),
            panel_encoder_use_batch_norm: bool = True,
            panel_encoder_dropout: float = 0.0,
            panel_encoder_num_blocks: int = 2,
            panel_encoder_block_depth: int = 2,
            panel_encoder_spatial_projection_output_ratio: float = 1.0,
            # reasoner
            reasoner_output_dim: int = 16,
            reasoner_kernel_size: int = 7,
            reasoner_stride: int = 1,
            reasoner_padding: int = 3,
            reasoner_activation_fn: Callable = partial(nn.ReLU, inplace=True),
            reasoner_use_batch_norm: bool = True,
            reasoner_dropout: float = 0.0,
            reasoner_depth: int = 3,
            reasoner_bottleneck_method: str = "avgpool",
            reasoner_bottleneck_ratios: List[float] = (0.125, 0.25),
            reasoner_block_depth: int = 2,
            reasoner_merge_method: str = "sum",
            reasoner_group_conv_merge_method: str = "sum",
            reasoner_group_conv_hidden_num_groups: int = 8,
            reasoner_row_pair_group_conv_hidden_num_groups: int = 4,
            reasoner_num_hidden_channels: int = 32,
            reasoner_group_conv_use_norm: bool = False,
            reasoner_group_conv_use_pre_norm: bool = False,
            reasoner_use_row_group_conv: bool = True,
            reasoner_use_row_pair_group_conv: bool = True,
            reasoner_use_full_context_conv: bool = True,
            # output projection
            output_projection_num_blocks: int = 1,
            output_projection_activation_fn: Callable = partial(nn.ReLU, inplace=True),
            output_projection_use_batch_norm: bool = True,
            output_projection_dropout: float = 0.0,
    ):
        super(Pong, self).__init__()
        assert 0.0 < panel_encoder_spatial_projection_output_ratio <= 1.0
        assert (
                len(reasoner_bottleneck_ratios) == reasoner_depth - 1
        ), f"For reasoner_depth: {reasoner_depth}, the number of bottleneck ratios must be: {reasoner_depth - 1}"

        self.embedding_size = embedding_size
        self.num_context_panels = num_rows * num_cols
        self.panel_encoder_num_hidden_channels = panel_encoder_num_hidden_channels

        # input shape: (b, p, c, h, w)
        self.panel_encoder = PanelEncoder(
            image_size,
            panel_encoder_num_input_channels,
            panel_encoder_num_hidden_channels,
            panel_encoder_kernel_size,
            panel_encoder_stride,
            panel_encoder_padding,
            panel_encoder_activation_fn,
            panel_encoder_use_batch_norm,
            panel_encoder_dropout,
            panel_encoder_num_blocks,
            panel_encoder_block_depth,
        )

        # input shape: (b, p, c, h * w)
        self.panel_embedding_spatial_projection_input_dim = (
                self.panel_encoder.output_image_size ** 2
        )
        self.panel_embedding_spatial_projection_output_dim = int(
            self.panel_embedding_spatial_projection_input_dim
            * panel_encoder_spatial_projection_output_ratio
        )
        self.panel_embedding_spatial_projection = nn.Sequential(
            nn.Linear(
                self.panel_embedding_spatial_projection_input_dim,
                self.panel_embedding_spatial_projection_output_dim,
            ),
            panel_encoder_activation_fn(),
        )

        # input shape: (b, p, c * h * w)
        self.panel_embedding_channel_projection_input_dim = (
                panel_encoder_num_hidden_channels
                * self.panel_embedding_spatial_projection_output_dim
        )
        self.panel_embedding_channel_projection_output_dim = (
            self.panel_embedding_channel_projection_input_dim
        )
        self.panel_embedding_channel_projection = FeedForwardResidualBlock(
            self.panel_embedding_channel_projection_input_dim,
            activation=panel_encoder_activation_fn,
            expansion_multiplier=2,
        )

        # input shape: (b, p, d)
        panel_embedding_position_encoder_input_dim = (
            self.panel_embedding_channel_projection_output_dim
        )
        panel_embedding_position_encoder_output_dim = (
            panel_embedding_position_encoder_input_dim
        )
        reasoner_num_input_channels = self.num_context_panels
        position_embedding_dim = (
                self.panel_embedding_channel_projection_output_dim
                // panel_encoder_num_hidden_channels
        )
        self.panel_embedding_position_encoder = ParallelApplyAndConcatenate(
            nn.Identity(),
            BatchBroadcast(
                DiscretePositionEmbedding(
                    num_rows,
                    num_cols,
                    position_embedding_dim,
                ),
            ),
            dim=-1,
        )
        panel_embedding_position_encoder_output_dim += position_embedding_dim

        # input shape: (b, p, d)
        reasoner_input_dim = panel_embedding_position_encoder_output_dim
        self.reasoner = Reasoner(
            reasoner_num_input_channels,
            reasoner_num_hidden_channels,
            reasoner_input_dim,
            reasoner_output_dim,
            reasoner_kernel_size,
            reasoner_stride,
            reasoner_padding,
            reasoner_activation_fn,
            reasoner_use_batch_norm,
            reasoner_dropout,
            reasoner_depth,
            reasoner_bottleneck_method,
            reasoner_bottleneck_ratios,
            reasoner_block_depth,
            reasoner_merge_method,
            reasoner_group_conv_merge_method,
            reasoner_group_conv_hidden_num_groups,
            reasoner_row_pair_group_conv_hidden_num_groups,
            reasoner_group_conv_use_norm,
            reasoner_group_conv_use_pre_norm,
            reasoner_use_row_group_conv,
            reasoner_use_row_pair_group_conv,
            reasoner_use_full_context_conv,
        )
        self.reasoner_output_dim = (
                reasoner_num_hidden_channels * reasoner_output_dim
        )

        # input shape: (b, d)
        self.output_projection = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(self.reasoner_output_dim, self.reasoner_output_dim),
                    output_projection_activation_fn(),
                    nn.BatchNorm1d(self.reasoner_output_dim)
                    if output_projection_use_batch_norm
                    else nn.Identity(),
                    nn.Dropout(output_projection_dropout),
                )
                for _ in range(output_projection_num_blocks)
            ],
            nn.Linear(self.reasoner_output_dim, embedding_size),
        )

    def forward(
            self, context: torch.Tensor, answers: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        x = torch.cat([context, answers], dim=1)
        batch_size, num_panels, num_channels, height, width = x.shape
        num_context_panels = context.size(1)
        num_answer_panels = answers.size(1)

        x = x.view(batch_size * num_panels, num_channels, height, width)
        x = self.panel_encoder(x)

        x = x.view(
            batch_size,
            num_panels,
            self.panel_encoder_num_hidden_channels,
            self.panel_embedding_spatial_projection_input_dim,
        )
        x = self.panel_embedding_spatial_projection(x)

        x = x.view(
            batch_size,
            num_panels,
            self.panel_encoder_num_hidden_channels
            * self.panel_embedding_spatial_projection_output_dim,
        )
        x = self.panel_embedding_channel_projection(x)

        x = arrange_for_ravens_matrix(x, num_context_panels, num_answer_panels)
        x = self.panel_embedding_position_encoder(x)

        x = self.reasoner(x)

        x = x.view(batch_size * num_answer_panels, self.reasoner_output_dim)
        x = self.output_projection(x)

        x = x.view(batch_size, num_answer_panels, self.embedding_size)
        return x


class PanelEncoder(nn.Module):
    def __init__(
            self,
            image_size: int = 80,
            num_input_channels: int = 1,
            num_hidden_channels: int = 32,
            kernel_size: int = 7,
            stride: int = 2,
            padding: int = 3,
            activation_fn: Callable = partial(nn.ReLU, inplace=True),
            use_batch_norm: bool = True,
            dropout: float = 0.0,
            num_blocks: int = 2,
            block_depth: int = 2,
    ):
        super().__init__()
        assert num_blocks in {1, 2}
        assert block_depth in {1, 2}

        batch_norm_fn = lambda: (
            nn.BatchNorm2d(num_hidden_channels) if use_batch_norm else nn.Identity()
        )
        dropout_fn = lambda: (nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())

        # Calculate the effect of applying a single convolution block on the embedding size
        output_image_size = (image_size - kernel_size + 2 * padding) / stride + 1
        output_image_ratio = round(image_size / output_image_size)
        self.output_image_size = round(
            image_size / (output_image_ratio ** (num_blocks * block_depth))
        )
        if output_image_ratio == 1:
            max_pool_kernel_size = 3
            max_pool_stride = 1
            max_pool_padding = 1
        elif output_image_ratio == 2:
            max_pool_kernel_size = 3
            max_pool_stride = 2
            max_pool_padding = 1
        else:
            raise ValueError(
                f"max pooling parameters not configured for image size: {image_size} and outputs size: {output_image_size}"
            )

        self.model = nn.Sequential(
            *[
                ParallelSum(
                    nn.Sequential(
                        *[
                            nn.Sequential(
                                nn.Conv2d(
                                    num_input_channels
                                    if block_idx == 0 and layer_idx == 0
                                    else num_hidden_channels,
                                    num_hidden_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                ),
                                activation_fn(),
                                batch_norm_fn(),
                                dropout_fn(),
                            )
                            for layer_idx in range(block_depth)
                        ],
                    ),
                    nn.Sequential(
                        *[
                            nn.MaxPool2d(
                                max_pool_kernel_size, max_pool_stride, max_pool_padding
                            )
                            for _ in range(block_depth)
                        ],
                        nn.Conv2d(
                            num_input_channels
                            if block_idx == 0
                            else num_hidden_channels,
                            num_hidden_channels,
                            kernel_size=1,
                            bias=False,
                        ),
                    ),
                )
                for block_idx in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Reasoner(nn.Module):
    def __init__(
            self,
            num_context_panels: int = 9,
            num_hidden_panels: int = 32,
            input_dim: int = 800,
            output_dim: int = 16,
            kernel_size: int = 7,
            stride: int = 1,
            padding: int = 3,
            activation_fn: Callable = partial(nn.ReLU, inplace=True),
            use_batch_norm: bool = True,
            dropout: float = 0.0,
            depth: int = 2,
            bottleneck_method: str = "maxpool",
            bottleneck_ratios: List[float] = (0.25,),
            block_depth: int = 2,
            merge_method: str = "sum",
            group_conv_merge_method: str = "sum",
            group_conv_hidden_num_groups: int = 8,
            row_pair_group_conv_hidden_num_groups: int = 4,
            group_conv_use_norm: bool = False,
            group_conv_use_pre_norm: bool = False,
            use_row_group_conv: bool = True,
            use_row_pair_group_conv: bool = True,
            use_full_context_conv: bool = True,
    ):
        super().__init__()
        first_block_depth = 1
        first_group_conv_hidden_num_groups = 3
        layers = []
        layers += [
            ReasonerBlock(
                num_context_panels,
                num_hidden_panels,
                kernel_size,
                stride,
                padding,
                activation_fn,
                use_batch_norm,
                dropout,
                first_block_depth,
                merge_method,
                group_conv_merge_method,
                first_group_conv_hidden_num_groups,
                first_group_conv_hidden_num_groups,
                group_conv_use_norm,
                group_conv_use_pre_norm,
                use_row_group_conv,
                use_row_pair_group_conv,
                use_full_context_conv,
            )
        ]
        for i in range(depth - 1):
            layers += [
                BottleneckFactory.create(
                    method=bottleneck_method,
                    ratio=bottleneck_ratios[i],
                    input_dim=input_dim,
                ),
                ReasonerBlock(
                    num_hidden_panels,
                    num_hidden_panels,
                    kernel_size,
                    stride,
                    padding,
                    activation_fn,
                    use_batch_norm,
                    dropout,
                    block_depth,
                    merge_method,
                    group_conv_merge_method,
                    group_conv_hidden_num_groups,
                    row_pair_group_conv_hidden_num_groups,
                    group_conv_use_norm,
                    group_conv_use_pre_norm,
                    use_row_group_conv,
                    use_row_pair_group_conv,
                    use_full_context_conv,
                ),
            ]
            input_dim = int(input_dim * bottleneck_ratios[i])
        layers += [nn.AdaptiveAvgPool1d(output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ReasonerBlock(nn.Module):
    MERGE_METHODS = {"sum", "split-attention"}

    def __init__(
            self,
            num_input_channels: int = 9,
            num_output_channels: int = 128,
            kernel_size: int = 7,
            stride: int = 1,
            padding: int = 3,
            activation_fn: Callable = partial(nn.ReLU, inplace=True),
            use_batch_norm: bool = True,
            dropout: float = 0.0,
            block_depth: int = 2,
            merge_method: str = "sum",
            group_conv_merge_method: str = "sum",
            group_conv_hidden_num_groups: int = 8,
            row_pair_group_conv_hidden_num_groups: int = 4,
            group_conv_use_norm: bool = False,
            group_conv_use_pre_norm: bool = False,
            use_row_group_conv: bool = True,
            use_row_pair_group_conv: bool = True,
            use_full_context_conv: bool = True,
    ):
        super().__init__()
        batch_norm_fn = lambda: (
            nn.BatchNorm1d(num_output_channels) if use_batch_norm else nn.Identity()
        )
        dropout_fn = lambda: (
            nn.Sequential(
                Rearrange("b p d -> b d p"),
                nn.Dropout1d(dropout),
                Rearrange("b d p -> b p d"),
            )
            if dropout > 0
            else nn.Identity()
        )
        if merge_method == "sum":
            merge_module = Sum(dim=1)
        elif merge_method == "split-attention":
            num_groups = 4
            merge_module = SplitAttention1d(num_groups, num_output_channels)
        else:
            raise ValueError(
                f"Merge_method must be one of {{{self.MERGE_METHODS}}}, but was: {merge_method}"
            )
        modules = [
            nn.Conv1d(
                num_input_channels, num_output_channels, kernel_size=1, bias=False
            )
        ]
        if use_row_group_conv:
            modules.append(
                nn.Sequential(
                    *[
                        nn.Sequential(
                            SharedGroupConv1d(
                                num_input_channels
                                if layer_idx == 0
                                else num_output_channels,
                                num_output_channels,
                                group_conv_hidden_num_groups,
                                group_conv_merge_method,
                                group_conv_use_norm,
                                group_conv_use_pre_norm,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                            ),
                            activation_fn(),
                            batch_norm_fn(),
                            dropout_fn(),
                        )
                        for layer_idx in range(block_depth)
                    ]
                )
            )
        if use_row_pair_group_conv:
            modules.append(
                nn.Sequential(
                    *[
                        nn.Sequential(
                            RowPairSharedGroupConv1d(
                                num_input_channels
                                if layer_idx == 0
                                else num_output_channels,
                                num_output_channels,
                                row_pair_group_conv_hidden_num_groups,
                                group_conv_merge_method,
                                group_conv_use_norm,
                                group_conv_use_pre_norm,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                            ),
                            activation_fn(),
                            batch_norm_fn(),
                            dropout_fn(),
                        )
                        for layer_idx in range(block_depth)
                    ]
                )
            )
        if use_full_context_conv:
            modules.append(
                nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                num_input_channels
                                if layer_idx == 0
                                else num_output_channels,
                                num_output_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                            ),
                            activation_fn(),
                            batch_norm_fn(),
                            dropout_fn(),
                        )
                        for layer_idx in range(block_depth)
                    ]
                )
            )
        self.model = nn.Sequential(
            ParallelMapReduce(
                *modules,
                reduce_module=Stack(dim=1),
            ),
            merge_module,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
