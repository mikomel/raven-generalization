import itertools
import math
from functools import partial
from typing import Callable, Optional, List

import torch
from einops.layers.torch import Rearrange
from torch import nn


class LinearBNReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBNReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        shape = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(shape)
        x = self.relu(x)
        return x


class NonLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm: str = "bn"):
        assert norm in ["bn", "ln", "none"]
        super(NonLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if norm == "bn":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        shape = x.shape
        x = x.flatten(0, -2)
        x = self.norm(x)
        x = x.view(shape)
        x = self.relu(x)
        return x


class DeepLinearBNReLU(nn.Module):
    def __init__(
            self, depth: int, in_dim: int, out_dim: int, change_dim_first: bool = True
    ):
        super(DeepLinearBNReLU, self).__init__()
        layers = []
        if change_dim_first:
            layers += [LinearBNReLU(in_dim, out_dim)]
            for _ in range(depth - 1):
                layers += [LinearBNReLU(out_dim, out_dim)]
        else:
            for _ in range(depth - 1):
                layers += [LinearBNReLU(in_dim, in_dim)]
            layers += [LinearBNReLU(in_dim, out_dim)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLP(nn.Module):
    def __init__(
            self,
            depth: int,
            in_dim: int,
            out_dim: int,
            change_dim_first: bool = True,
            norm: str = "bn",
    ):
        assert norm in ["bn", "ln", "none"]
        super(MLP, self).__init__()
        layers = []
        if change_dim_first:
            layers += [NonLinear(in_dim, out_dim, norm)]
            for _ in range(depth - 1):
                layers += [NonLinear(out_dim, out_dim, norm)]
        else:
            for _ in range(depth - 1):
                layers += [NonLinear(in_dim, in_dim, norm)]
            layers += [NonLinear(in_dim, out_dim, norm)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TwoLayerPerceptron(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, num_input_channels: int, num_output_channels: int, **kwargs):
        super(ConvBnRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_input_channels, num_output_channels, **kwargs),
            nn.BatchNorm2d(num_output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class FeedForwardResidualBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            expansion_multiplier: int = 1,
            activation: Callable = partial(nn.ReLU, inplace=True),
    ):
        super(FeedForwardResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * expansion_multiplier),
            activation(),
            nn.LayerNorm(dim * expansion_multiplier),
            nn.Linear(dim * expansion_multiplier, dim),
        )

    def forward(self, x: torch.Tensor):
        return x + self.layers(x)


def FeedForward(
        dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
        dense: Callable[..., nn.Module] = nn.Linear,
        activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
        output_dim: Optional[int] = None,
):
    output_dim = output_dim if output_dim else dim
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        activation(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, output_dim),
        nn.Dropout(dropout),
    )


class Stack(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return torch.stack(*xs, dim=self.dim)


class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.fns))


class ParallelApplyAndConcatenate(nn.Module):
    def __init__(self, *modules, dim: int):
        super().__init__()
        self.parallel_modules = nn.ModuleList(modules)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([m(x) for m in self.parallel_modules], dim=self.dim)


class ParallelMapReduce(nn.Module):
    def __init__(self, *map_modules: nn.Module, reduce_module: nn.Module):
        super().__init__()
        self.map_modules = nn.ModuleList(map_modules)
        self.reduce_module = reduce_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce_module([m(x) for m in self.map_modules])


class Scattering(nn.Module):
    def __init__(self, num_groups: int):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Equivalent to Rearrange('b c (ng gs) -> b ng (c gs)', ng=num_groups, gs=group_size)
        :param x: a Tensor with rank >= 3 and last dimension divisible by number of groups
        :param num_groups: number of groups
        """
        shape_1 = x.shape[:-1] + (self.num_groups,) + (x.shape[-1] // self.num_groups,)
        x = x.view(shape_1)
        x = x.transpose(-3, -2).contiguous()
        return x.flatten(start_dim=-2)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GroupObjectsIntoPairs(nn.Module):
    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = objects.size()
        return torch.cat(
            [
                objects.unsqueeze(1).repeat(1, num_objects, 1, 1),
                objects.unsqueeze(2).repeat(1, 1, num_objects, 1),
            ],
            dim=3,
        ).view(batch_size, num_objects ** 2, 2 * object_size)


class GroupObjectsIntoTriples(nn.Module):
    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = objects.size()
        return torch.cat(
            [
                objects.unsqueeze(1)
                .repeat(1, num_objects, 1, 1)
                .unsqueeze(2)
                .repeat(1, 1, num_objects, 1, 1),
                objects.unsqueeze(1)
                .repeat(1, num_objects, 1, 1)
                .unsqueeze(3)
                .repeat(1, 1, 1, num_objects, 1),
                objects.unsqueeze(2)
                .repeat(1, 1, num_objects, 1)
                .unsqueeze(3)
                .repeat(1, 1, 1, num_objects, 1),
            ],
            dim=4,
        ).view(batch_size, num_objects ** 3, 3 * object_size)


class DiscretePositionEmbedding(nn.Module):
    def __init__(self, num_rows: int, num_cols: int, output_dim: int):
        super().__init__()
        coordinates = self.get_coordinate_matrix(num_rows, num_cols)
        self.coordinates = nn.Parameter(coordinates, requires_grad=False)
        self.embedding = nn.Linear(self.coordinates.size(-1), output_dim)

    def forward(self) -> torch.Tensor:
        """
        :return: a tensor of shape (num_rows * num_cols, output_dim)
        """
        x = self.embedding(self.coordinates)
        return x

    @staticmethod
    def get_coordinate_matrix(num_rows: int, num_cols: int) -> torch.Tensor:
        """
        Creates a coordinate matrix, e.g. for a problem with 2x2 structure:
        [[1, 0, 1, 0],
         [1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 0, 1]]
        :return: a tensor of shape (num_rows * num_cols, num_rows + num_cols)
        """
        coordinates = []
        for row in range(num_rows):
            for col in range(num_cols):
                coordinate = torch.zeros(num_rows + num_cols, dtype=torch.float32)
                coordinate[row] = 1.0
                coordinate[num_rows + col] = 1.0
                coordinates.append(coordinate)
        coordinates = torch.stack(coordinates, dim=0)
        return coordinates


class BatchBroadcast(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.module.forward()
        repeat = [x.shape[0]] + [1] * (x.ndim - 1)
        y = y.unsqueeze(0).repeat(repeat)
        return y


class Sum(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dim)


class SplitAttention1d(nn.Module):
    def __init__(self, num_groups: int, in_channels: int):
        super().__init__()
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.reweight = TwoLayerPerceptron(
            in_channels, in_channels // 4, in_channels * num_groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a tensor of shape (batch_size, num_groups, in_channels, dim)
        :return: a tensor of shape (batch_size, in_channels, dim)
        """
        batch_size, num_groups, in_channels, dim = x.shape
        weights = x.sum(dim=1)  # (batch_size, in_channels, dim)
        weights = weights.mean(2)  # (batch_size, in_channels)
        weights = self.reweight(weights)  # (batch_size, in_channels * num_groups)
        weights = weights.view(
            batch_size, in_channels, num_groups
        )  # (batch_size, in_channels, num_groups)
        weights = weights.permute(2, 0, 1)  # (num_groups, batch_size, in_channels)
        weights = weights.softmax(dim=0)  # (num_groups, batch_size, in_channels)
        weights = weights.unsqueeze(-1)  # (num_groups, batch_size, in_channels, 1)

        xs = x.split(1, dim=1)  # num_groups * (batch_size, in_channels, dim)
        weights = weights.split(1, dim=0)  # num_groups * (batch_size, in_channels, 1)
        x = torch.stack(
            [x.squeeze(1) * w.squeeze(0) for x, w in zip(xs, weights)], dim=1
        )  # (batch_size, num_groups, in_channels, dim)
        x = x.sum(dim=1)  # (batch_size, in_channels, dim)
        return x


class StructureAwareNormalization(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        x = (x - mean) / std
        return x


class ParameterizedStructureAwareNormalization(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var, mean = torch.var_mean(x, dim=1, keepdim=True)
        std = (var + self.eps).sqrt()
        x = (x - mean) / std
        x = x * self.scale + self.shift
        return x


class SharedGroupConv1d(nn.Module):
    """
    Group convolution with shared weights.
    """

    MERGE_METHODS = {"sum", "concat", "split-attention"}

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_groups: int,
            merge_method: str,
            use_norm: bool = False,
            use_pre_norm: bool = False,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.merge_method = merge_method
        assert in_channels % num_groups == 0
        group_in_channels = in_channels // num_groups
        self.split_into_groups = Rearrange(
            "b (g c) d -> (b g) c d", g=num_groups, c=group_in_channels
        )
        if merge_method == "sum":
            self.out_channels = out_channels
        elif merge_method == "concat":
            assert out_channels % num_groups == 0
            self.out_channels = out_channels // num_groups
        elif merge_method == "split-attention":
            self.out_channels = out_channels
            self.split_attention = SplitAttention1d(num_groups, out_channels)
        else:
            raise ValueError(
                f"Merge_method must be one of {{{self.MERGE_METHODS}}}, but was: {merge_method}"
            )
        if use_pre_norm:
            self.conv = nn.Sequential(
                StructureAwareNormalization() if use_norm else nn.Identity(),
                nn.Conv1d(group_in_channels, self.out_channels, *args, **kwargs),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(group_in_channels, self.out_channels, *args, **kwargs),
                StructureAwareNormalization() if use_norm else nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a tensor of shape (batch_size, in_channels, dim)
        :return: a tensor of shape (batch_size, group_out_channels
        """
        batch_size, in_channels, dim = x.shape
        x = self.split_into_groups(
            x
        )  # (batch_size * num_groups, group_in_channels, dim)
        x = self.conv(x)  # (batch_size * num_groups, out_channels, dim)
        x = x.view(batch_size, self.num_groups, self.out_channels, dim)
        if self.merge_method == "sum":
            x = x.sum(dim=1)
        elif self.merge_method == "concat":
            x = x.view(batch_size, self.num_groups * self.out_channels, dim)
        elif self.merge_method == "split-attention":
            x = self.split_attention(x)
        else:
            raise ValueError(
                f"Merge_method must be one of {{{self.MERGE_METHODS}}}, but was: {self.merge_method}"
            )
        return x


class RowPairSharedGroupConv1d(nn.Module):
    """
    Group convolution with shared weights that operates on pairs of RPM rows.
    """

    MERGE_METHODS = {"sum", "concat", "split-attention"}
    NUM_ROW_PAIRS = 3
    NUM_CHANNELS_IN_ROW_PAIR_GROUP = 6

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_groups: int,
            merge_method: str,
            use_norm: bool = False,
            use_pre_norm: bool = False,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.merge_method = merge_method
        if merge_method == "sum":
            self.out_channels = out_channels
        elif merge_method == "concat":
            assert out_channels % self.NUM_ROW_PAIRS == 0
            self.out_channels = out_channels // self.NUM_ROW_PAIRS
        elif merge_method == "split-attention":
            self.out_channels = out_channels
            self.split_attention = SplitAttention1d(self.NUM_ROW_PAIRS, out_channels)
        else:
            raise ValueError(
                f"Merge_method must be one of {{{self.MERGE_METHODS}}}, but was: {merge_method}"
            )

        if in_channels % num_groups != 0:
            raise ValueError(
                f"in_channels={in_channels} must be divisible by num_groups={num_groups}"
            )
        self.num_group_pairs = math.comb(num_groups, 2)
        self.num_group_pair_channels = in_channels // num_groups * 2
        self.groups = self.get_groups(in_channels, num_groups)
        if use_pre_norm:
            self.conv = nn.Sequential(
                StructureAwareNormalization() if use_norm else nn.Identity(),
                nn.Conv1d(self.num_group_pair_channels, self.out_channels, *args, **kwargs),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(self.num_group_pair_channels, self.out_channels, *args, **kwargs),
                StructureAwareNormalization() if use_norm else nn.Identity(),
            )

    @staticmethod
    def get_groups(in_channels: int, num_groups: int) -> torch.Tensor:
        if in_channels % num_groups != 0:
            raise ValueError(
                f"in_channels={in_channels} must be divisible by num_groups={num_groups}"
            )
        group_size = in_channels // num_groups
        indices = list(range(num_groups))
        groups = []
        for group_idx_1, group_idx_2 in itertools.combinations(indices, r=2):
            groups.append(
                list(
                    itertools.chain(
                        range(group_idx_1 * group_size, (group_idx_1 + 1) * group_size),
                        range(group_idx_2 * group_size, (group_idx_2 + 1) * group_size),
                    )
                )
            )
        return torch.tensor(groups, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a tensor of shape (batch_size, in_channels, dim)
        :return: a tensor of shape (batch_size, out_channels, *)
        """
        batch_size, in_channels, dim = x.shape
        x = torch.stack(
            [x[:, group, :] for group in self.groups], dim=1
        )  # (batch_size, num_group_pairs, num_group_pair_channels, dim)
        x = x.view(
            batch_size * self.num_group_pairs, self.num_group_pair_channels, dim
        )
        x = self.conv(x)  # (batch_size * num_group_pairs, out_channels, dim)
        x = x.view(batch_size, self.num_group_pairs, self.out_channels, dim)
        if self.merge_method == "sum":
            x = x.sum(dim=1)
        elif self.merge_method == "concat":
            x = x.view(batch_size, self.NUM_ROW_PAIRS * self.out_channels, dim)
        elif self.merge_method == "split-attention":
            x = self.split_attention(x)
        else:
            raise ValueError(
                f"Merge_method must be one of {{{self.MERGE_METHODS}}}, but was: {self.merge_method}"
            )
        return x


def arrange_for_ravens_matrix(
        x: torch.Tensor, num_context_panels: int, num_answer_panels: int
) -> torch.Tensor:
    batch_size, num_panels, embedding_dim = x.shape
    x = torch.stack(
        [
            torch.cat((x[:, :num_context_panels], x[:, i].unsqueeze(1)), dim=1)
            for i in range(num_context_panels, num_panels)
        ],
        dim=1,
    )
    x = x.view(batch_size * num_answer_panels, num_context_panels + 1, embedding_dim)
    return x
