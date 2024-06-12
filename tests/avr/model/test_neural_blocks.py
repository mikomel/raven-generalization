import torch

from avr.model.neural_blocks import RowPairSharedGroupConv1d


def test_row_pair_shared_group_conv_1d_get_groups():
    # in_channels=9, num_groups=3
    x = torch.rand(2, 9, 16)

    x1 = torch.stack(
        [
            x[:, [0, 1, 2, 3, 4, 5], :],
            x[:, [0, 1, 2, 6, 7, 8], :],
            x[:, [3, 4, 5, 6, 7, 8], :],
        ],
        dim=1,
    )

    groups = RowPairSharedGroupConv1d.get_groups(9, 3)
    x2 = torch.stack([x[:, group, :] for group in groups], dim=1)

    assert torch.allclose(x1, x2)

    # in_channels=32, num_groups=4
    x = torch.rand(2, 32, 16)

    x1 = torch.stack(
        [
            x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], :],
            x[:, [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23], :],
            x[:, [0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31], :],
            x[:, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], :],
            x[:, [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31], :],
            x[:, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], :],
        ],
        dim=1,
    )

    groups = RowPairSharedGroupConv1d.get_groups(32, 4)
    x2 = torch.stack([x[:, group, :] for group in groups], dim=1)

    assert torch.allclose(x1, x2)


def test_row_pair_shared_group_conv_1d():
    # in_channels=9, num_groups=3
    x = torch.rand(2, 9, 16)
    model = RowPairSharedGroupConv1d(
        in_channels=9,
        out_channels=32,
        num_groups=3,
        merge_method="sum",
        kernel_size=7,
        stride=1,
        padding=3,
    )
    y = model(x)
    assert y.shape == (2, 32, 16)

    # in_channels=32, num_groups=4
    x = torch.rand(2, 33, 16)
    model = RowPairSharedGroupConv1d(
        in_channels=32,
        out_channels=33,
        num_groups=4,
        merge_method="sum",
        kernel_size=7,
        stride=1,
        padding=3,
    )
    y = model(x)
    assert y.shape == (2, 33, 16)
