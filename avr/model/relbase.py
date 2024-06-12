"""
Source: https://github.com/SvenShade/Rel-AIR
"""

import torch
import torch.nn as nn

from avr.model.neural_blocks import arrange_for_ravens_matrix

DR_S, DR_F = 0.1, 0.5  # Dropout prob. for spatial and fully-connected layers.
O_HC, O_OC = 64, 64  # Hidden and output channels for original enc.
F_HC, F_OC = 64, 16  # Hidden and output channels for frame enc.
S_HC, S_OC = 128, 64  # Hidden and output channels for sequence enc.
F_PL, S_PL = 5 * 5, 16  # Pooled sizes for frame and sequence enc. outputs.
F_Z = F_OC * F_PL  # Frame embedding dimensions.
K_D = 7  # Conv. kernel dimensions.

BL_IN = 3
BLOUT = F_Z
G_IN = BLOUT
G_HID = G_IN
G_OUT = G_IN
R_OUT = 32
C_DIM = 2
P_DIM = 32
C = 1.0


class perm(nn.Module):
    def __init__(self):
        super(perm, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


# Convolutional block class (conv, elu, bnorm, dropout). If 1D block, no downsampling. If 2D, stride==2.
# Implements spatial dropout for both 1D and 2D convolutional layers.
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim):
        super(ConvBlock, self).__init__()
        self.conv = getattr(nn, "Conv{}d".format(dim))(
            in_ch, out_ch, K_D, stride=dim, padding=K_D // 2
        )
        self.bnrm = getattr(nn, "BatchNorm{}d".format(dim))(out_ch)
        self.drop = (
            nn.Sequential(perm(), nn.Dropout2d(DR_S), perm())
            if dim == 1
            else nn.Dropout2d(DR_S)
        )
        self.block = nn.Sequential(self.conv, nn.ELU(), self.bnrm, self.drop)

    def forward(self, x):
        return self.block(x)


# Residual block class, made up of two convolutional blocks.
class ResBlock(nn.Module):
    def __init__(self, in_ch, hd_ch, out_ch, dim):
        super(ResBlock, self).__init__()
        self.dim = dim
        self.conv = nn.Sequential(
            ConvBlock(in_ch, hd_ch, dim), ConvBlock(hd_ch, out_ch, dim)
        )
        self.down = nn.Sequential(nn.MaxPool2d(3, 2, 1), nn.MaxPool2d(3, 2, 1))
        self.skip = getattr(nn, "Conv{}d".format(dim))(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.conv(x) + self.skip(x if self.dim == 1 else self.down(x))


class RelBase(nn.Module):
    def __init__(self, embedding_size: int = 128, num_context_panels: int = 8):
        super(RelBase, self).__init__()
        self.embedding_size = embedding_size
        self.num_context_panels = num_context_panels

        lin_in = S_OC * S_PL
        self.obj_enc = nn.Sequential(
            ResBlock(1, F_HC, F_HC, 2), ResBlock(F_HC, F_HC, F_OC, 2)
        )
        self.seq_enc = nn.Sequential(
            ResBlock(num_context_panels + 1, S_OC, S_HC, 1),
            nn.MaxPool1d(6, 4, 1),
            ResBlock(S_HC, S_HC, S_OC, 1),
            nn.AdaptiveAvgPool1d(S_PL),
        )

        self.linear = nn.Sequential(
            nn.Linear(lin_in, 512),
            nn.ELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(DR_F),
            nn.Linear(512, embedding_size),
        )

    def forward(
            self, context: torch.Tensor, answers: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        x = torch.cat([context, answers], dim=1)
        batch_size, num_panels, num_channels, height, width = x.shape
        num_context_panels = context.size(1)
        num_answer_panels = answers.size(1)
        x = x.view(batch_size * num_panels, num_channels, height, width)
        x = self.obj_enc(x).flatten(1)
        x = x.view(batch_size, num_panels, F_Z)
        x = arrange_for_ravens_matrix(x, num_context_panels, num_answer_panels)
        x = self.seq_enc(x).flatten(1)
        return self.linear(x).view(batch_size, num_answer_panels, self.embedding_size)
