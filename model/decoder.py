import numpy as np
import torch
from torch import nn


class Conv2dBA(nn.Sequential):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class UnetHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, up_ratio=1):
        super(UnetHead, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                      padding=(kernel_size // 2, kernel_size // 2)),
            nn.UpsamplingBilinear2d(scale_factor=up_ratio) if up_ratio != 1 else nn.Identity()
        )


class CUPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(CUPBlock, self).__init__()
        skip_channels = skip_channels if skip_channels is not None else 0
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
            Conv2dBA(
                in_channels=in_channels + skip_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            Conv2dBA(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x, x_s=None):
        x = self.up(x)
        if x_s is not None:
            x = torch.cat([x, x_s], dim=1)
        x = self.conv(x)
        return x


class CUP(nn.Module):
    def __init__(self, hidden_dim,
                 up_channels=(256, 128, 64, 16),
                 skip_channels=(512, 256, 64, 16),
                 n_skip_links=3):
        super(CUP, self).__init__()

        head_channels = 512
        in_channels = [head_channels] + list(up_channels[:-1])
        out_channels = up_channels
        self.n_skip_links = n_skip_links

        self.conv = Conv2dBA(
            in_channels=hidden_dim,
            out_channels=head_channels,
            kernel_size=3,
            padding=1
        )
        skip_channels = list(skip_channels)
        for i in range(4 - n_skip_links):
            skip_channels[3 - i] = None

        self.layers = nn.ModuleList([CUPBlock(i, o, s) for i, o, s in zip(in_channels, out_channels, skip_channels)])

    def forward(self, x, skips):
        n_batch, n_patch, hidden_d = x.size()
        h = w = int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(n_batch, hidden_d, h, w)
        x = self.conv(x)
        for idx, layer in enumerate(self.layers):
            s = skips[idx] if idx < self.n_skip_links else None
            x = layer(x, s)
        return x


class Decoder(nn.Module):
    def __init__(self, vit_dim, n_classes, up_channels, skip_channels, n_skip_links):
        super(Decoder, self).__init__()
        self.cup = CUP(vit_dim, up_channels, skip_channels, n_skip_links)
        self.head = UnetHead(up_channels[-1], n_classes)

    def forward(self, x, skips):
        x = self.cup(x, skips)
        x = self.head(x)
        return x
