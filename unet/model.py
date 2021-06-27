import torch
import torch.nn as nn
from torch.nn.functional import pad


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_up, x_cat):
        x_up = self.up(x_up)
        padding_y = x_cat.size()[2] - x_up.size()[2]
        padding_x = x_cat.size()[3] - x_up.size()[3]
        x_up = pad(x_up, [padding_x // 2, padding_x - (padding_x // 2), padding_y // 2, padding_y - (padding_y // 2)])
        x = torch.cat([x_cat, x_up], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input_channels, out_classes, bilinear=True):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.bilinear = bilinear

        self.first_layer = DoubleConv(input_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownConv(512, 1024 // factor)
        self.up1 = UpConv(1024, 512 // factor, bilinear)
        self.up2 = UpConv(512, 256 // factor, bilinear)
        self.up3 = UpConv(256, 128 // factor, bilinear)
        self.up4 = UpConv(128, 64, bilinear)
        self.last_layer = nn.Conv2d(64, out_classes, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.first_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.last_layer(x)
