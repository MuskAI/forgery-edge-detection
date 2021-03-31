""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilateDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class MaxPool(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.maxpool(x)


class Down_no_pool(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class RelationMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RelationMap, self).__init__()
        self.relation_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.relation_conv(x)
        return x


class RelationFuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RelationFuse, self).__init__()
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.final(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FuseStageOut(nn.Module):
    def __init__(self, in_channels, out_channles):
        super(FuseStageOut, self).__init__()
        self.fuse_stage_out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channles, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse_stage_out(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.Sigmoid()(x)
        return x
class TwoStageOut(nn.Module):
    def __init__(self, in_channels, out_channles):
        super(TwoStageOut, self).__init__()
        self.fuse_stage_out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channles, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channles, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse_stage_out(x)
        return x
