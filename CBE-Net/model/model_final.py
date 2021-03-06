#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：forgery-edge-detection
@File    ：model_final.py
@IDE     ：PyCharm
@Author  ：haoran
@Date    ：2022/4/5 9:44 PM
最终的两个阶段的模型

'''
import pdb

import torch.nn.functional as F
from torchsummary import summary
import sys

sys.path.append('')
import torch.nn as nn
from .unet_parts import *



class UNetStage1(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(UNetStage1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        # /1 --> /4
        # self.down1 = Down(64, 128)
        self.down2 = Down_no_pool(64, 128)

        # /4 --> /8
        self.down3 = Down(128, 256)

        # /8 --> /16
        self.down4 = Down(256, 512)

        self.hold = DilateDoubleConv(512, 256)

        # /16 --> /8
        self.up1 = Up(256+256, 128, bilinear)

        # /8 --> /4
        self.up2 = Up(128+128, 64, bilinear)

        # /4 --> /1
        self.up4 = Up2x(64+64, 32, bilinear)

        self.outc = OutConv(32,1)

        # self.cls_module =

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        hold = self.hold(x4)

        stage_x1 = self.up1(hold, x3)

        stage_x2 = self.up2(stage_x1, x2)

        stage_x4 = self.up4(stage_x2, x1)
        logits = self.outc(stage_x4)
        return {
            'logits':logits,
            '16':hold,
            '8':stage_x1,
            '4':stage_x2,
            '1':stage_x4
        }

# class UNetStage2(nn.Module):
#     def __init__(self, n_channels=6, bilinear=True):
#         super(UNetStage2, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = 1
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 64)
#         self.maxpool = MaxPool()
#         self.down1 = Down_no_pool(64, 128)
#         self.down2 = Down_no_pool(128, 256)
#         self.down3 = Down_no_pool(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down_no_pool(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         # self.outc = OutConv(64, 1)
#
#         self.fuse2 = FuseStageOut(in_channels=64 + 64, out_channles=64)
#         self.fuse3 = FuseStageOut(in_channels=128 + 128, out_channles=128)
#         self.fuse4 = FuseStageOut(in_channels=256 + 256, out_channles=256)
#
#
#         self.with_relation = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, padding=1),
#             nn.BatchNorm2d(8),
#             nn.ReLU()
#         )
#
#     def forward(self, x, stage3, stage2, stage1):
#         x1 = self.inc(x)
#
#         # fuse stage
#         x2 = self.maxpool(x1)
#         x2 = self.fuse2(x2, stage1)
#         x2 = self.down1(x2)
#
#         x3 = self.maxpool(x2)
#         x3 = self.fuse3(x3, stage2)
#         x3 = self.down2(x3)
#
#         x4 = self.maxpool(x3)
#         x4 = self.fuse4(x4, stage3)
#         x4 = self.down3(x4)
#
#         x5 = self.maxpool(x4)
#         x5 = self.down4(x5)
#
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#
#         r1 = self.relation1(x)
#         r2 = self.relation2(x)
#
#         r3 = self.relation3(x)
#         r4 = self.relation4(x)
#
#         r5 = self.relation5(x)
#         r6 = self.relation6(x)
#
#         r7 = self.relation7(x)
#         r8 = self.relation8(x)
#
#         with_r = self.with_relation(x)
#         x = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8, with_r], dim=1)
#         x = self.final(x)
#         x = nn.Sigmoid()(x)
#
#         r1 = nn.Sigmoid()(r1)
#         r2 = nn.Sigmoid()(r2)
#         r3 = nn.Sigmoid()(r3)
#         r4 = nn.Sigmoid()(r4)
#
#         r5 = nn.Sigmoid()(r5)
#         r6 = nn.Sigmoid()(r6)
#         r7 = nn.Sigmoid()(r7)
#         r8 = nn.Sigmoid()(r8)
#         # logits = self.outc(x)
#         return [x, r1, r2, r3, r4, r5, r6, r7, r8]
#

if __name__ == '__main__':
    model1 = UNetStage1(3,bilinear=True).cpu()
    # model2 = UNetStage2(4, bilinear=True).cpu()
    in_size = 320
    # summary(model=model1,(3,320,320),device='cpu',batch_size=2)
    # summary(model2, [(4, in_size, in_size), (128, in_size // 2, in_size // 2), (256, in_size // 4, in_size // 4),
    #                 (512, in_size // 8, in_size // 8)], device='cpu', batch_size=2)
    summary(model1, (3, 320, 320), device='cpu', batch_size=2)
