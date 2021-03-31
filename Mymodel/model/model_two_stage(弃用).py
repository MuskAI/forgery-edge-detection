import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary
# from tensorboardX import SummaryWriter
import numpy as np

class SRMConv(nn.Module):
    def __init__(self, channels=3, kernel='filter1'):
        super(SRMConv, self).__init__()
        self.channels = channels
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        if kernel == 'filter1':
            kernel = filter1
        elif kernel =='filter2':
            kernel = filter2
        elif kernel == 'filter3':
            kernel = filter3
        else:
            print('kernel error')
            exit(0)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x

class ResBlock_Stage_1(nn.Module):
    """
    残差模块
       self.res_block1 = ResBlock(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=2,
                                   kernel_size=3)
    """

    def __init__(self, in_channel, out_channel, mid_channel1, mid_channel2, kernel_size, stride=1):
        super(ResBlock_Stage_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_channel1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=mid_channel1, out_channels=mid_channel2, kernel_size=kernel_size, stride=1,
                               padding_mode='replicate', padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(mid_channel2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=mid_channel2, out_channels=out_channel, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class Aspp_Stage_1(nn.Module):
    """
    dilation rate:6 12 12
    """
    def __init__(self, input_shape, out_stride):
        super(Aspp_Stage_1, self).__init__()
        self.out_shape = int(input_shape[0] / out_stride)
        self.out_shape1 = int(input_shape[1] / out_stride)

        self.b0 = nn.Sequential(OrderedDict([
            ('b0_conv', nn.Conv2d(256, 128, kernel_size=1, padding_mode='replicate', bias=False)),
            ('b0_bn', nn.BatchNorm2d(128)),
            ('b0_relu', nn.ReLU(inplace=True))
        ]))
        # 可分离卷积
        self.b1 = nn.Sequential(OrderedDict([
            ('b1_depthwise_conv',
             nn.Conv2d(256, 256, kernel_size=3, groups=256, dilation=(6, 6), padding_mode='replicate', padding=6,
                       bias=False)),
            ('b1_bn', nn.BatchNorm2d(256)),
            ('b1_relu', nn.ReLU(inplace=True)),
            ('b1_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b1_bn', nn.BatchNorm2d(256)),
            ('b1_relu', nn.ReLU(inplace=True))
        ]))

        # 又是一个可分离卷积

        self.b2 = nn.Sequential(OrderedDict([
            ('b2_depthwise_conv',
             nn.Conv2d(256, 256, kernel_size=3, groups=256, dilation=(12, 12), padding_mode='replicate', padding=12,
                       bias=False)),
            ('b2_bn', nn.BatchNorm2d(256)),
            ('b2_relu', nn.ReLU(inplace=True)),
            ('b2_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b2_bn', nn.BatchNorm2d(256)),
            ('b2_relu', nn.ReLU(inplace=True))
        ]))

        self.b3 = nn.Sequential(OrderedDict([
            ('b3_depthwise_conv',
             nn.Conv2d(256, 256, kernel_size=3, groups=256, dilation=(12, 12), padding_mode='replicate', padding=12,
                       bias=False)),
            ('b3_bn', nn.BatchNorm2d(256)),
            ('b3_relu', nn.ReLU(inplace=True)),
            ('b3_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b3_bn', nn.BatchNorm2d(256)),
            ('b3_relu', nn.ReLU(inplace=True))
        ]))
        # self.b4_ = nn.AdaptiveAvgPool2d((1, 1))
        self.b4 = nn.Sequential(OrderedDict([
            ('b4_averagepool', nn.AdaptiveAvgPool2d((1,1))),
            ('b4_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b4_bn', nn.BatchNorm2d(128)),
            ('b4_relu', nn.ReLU(inplace=True)),
            ('b4_bilinearUpsampling',nn.UpsamplingBilinear2d(size=(self.out_shape, self.out_shape1), scale_factor=None))
        ]))


    def forward(self, x):
        b0_ = self.b0(x)
        b1_ = self.b1(x)
        b2_ = self.b2(x)
        b3_ = self.b3(x)
        # b4_ = self.b4(x)
        b4_ = self.b4(x)
        x = torch.cat([b4_, b0_, b1_, b2_, b3_], dim=1)
        return x

class Net_Stage_1(nn.Module):
    def __init__(self, input_shape=(320, 320, 3)):
        super(Net_Stage_1, self).__init__()
        self.input_shape = input_shape

        # Step1: 对输入的图进行处理
        self.in_conv_bn_relu = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding_mode='replicate', padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #######################################

        # Step2:第一个和第二个res_block + shortcut add

        self.res_block1 = ResBlock_Stage_1(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=1,
                                   kernel_size=3)
        self.res_block1_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.res_block2 = ResBlock_Stage_1(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=1,
                                   kernel_size=3)

        #####################################

        # Step3:

        self.res_block3 = ResBlock_Stage_1(in_channel=64, out_channel=128, mid_channel1=64, mid_channel2=64, stride=1,
                                   kernel_size=3)
        self.res_block3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        self.res_block4 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=64, mid_channel2=64, kernel_size=3)
        self.res_block5 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=64, mid_channel2=64, kernel_size=3)

        ####################################

        # Step4:

        self.res_block6 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=2)
        self.res_block6_shortcut = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )

        self.res_block7 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=1)
        self.res_block8 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=1)
        self.res_block9 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=1)
        # 下一个
        self.res_block10 = ResBlock_Stage_1(in_channel=128, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=2)
        self.res_block10_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1,stride=2),
            nn.BatchNorm2d(256)
        )

        self.res_block11 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block12 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block13 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block14 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block15 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)

        #####################################

        # Step5: aspp上面的几个模块
        self.res_block16 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=2)
        self.res_block16_shortcut = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1,stride=2),
            nn.BatchNorm2d(256)
        )

        self.res_block17 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block18 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)

        ######################################

        # Step6:Aspp和它下面的部分
        self.aspp = Aspp_Stage_1(input_shape=(320, 320, 3), out_stride=16)

        ####################################

        # Step7:开始上采样部分
        ## 20--->40部分

        self.aspp_below_20_40 = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip40_40 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        self.skip80_40 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.aspp_shortcut_40_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 40--->80解码器部分
        self.up40_80 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip80_80 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip160_80 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up40_80_shortcut_after_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 80--->160解码器部分
        self.up80_160 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip160_160_l = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip160_160_r = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip320_320 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up80_160_shortcut_after_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 160--->320解码器部分
        self.up160_320 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up160_320_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_final = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        ###################################

        # 结束上采样部分，开始做最后输出的准备

        ## 8张图
        self.relation_map_1 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )

        self.relation_map_2 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_3 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_4 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_5 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_6 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_7 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_8 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        ## 8张图旁边的skip
        self.relation_map_skip = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Sigmoid()
        )
        ##################################

        # 最后的输出
        self.fusion_out = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_conv_bn_relu(x)
        res_block = self.res_block1(x)
        res_block_shortcut = self.res_block1_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block2(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        # 这里的分支有点复杂，仔细一点

        ######### 320--->320 尺寸的跳跃连接
        skip_320_320 = self.skip320_320(x)
        #########################

        ## 左分支，MaxPooling2D,这里通过计算p=0.5，所以p = 1但是还需要验证一下
        """160 尺度"""
        maxpool_down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)


        ######### 160--->160尺寸的跳跃连接
        skip_160_160_r = self.skip160_160_r(maxpool_down)
        #########################

        res_block = self.res_block3(maxpool_down)
        res_block_shortcut = self.res_block3_shortcut(maxpool_down)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block4(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block5(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        ##### 160--->80尺度的跳远连接
        skip_160_80 = self.skip160_80(x)
        ###########################
        dropout = nn.Dropout2d(0.5)(x)
        ## 160--->160的跳跃连接
        skip_160_160_l = self.skip160_160_l(dropout)
        skip_160_shortcut = torch.cat([skip_160_160_l, skip_160_160_r], 1)
        #######################
        """80尺度"""
        res_block = self.res_block6(dropout)
        res_block_shortcut = self.res_block6_shortcut(dropout)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block7(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block8(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block9(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)
        ## 80--->40 skip
        """40尺度"""
        skip_80_40 = self.skip80_40(x)
        ##############################
        ## 第二个dropout的地方
        dropout = nn.Dropout2d(0.5)(x)
        #### 80 --->80尺度的skip
        skip_80_80 = self.skip80_80(dropout)
        skip_80_shortcut = torch.cat([skip_160_80, skip_80_80], 1)
        #######################

        ## 左分支
        res_block = self.res_block10(x)
        res_block_shortcut = self.res_block10_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block11(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block12(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block13(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block14(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block15(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        ## 第三个dropout的地方
        x = nn.Dropout2d(0.5)(x)
        ## 40--->40尺度的skip
        skip_40_40 = self.skip40_40(x)

        skip_40_shortcut = torch.cat([skip_40_40, skip_80_40], 1)
        #######################
        """20尺度"""
        res_block = self.res_block16(x)
        res_block_shortcut = self.res_block16_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)

        res_block = self.res_block17(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block18(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.Dropout2d(0.5)(x)

        x = self.aspp(x)
        # 20->40
        x = self.aspp_below_20_40(x)
        skip = self.aspp_shortcut_40_cat(skip_40_shortcut)
        up1 = torch.cat([x, skip], 1)

        #####################################
        # 下面就是准备输入第二个stage的图
        self.stage_1_out_40 = up1

        up1 = self.up40_80(up1)

        skip = self.up40_80_shortcut_after_cat(skip_80_shortcut)
        up2 = torch.cat([up1, skip], 1)

        self.stage_1_out_80 = up2
        up2 = self.up80_160(up2)

        skip = self.up80_160_shortcut_after_cat(skip_160_shortcut)
        up3 = torch.cat([up2, skip], 1)
        self.stage_1_out_160 = up3
        up3 = self.up160_320(up3)

        x = torch.cat([up3, skip_320_320], 1)

        x = self.up_final(x)
        relation_map_1 = self.relation_map_1(x)
        relation_map_2 = self.relation_map_2(x)
        relation_map_3 = self.relation_map_3(x)
        relation_map_4 = self.relation_map_4(x)

        relation_map_5 = self.relation_map_5(x)
        relation_map_6 = self.relation_map_6(x)
        relation_map_7 = self.relation_map_7(x)
        relation_map_8 = self.relation_map_8(x)

        relation_map_skip = self.relation_map_skip(x)

        x = torch.cat([relation_map_1, relation_map_2, relation_map_3, relation_map_4, relation_map_5, relation_map_6,
                       relation_map_7, relation_map_8, relation_map_skip], 1)
        x = self.fusion_out(x)
        #################################

        return [x,relation_map_1, relation_map_2, relation_map_3, relation_map_4, relation_map_5, relation_map_6,
                   relation_map_7, relation_map_8,self.stage_1_out_40,self.stage_1_out_80,self.stage_1_out_160]

class ResBlock_Stage_2(nn.Module):
    """
    残差模块
       self.res_block1 = ResBlock(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=2,
                                   kernel_size=3)
    """

    def __init__(self, in_channel, out_channel, mid_channel1, mid_channel2, kernel_size, stride=1):
        super(ResBlock_Stage_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_channel1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=mid_channel1, out_channels=mid_channel2, kernel_size=kernel_size, stride=1,
                               padding_mode='replicate', padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(mid_channel2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=mid_channel2, out_channels=out_channel, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class Aspp_Stage_2(nn.Module):
    """
    dilation rate:6 12 12
    """

    def __init__(self, input_shape, out_stride):
        super(Aspp_Stage_2, self).__init__()
        self.out_shape = int(input_shape[0] / out_stride)
        self.out_shape1 = int(input_shape[1] / out_stride)

        self.b0 = nn.Sequential(OrderedDict([
            ('b0_conv', nn.Conv2d(256, 128, kernel_size=1, padding_mode='replicate', bias=False)),
            ('b0_bn', nn.BatchNorm2d(128)),
            ('b0_relu', nn.ReLU(inplace=True))
        ]))
        # 可分离卷积
        self.b1 = nn.Sequential(OrderedDict([
            ('b1_depthwise_conv',
             nn.Conv2d(256, 256, kernel_size=3, groups=256, dilation=(6, 6), padding_mode='replicate', padding=6,
                       bias=False)),
            ('b1_bn', nn.BatchNorm2d(256)),
            ('b1_relu', nn.ReLU(inplace=True)),
            ('b1_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b1_bn', nn.BatchNorm2d(256)),
            ('b1_relu', nn.ReLU(inplace=True))
        ]))

        # 又是一个可分离卷积

        self.b2 = nn.Sequential(OrderedDict([
            ('b2_depthwise_conv',
             nn.Conv2d(256, 256, kernel_size=3, groups=256, dilation=(12, 12), padding_mode='replicate', padding=12,
                       bias=False)),
            ('b2_bn', nn.BatchNorm2d(256)),
            ('b2_relu', nn.ReLU(inplace=True)),
            ('b2_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b2_bn', nn.BatchNorm2d(256)),
            ('b2_relu', nn.ReLU(inplace=True))
        ]))

        self.b3 = nn.Sequential(OrderedDict([
            ('b3_depthwise_conv',
             nn.Conv2d(256, 256, kernel_size=3, groups=256, dilation=(12, 12), padding_mode='replicate', padding=12,
                       bias=False)),
            ('b3_bn', nn.BatchNorm2d(256)),
            ('b3_relu', nn.ReLU(inplace=True)),
            ('b3_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b3_bn', nn.BatchNorm2d(256)),
            ('b3_relu', nn.ReLU(inplace=True))
        ]))
        # self.b4_ = nn.AdaptiveAvgPool2d((1, 1))
        self.b4 = nn.Sequential(OrderedDict([
            ('b4_averagepool', nn.AdaptiveAvgPool2d((1, 1))),
            ('b4_conv', nn.Conv2d(256, 128, kernel_size=1, bias=False)),
            ('b4_bn', nn.BatchNorm2d(128)),
            ('b4_relu', nn.ReLU(inplace=True)),
            (
            'b4_bilinearUpsampling', nn.UpsamplingBilinear2d(size=(self.out_shape, self.out_shape1), scale_factor=None))
        ]))

    def forward(self, x):
        b0_ = self.b0(x)
        b1_ = self.b1(x)
        b2_ = self.b2(x)
        b3_ = self.b3(x)
        # b4_ = self.b4(x)
        b4_ = self.b4(x)
        x = torch.cat([b4_, b0_, b1_, b2_, b3_], dim=1)
        return x

class Net_Stage_2(nn.Module):
    def __init__(self, input_shape=(320, 320, 6)):
        super(Net_Stage_2, self).__init__()
        self.input_shape = input_shape

        # stage out
        self.stage_out_conv_bn_160 = nn.Sequential(
            nn.Conv2d(64+128, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.stage_out_conv_bn_80 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        self.stage_out_conv_bn_40 = nn.Sequential(
            nn.Conv2d(256 + 128, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )

        # Step1: 对输入的图进行处理
        self.in_conv_bn_relu = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, padding_mode='replicate', padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #######################################

        # Step2:第一个和第二个res_block + shortcut add

        self.res_block1 = ResBlock_Stage_1(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=1,
                                           kernel_size=3)
        self.res_block1_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.res_block2 = ResBlock_Stage_1(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=1,
                                           kernel_size=3)

        #####################################

        # Step3:

        self.res_block3 = ResBlock_Stage_1(in_channel=64, out_channel=128, mid_channel1=64, mid_channel2=64, stride=1,
                                           kernel_size=3)
        self.res_block3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        self.res_block4 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=64, mid_channel2=64,
                                           kernel_size=3)
        self.res_block5 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=64, mid_channel2=64,
                                           kernel_size=3)

        ####################################

        # Step4:

        self.res_block6 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128,
                                           kernel_size=3,
                                           stride=2)
        self.res_block6_shortcut = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )

        self.res_block7 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128,
                                           kernel_size=3,
                                           stride=1)
        self.res_block8 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128,
                                           kernel_size=3,
                                           stride=1)
        self.res_block9 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128,
                                           kernel_size=3,
                                           stride=1)
        # 下一个
        self.res_block10 = ResBlock_Stage_1(in_channel=128, out_channel=256, mid_channel1=128, mid_channel2=128,
                                            kernel_size=3,
                                            stride=2)
        self.res_block10_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256)
        )

        self.res_block11 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128,
                                            kernel_size=3,
                                            stride=1)
        self.res_block12 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128,
                                            kernel_size=3,
                                            stride=1)
        self.res_block13 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128,
                                            kernel_size=3,
                                            stride=1)
        self.res_block14 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128,
                                            kernel_size=3,
                                            stride=1)
        self.res_block15 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128,
                                            kernel_size=3,
                                            stride=1)

        #####################################

        # Step5: aspp上面的几个模块
        self.res_block16 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128,
                                            kernel_size=3,
                                            stride=2)
        self.res_block16_shortcut = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256)
        )

        self.res_block17 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128,
                                            kernel_size=3,
                                            stride=1)
        self.res_block18 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128,
                                            kernel_size=3,
                                            stride=1)

        ######################################

        # Step6:Aspp和它下面的部分
        self.aspp = Aspp_Stage_1(input_shape=(320, 320, 3), out_stride=16)

        ####################################

        # Step7:开始上采样部分
        ## 20--->40部分

        self.aspp_below_20_40 = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip40_40 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        self.skip80_40 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.aspp_shortcut_40_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 40--->80解码器部分
        self.up40_80 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip80_80 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip160_80 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up40_80_shortcut_after_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 80--->160解码器部分
        self.up80_160 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip160_160_l = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip160_160_r = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip320_320 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up80_160_shortcut_after_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 160--->320解码器部分
        self.up160_320 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up160_320_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_final = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        ###################################

        # 结束上采样部分，开始做最后输出的准备

        ## 8张图
        self.relation_map_1 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )

        self.relation_map_2 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_3 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_4 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_5 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_6 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_7 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_8 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        ## 8张图旁边的skip
        self.relation_map_skip = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Sigmoid()
        )
        ##################################

        # 最后的输出
        self.fusion_out = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x,stage_40,stage_80,stage_160):
        x = self.in_conv_bn_relu(x)
        res_block = self.res_block1(x)
        res_block_shortcut = self.res_block1_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block2(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        # 这里的分支有点复杂，仔细一点

        ######### 320--->320 尺寸的跳跃连接
        skip_320_320 = self.skip320_320(x)
        #########################

        ## 左分支，MaxPooling2D,这里通过计算p=0.5，所以p = 1但是还需要验证一下


        """stage out: 160 堆叠"""
        maxpool_down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        maxpool_down = torch.cat((maxpool_down,stage_160),1)
        maxpool_down = self.stage_out_conv_bn_160(maxpool_down)

        ######### 160--->160尺寸的跳跃连接
        skip_160_160_r = self.skip160_160_r(maxpool_down)
        #########################
        # 先cat 再1*1 64+128


        res_block = self.res_block3(maxpool_down)
        res_block_shortcut = self.res_block3_shortcut(maxpool_down)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block4(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block5(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        ##### 160--->80尺度的跳远连接
        skip_160_80 = self.skip160_80(x)
        ###########################
        dropout = nn.Dropout2d(0.5)(x)
        ## 160--->160的跳跃连接
        skip_160_160_l = self.skip160_160_l(dropout)
        skip_160_shortcut = torch.cat([skip_160_160_l, skip_160_160_r], 1)
        #######################
        """stage out：80的尺寸"""
        res_block = self.res_block6(dropout)
        ################################
        # stage 80
        res_block = torch.cat((res_block,stage_80), 1)
        res_block = self.stage_out_conv_bn_80(res_block)


        ##################################


        res_block_shortcut = self.res_block6_shortcut(dropout)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block7(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block8(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block9(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)
        ## 80--->40 skip
        skip_80_40 = self.skip80_40(x)
        ##############################
        ## 第二个dropout的地方
        dropout = nn.Dropout2d(0.5)(x)
        #### 80 --->80尺度的skip
        skip_80_80 = self.skip80_80(dropout)
        skip_80_shortcut = torch.cat([skip_160_80, skip_80_80], 1)
        #######################

        ## 左分支
        """stage out : 40的尺寸"""
        res_block = self.res_block10(x)
        ####################################
        # 256+128
        res_block = torch.cat((res_block,stage_40),1)
        res_block = self.stage_out_conv_bn_40(res_block)
        ##############################
        res_block_shortcut = self.res_block10_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block11(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block12(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block13(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block14(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block15(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        ## 第三个dropout的地方
        x = nn.Dropout2d(0.5)(x)
        ## 40--->40尺度的skip
        skip_40_40 = self.skip40_40(x)

        skip_40_shortcut = torch.cat([skip_40_40, skip_80_40], 1)
        #######################
        res_block = self.res_block16(x)
        res_block_shortcut = self.res_block16_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)

        res_block = self.res_block17(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block18(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.Dropout2d(0.5)(x)

        x = self.aspp(x)
        x = self.aspp_below_20_40(x)
        skip = self.aspp_shortcut_40_cat(skip_40_shortcut)
        up1 = torch.cat([x, skip], 1)
        up1 = self.up40_80(up1)

        skip = self.up40_80_shortcut_after_cat(skip_80_shortcut)
        up2 = torch.cat([up1, skip], 1)
        up2 = self.up80_160(up2)

        skip = self.up80_160_shortcut_after_cat(skip_160_shortcut)
        up3 = torch.cat([up2, skip], 1)
        up3 = self.up160_320(up3)

        x = torch.cat([up3, skip_320_320], 1)

        x = self.up_final(x)
        relation_map_1 = self.relation_map_1(x)
        relation_map_2 = self.relation_map_2(x)
        relation_map_3 = self.relation_map_3(x)
        relation_map_4 = self.relation_map_4(x)

        relation_map_5 = self.relation_map_5(x)
        relation_map_6 = self.relation_map_6(x)
        relation_map_7 = self.relation_map_7(x)
        relation_map_8 = self.relation_map_8(x)

        relation_map_skip = self.relation_map_skip(x)

        x = torch.cat([relation_map_1, relation_map_2, relation_map_3, relation_map_4, relation_map_5, relation_map_6,
                       relation_map_7, relation_map_8, relation_map_skip], 1)
        x = self.fusion_out(x)
        #################################
        return [x, relation_map_1, relation_map_2, relation_map_3, relation_map_4, relation_map_5, relation_map_6,
                relation_map_7, relation_map_8,up1,up2,up3]

class Two_Stage_Net(nn.Module):
    def __init__(self, input_shape=(320, 320, 3)):
        super(Two_Stage_Net, self).__init__()
        # super(Two_Stage_Net, self).__init__()
        self.combinehepler = CombineHelper()

    def forward(self,x):
        """
        Net_Stage_1:[x, 8张图，up1 up2 up3]
        :param x:
        :return:
        """
        rgb = x
        # 第一个网络开始计算
        stage_1_output = Net_Stage_1()(x)
        #########################################
        pred_1 = stage_1_output[0]
        stage_1_relation_1 = stage_1_output[1]
        stage_1_relation_2 = stage_1_output[2]
        stage_1_relation_3 = stage_1_output[3]
        stage_1_relation_4 = stage_1_output[4]
        stage_1_relation_5 = stage_1_output[5]
        stage_1_relation_6 = stage_1_output[6]
        stage_1_relation_7 = stage_1_output[7]
        stage_1_relation_8 = stage_1_output[8]
        ###########################################
        stage_1_40 = stage_1_output[9]
        stage_1_80 = stage_1_output[10]
        stage_1_160 = stage_1_output[11]
        ###########################################
        #
        # 准备输入第二个网络的东西
        rgb_pred = self.combinehepler.two_stage_input(rgb, pred_1)
        stage_2_output = Net_Stage_2()(rgb_pred,stage_1_40,stage_1_80,stage_1_160)
        pred_2 = stage_2_output[0]
        # 第二阶段
        stage_2_relation_1 = stage_2_output[1]
        stage_2_relation_2 = stage_2_output[2]
        stage_2_relation_3 = stage_2_output[3]
        stage_2_relation_4 = stage_2_output[4]
        stage_2_relation_5 = stage_2_output[5]
        stage_2_relation_6 = stage_2_output[6]
        stage_2_relation_7 = stage_2_output[7]
        stage_2_relation_8 = stage_2_output[8]
        ###########################################
        up80 = stage_2_output[9]
        up160 = stage_2_output[10]
        up320 = stage_2_output[11]


        """
        1. 返回一个元祖，第一个元素是第一阶段的结果
        2. 第二个元素是第二个阶段的结果
        3. rgb/rgb_pred, 8张图, 20 40 80 160 四个尺度
        """
        return [pred_1,stage_1_relation_1, stage_1_relation_2, stage_1_relation_3, stage_1_relation_4, stage_1_relation_5,
                stage_1_relation_6,stage_1_relation_7, stage_1_relation_8,stage_1_40,stage_1_80,stage_1_160],[pred_2, stage_2_relation_1, stage_2_relation_2, stage_2_relation_3, stage_2_relation_4, stage_2_relation_5,
                 stage_2_relation_6, stage_2_relation_7, stage_2_relation_8,up80,up160,up320]


class Net_Stage_1_SRM(nn.Module):
    def __init__(self, input_shape=(320, 320, 3)):
        super(Net_Stage_1_SRM, self).__init__()
        self.input_shape = input_shape

        # Step1: 对输入的图进行处理
        self.in_conv_bn_relu = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, padding_mode='replicate', padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #######################################

        # Step2:第一个和第二个res_block + shortcut add

        self.res_block1 = ResBlock_Stage_1(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=1,
                                   kernel_size=3)
        self.res_block1_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.res_block2 = ResBlock_Stage_1(in_channel=64, out_channel=64, mid_channel1=32, mid_channel2=32, stride=1,
                                   kernel_size=3)

        #####################################

        # Step3:

        self.res_block3 = ResBlock_Stage_1(in_channel=64, out_channel=128, mid_channel1=64, mid_channel2=64, stride=1,
                                   kernel_size=3)
        self.res_block3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        self.res_block4 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=64, mid_channel2=64, kernel_size=3)
        self.res_block5 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=64, mid_channel2=64, kernel_size=3)

        ####################################

        # Step4:

        self.res_block6 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=2)
        self.res_block6_shortcut = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )

        self.res_block7 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=1)
        self.res_block8 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=1)
        self.res_block9 = ResBlock_Stage_1(in_channel=128, out_channel=128, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                   stride=1)
        # 下一个
        self.res_block10 = ResBlock_Stage_1(in_channel=128, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=2)
        self.res_block10_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1,stride=2),
            nn.BatchNorm2d(256)
        )

        self.res_block11 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block12 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block13 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block14 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block15 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)

        #####################################

        # Step5: aspp上面的几个模块
        self.res_block16 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=2)
        self.res_block16_shortcut = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1,stride=2),
            nn.BatchNorm2d(256)
        )

        self.res_block17 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)
        self.res_block18 = ResBlock_Stage_1(in_channel=256, out_channel=256, mid_channel1=128, mid_channel2=128, kernel_size=3,
                                    stride=1)

        ######################################

        # Step6:Aspp和它下面的部分
        self.aspp = Aspp_Stage_1(input_shape=(320, 320, 3), out_stride=16)

        ####################################

        # Step7:开始上采样部分
        ## 20--->40部分

        self.aspp_below_20_40 = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip40_40 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        self.skip80_40 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.aspp_shortcut_40_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 40--->80解码器部分
        self.up40_80 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip80_80 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.skip160_80 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up40_80_shortcut_after_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 80--->160解码器部分
        self.up80_160 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip160_160_l = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip160_160_r = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.skip320_320 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up80_160_shortcut_after_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ## 160--->320解码器部分
        self.up160_320 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up160_320_shortcut = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_final = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        ###################################

        # 结束上采样部分，开始做最后输出的准备

        ## 8张图
        self.relation_map_1 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )

        self.relation_map_2 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_3 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_4 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_5 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_6 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_7 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        self.relation_map_8 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding_mode='replicate', padding=1),
            nn.Sigmoid(),
        )
        ## 8张图旁边的skip
        self.relation_map_skip = nn.Sequential(
            nn.Conv2d(128, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Sigmoid()
        )
        ##################################

        # 最后的输出
        self.fusion_out = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 8, kernel_size=3, padding_mode='replicate', padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        noise_3dim = SRMConv()(x)
        src_noise = torch.cat((x, noise_3dim), 1)
        x = self.in_conv_bn_relu(src_noise)

        res_block = self.res_block1(x)
        res_block_shortcut = self.res_block1_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block2(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        # 这里的分支有点复杂，仔细一点

        ######### 320--->320 尺寸的跳跃连接
        skip_320_320 = self.skip320_320(x)
        #########################

        ## 左分支，MaxPooling2D,这里通过计算p=0.5，所以p = 1但是还需要验证一下
        """160 尺度"""
        maxpool_down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)


        ######### 160--->160尺寸的跳跃连接
        skip_160_160_r = self.skip160_160_r(maxpool_down)
        #########################

        res_block = self.res_block3(maxpool_down)
        res_block_shortcut = self.res_block3_shortcut(maxpool_down)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block4(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block5(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        ##### 160--->80尺度的跳远连接
        skip_160_80 = self.skip160_80(x)
        ###########################
        dropout = nn.Dropout2d(0.5)(x)
        ## 160--->160的跳跃连接
        skip_160_160_l = self.skip160_160_l(dropout)
        skip_160_shortcut = torch.cat([skip_160_160_l, skip_160_160_r], 1)
        #######################
        """80尺度"""
        res_block = self.res_block6(dropout)
        res_block_shortcut = self.res_block6_shortcut(dropout)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block7(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block8(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block9(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)
        ## 80--->40 skip
        """40尺度"""
        skip_80_40 = self.skip80_40(x)
        ##############################
        ## 第二个dropout的地方
        dropout = nn.Dropout2d(0.5)(x)
        #### 80 --->80尺度的skip
        skip_80_80 = self.skip80_80(dropout)
        skip_80_shortcut = torch.cat([skip_160_80, skip_80_80], 1)
        #######################

        ## 左分支
        res_block = self.res_block10(x)
        res_block_shortcut = self.res_block10_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block11(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block12(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block13(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block14(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block15(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        ## 第三个dropout的地方
        x = nn.Dropout2d(0.5)(x)
        ## 40--->40尺度的skip
        skip_40_40 = self.skip40_40(x)

        skip_40_shortcut = torch.cat([skip_40_40, skip_80_40], 1)
        #######################
        """20尺度"""
        res_block = self.res_block16(x)
        res_block_shortcut = self.res_block16_shortcut(x)
        x = torch.add(res_block, res_block_shortcut)

        res_block = self.res_block17(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)

        res_block = self.res_block18(x)
        x = torch.add(res_block, x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.Dropout2d(0.5)(x)

        x = self.aspp(x)
        # 20->40
        x = self.aspp_below_20_40(x)
        skip = self.aspp_shortcut_40_cat(skip_40_shortcut)
        up1 = torch.cat([x, skip], 1)

        #####################################
        # 下面就是准备输入第二个stage的图
        self.stage_1_out_40 = up1

        up1 = self.up40_80(up1)

        skip = self.up40_80_shortcut_after_cat(skip_80_shortcut)
        up2 = torch.cat([up1, skip], 1)

        self.stage_1_out_80 = up2
        up2 = self.up80_160(up2)

        skip = self.up80_160_shortcut_after_cat(skip_160_shortcut)
        up3 = torch.cat([up2, skip], 1)
        self.stage_1_out_160 = up3
        up3 = self.up160_320(up3)

        x = torch.cat([up3, skip_320_320], 1)

        x = self.up_final(x)
        relation_map_1 = self.relation_map_1(x)
        relation_map_2 = self.relation_map_2(x)
        relation_map_3 = self.relation_map_3(x)
        relation_map_4 = self.relation_map_4(x)

        relation_map_5 = self.relation_map_5(x)
        relation_map_6 = self.relation_map_6(x)
        relation_map_7 = self.relation_map_7(x)
        relation_map_8 = self.relation_map_8(x)

        relation_map_skip = self.relation_map_skip(x)

        x = torch.cat([relation_map_1, relation_map_2, relation_map_3, relation_map_4, relation_map_5, relation_map_6,
                       relation_map_7, relation_map_8, relation_map_skip], 1)
        x = self.fusion_out(x)
        #################################

        return [x,relation_map_1, relation_map_2, relation_map_3, relation_map_4, relation_map_5, relation_map_6,
                   relation_map_7, relation_map_8,self.stage_1_out_40,self.stage_1_out_80,self.stage_1_out_160]

"""程序入口, 两阶段联合训练"""
class CombineHelper():
    def __init__(self):
        super(CombineHelper, self).__init__()
        pass
    def two_stage_input(self, rgb_img, band_pred):
        """
        (N,C,H,W)
        :param rgb_img:
        :param band_pred:
        :return:
        """
        return torch.cat((rgb_img * band_pred,rgb_img,),1)

    def __two_stage_input_input_check(self, rgb_img, band_pred):
        pass

    def __show_rgb_pred_result(self):
        pass


if __name__ == '__main__':
    # xx = torch.rand(2,3,320,320).cuda()
    # model = Net().cuda()
    # writer.add_graph(model,xx)
    # summary(model,(3,320,320))
    # print('ok')
    x = torch.rand(2,3,320,320).cpu()
    model = Two_Stage_Net(x).cpu()
    # summary(model,(3,320,320))
    print('ok')