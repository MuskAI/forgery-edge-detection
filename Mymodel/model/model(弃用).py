import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def side_branch(x, input_channel, output_channel, factor):
    x_ = nn.Conv2d(input_channel, output_channel, kernel_size=1)
    x = x_(x)
    kernel_size = (2 * factor, 2 * factor)
    x_1 = nn.ConvTranspose2d(output_channel, output_channel, kernel_size=kernel_size, stride=factor,padding=int(factor/2), bias=False)
    x = x_1(x)
    return x



def aspp(x, input_shape, out_stride):
    b0 = nn.Sequential(OrderedDict([
        ('b0_conv', nn.Conv2d(512, 128, kernel_size=1, padding_mode='replicate', bias=False)),
        ('b0_bn', nn.BatchNorm2d(128)),
        ('b0_relu', nn.ReLU(inplace=True))
    ]))
    # 可分离卷积
    b1 = nn.Sequential(OrderedDict([
       ('b1_depthwise_conv',
         nn.Conv2d(128, 128, kernel_size=3, groups=128, dilation=(6, 6), padding_mode='replicate', padding=6,
                   bias=False)),
        ('b1_bn', nn.BatchNorm2d(128)),
        ('b1_relu', nn.ReLU(inplace=True)),
        ('b1_conv', nn.Conv2d(128, 128, kernel_size=1, bias=False)),
        ('b1_bn', nn.BatchNorm2d(128)),
        ('b1_relu', nn.ReLU(inplace=True))
    ]))

    # 又是一个可分离卷积

    b2 = nn.Sequential(OrderedDict([
        ('b2_depthwise_conv',
         nn.Conv2d(128, 128, kernel_size=3, groups=128, dilation=(12, 12), padding_mode='replicate', padding=12,
                   bias=False)),
        ('b2_bn', nn.BatchNorm2d(128)),
        ('b2_relu', nn.ReLU(inplace=True)),
        ('b2_conv', nn.Conv2d(128, 128, kernel_size=1, bias=False)),
        ('b2_bn', nn.BatchNorm2d(128)),
        ('b2_relu', nn.ReLU(inplace=True))
    ]))

    b3 = nn.Sequential(OrderedDict([
        ('b3_depthwise_conv',
         nn.Conv2d(128, 128, kernel_size=3, groups=128, dilation=(12, 12), padding_mode='replicate', padding=12,
                   bias=False)),
        ('b3_bn', nn.BatchNorm2d(128)),
        ('b3_relu', nn.ReLU(inplace=True)),
        ('b3_conv', nn.Conv2d(128, 128, kernel_size=1, bias=False)),
        ('b3_bn', nn.BatchNorm2d(128)),
        ('b3_relu', nn.ReLU(inplace=True))
    ]))

    out_shape = int(input_shape[0] / out_stride)
    out_shape1 = int(input_shape[1] / out_stride)
    b4 = nn.Sequential(OrderedDict([
        ('b4_averagepool', nn.AvgPool2d(kernel_size=(out_shape, out_shape1))),
        ('b4_conv', nn.Conv2d(128, 128, kernel_size=1, bias=False)),
        ('b4_bn', nn.BatchNorm2d(128)),
        ('b4_relu', nn.ReLU(inplace=True)),
        ('b4_bilinearUpsampling', nn.UpsamplingBilinear2d(size=(out_shape, out_shape1), scale_factor=None))
    ]))

    b0_ = b0(x)
    b1_ = b1(b0_)
    b2_ = b2(b1_)
    b3_ = b3(b2_)
    b4_ = b4(b3_)
    x = torch.cat([b4_, b0_, b1_, b2_, b3_], dim=1)
    return x


class Net(nn.Module):
    def __init__(self, input_shape=(320, 320, 3)):
        super(Net, self).__init__()
        self.input_shape = input_shape
        self.block1 = nn.Sequential(OrderedDict([
            ('block1_conv1',
             nn.Conv2d(3, 64, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block1_relu', nn.ReLU(inplace=True)),
            ('block1_conv2',
             nn.Conv2d(64, 64, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block1_conv2_bn', nn.BatchNorm2d(64)),
            ('block1_relu', nn.ReLU(inplace=True))
        ]))
        self.block1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 160 160 64

        self.block2 = nn.Sequential(OrderedDict([
            ('block2_conv1', nn.Conv2d(64, 128, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block2_relu', nn.ReLU(inplace=True)),
            ('block2_conv2', nn.Conv2d(128, 128, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block2_relu', nn.ReLU(inplace=True)),
            ('block2_conv2_bn', nn.BatchNorm2d(128)),
            ('block2_relu', nn.ReLU(inplace=True)),
            ('block2_dropout', nn.Dropout2d(0.5))
        ]))
        self.block2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 80 80 128

        self.block3 = nn.Sequential(OrderedDict([
            ('block3_conv1', nn.Conv2d(128, 256, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block3_relu', nn.ReLU(inplace=True)),
            ('block3_conv2', nn.Conv2d(256, 256, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block3_relu', nn.ReLU(inplace=True)),
            ('block3_conv3', nn.Conv2d(256, 256, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block3_relu', nn.ReLU(inplace=True)),
            ('block3_bn', nn.BatchNorm2d(256)),
            ('block3_relu', nn.ReLU(inplace=True)),
            ('block3_dropout', nn.Dropout2d(0.5))
        ]))
        self.block3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 80 80 256

        self.block4 = nn.Sequential(OrderedDict([
            ('block4_conv1', nn.Conv2d(256, 512, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block4_relu', nn.ReLU(inplace=True)),

            ('block4_conv2', nn.Conv2d(512, 512, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block4_relu', nn.ReLU(inplace=True)),

            ('block4_conv3', nn.Conv2d(512, 512, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block4_relu', nn.ReLU(inplace=True)),

            ('block4_bn', nn.BatchNorm2d(512)),
            ('block4_dropput', nn.Dropout2d(0.5)),
            ('block4_relu', nn.ReLU(inplace=True))
        ]))
        self.block4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 40 40 512

        self.block5 = nn.Sequential(OrderedDict([
            ('block5_conv1', nn.Conv2d(512, 512, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block4_relu', nn.ReLU(inplace=True)),

            ('block5_conv2', nn.Conv2d(512, 512, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block5_relu', nn.ReLU(inplace=True)),

            ('block5_conv3', nn.Conv2d(512, 512, kernel_size=3, padding_mode='replicate', padding=1, bias=False)),
            ('block5_relu', nn.ReLU(inplace=True)),

            ('block5_bn', nn.BatchNorm2d(512)),
            ('block5_relu', nn.ReLU(inplace=True)),

        ]))
        self.block5_dropout = nn.Dropout2d(0, 5)
        self.block5_continue = nn.Sequential(OrderedDict([
            ('block5_conv4', nn.Conv2d(640, 512, kernel_size=1)),
            ('block5_relu', nn.ReLU(inplace=True))
        ]))

        self.up1 = nn.Sequential(OrderedDict([
            ('up1_subpixel', nn.PixelShuffle(upscale_factor=2))
        ]))

        # self.up1_b4 = side_branch()

        self.up2 = nn.Sequential(OrderedDict([
            ('up2_conv1', nn.Conv2d(640, 512, kernel_size=1)),
            ('up2_relu', nn.ReLU(inplace=True)),
            ('up2_subpixel', nn.PixelShuffle(upscale_factor=2))
        ]))

        # self.up2_b3 = side_branch()

        self.up3 = nn.Sequential(OrderedDict([
            ('up3_conv1', nn.Conv2d(384, 128, kernel_size=1)),
            ('up3_relu', nn.ReLU(inplace=True)),
            ('up3_subpixel', nn.PixelShuffle(upscale_factor=2))
        ]))

        # self.up3_b2 = side_branch()

        self.up4 = nn.Sequential(OrderedDict([
            ('up3_conv1', nn.Conv2d(160, 128, kernel_size=1)),
            ('up3_relu', nn.ReLU(inplace=True)),
            ('up3_subpixel', nn.PixelShuffle(upscale_factor=2))
        ]))

        self.relation_map1 = nn.Conv2d(96, 2, kernel_size=3, padding_mode='replicate', padding=1)
        self.relation_map2 = nn.Conv2d(96, 2, kernel_size=3, padding_mode='replicate', padding=1)
        self.relation_map3 = nn.Conv2d(96, 2, kernel_size=3, padding_mode='replicate', padding=1)

        self.relation_map4 = nn.Conv2d(96, 2, kernel_size=3, padding_mode='replicate', padding=1)
        self.relation_map5 = nn.Conv2d(96, 2, kernel_size=3, padding_mode='replicate', padding=1)
        self.relation_map6 = nn.Conv2d(96, 2, kernel_size=3, padding_mode='replicate', padding=1)
        self.relation_map7 = nn.Conv2d(96, 2, kernel_size=3, padding_mode='replicate', padding=1)
        self.relation_map8 = nn.Conv2d(96, 2, kernel_size=3, padding_mode='replicate', padding=1)

        self.fuse = nn.Conv2d(19, 1, kernel_size=1, padding_mode='replicate', bias=False)


        self.o2 = nn.Sigmoid()
        self.o3 = nn.Sigmoid()
        self.o4 = nn.Sigmoid()
        self.relation_map1_ = nn.Sigmoid()
        self.relation_map2_ = nn.Sigmoid()
        self.relation_map3_ = nn.Sigmoid()
        self.relation_map4_ = nn.Sigmoid()
        self.relation_map5_ = nn.Sigmoid()
        self.relation_map6_ = nn.Sigmoid()
        self.relation_map7_ = nn.Sigmoid()
        self.relation_map8_ = nn.Sigmoid()
        self.fuse_ = nn.Sigmoid()

    def forward(self, x):
        x1 = self.block1(x)
        x = self.block1_maxpool(x1)

        x2 = self.block2(x)
        x = self.block2_maxpool(x2)

        x3 = self.block3(x)
        x = self.block3_maxpool(x3)

        x4 = self.block4(x)
        x = self.block4_maxpool(x4)

        x5 = self.block5(x)
        x5 = aspp(x5, input_shape=self.input_shape, out_stride=16)
        x5 = self.block5_dropout(x5)
        x5 = self.block5_continue(x5)

        up = self.up1(x5)
        up = torch.cat([x4, up], dim=1)
        b4 = side_branch(up, 640, 1, 8)

        up = self.up2(up)
        up = torch.cat([x3, up], dim=1)
        b3 = side_branch(up, 384, 1, 4)

        up = self.up3(up)
        up = torch.cat([x2, up], dim=1)
        b2 = side_branch(up, 160, 1, 2)

        up = self.up4(up)
        final = torch.cat([x1, up], dim=1)

        relation_map1 = self.relation_map1(final)
        relation_map2 = self.relation_map2(final)
        relation_map3 = self.relation_map3(final)
        relation_map4 = self.relation_map4(final)
        relation_map5 = self.relation_map5(final)
        relation_map6 = self.relation_map6(final)
        relation_map7 = self.relation_map7(final)
        relation_map8 = self.relation_map8(final)

        # fuse = torch.cat(
        #     [b2, b3, b4, relation_map1, relation_map2, relation_map3, relation_map4, relation_map5, relation_map6,
        #      relation_map7, relation_map8], dim=1)
        # fuse = self.fuse(fuse)
        fuse =None
        o2 = self.o2(b2)
        o3 = self.o3(b3)
        o4 = self.o4(b4)

        relation_map1_ = self.relation_map1_(relation_map1)
        relation_map2_ = self.relation_map2_(relation_map2)
        relation_map3_ = self.relation_map3_(relation_map3)
        relation_map4_ = self.relation_map4_(relation_map4)
        relation_map5_ = self.relation_map5_(relation_map5)
        relation_map6_ = self.relation_map6_(relation_map6)
        relation_map7_ = self.relation_map7_(relation_map7)
        relation_map8_ = self.relation_map8_(relation_map8)
        return [relation_map1_, relation_map2_, relation_map3_, relation_map4_, relation_map5_, relation_map6_, relation_map7_, relation_map8_, fuse, o2, o3, o4]


if __name__ == '__main__':
    model = Net()
    print(model)
