#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：forgery-edge-detection 
@File    ：r101_cls.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/2 1:42 PM 
'''
import torch
import torch.optim as optim
import torch.utils.data.dataloader
import os, sys
sys.path.append('../')
sys.path.append('../utils')
import argparse
import time, datetime
# from functions import my_f1_score, my_acc_score, my_precision_score ,CE_loss
# from torch.nn import init
from datasets.dataloader_with_cls import TamperDataset
# from PIL import Image
# import shutil
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from utils import Logger, Averagvalue, weights_init, load_pretrained
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
# from model.resnet import resnet101 as Net
from torchvision.models import resnet101 as Net
import wandb

""""""""""""""""""""""""""""""
"          参数               "
""""""""""""""""""""""""""""""

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=40, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--model_save_dir', type=str, help='model_save_dir',
                    default='save_model/test2')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=10, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
#####################resume##########################
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--per_epoch_freq', type=int, help='per_epoch_freq', default=50)

parser.add_argument('--fuse_loss_weight', type=int, help='fuse_loss_weight', default=12)
# ================ dataset

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

""""""""""""""""""""""""""""""
"          路径               "
""""""""""""""""""""""""""""""
model_save_dir = abspath(dirname(__file__))
model_save_dir = join(model_save_dir, args.model_save_dir)
if not isdir(model_save_dir):
    os.makedirs(model_save_dir)

""""""""""""""""""""""""""""""
"    ↓↓↓↓需要修改的参数↓↓↓↓     "
""""""""""""""""""""""""""""""

# wandb 使用

wandb.init(project="Forgery_cls", entity="muskai")
wandb.config = {
  "learning_rate": args.lr,
  "epochs": args.maxepoch,
  "batch_size": args.batch_size
}
wandb.watch_called = True

""""""""""""""""""""""""""""""
"    ↑↑↑↑需要修改的参数↑↑↑↑     "
""""""""""""""""""""""""""""""



""""""""""""""""""""""""""""""
"          程序入口            "
""""""""""""""""""""""""""""""


def main():
    args.cuda = True
    # 1 choose the data you want to use
    using_data = {'my_sp': False,
                  'my_cm': False,
                  'template_casia_casia': False,
                  'template_coco_casia': False,
                  'cod10k': True,
                  'casia': False,
                  'coverage': False,
                  'columb': False,
                  'negative': True,
                  'negative_casia': False,
                  'texture_sp': True,
                  'texture_cm': False,
                  }

    # 2 define 3 types
    trainData = TamperDataset(using_data=using_data, train_val_test_mode='train')
    valData = TamperDataset( using_data=using_data, train_val_test_mode='val')

    # 3 specific dataloader
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                                  pin_memory=True)
    # valDataLoader = torch.utils.data.DataLoader(valData, batch_size=args.batch_size, num_workers=4)

    # model
    model = Net(pretrained=False, num_classes=2)

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    model.apply(weights_init)
    # 模型初始化
    # 如果没有这一步会根据正态分布自动初始化
    # model.apply(weights_init)

    # 模型可持续化

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'".format(args.resume))
            # optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            print("=> 想要使用预训练模型，但是路径出错 '{}'".format(args.resume))
            sys.exit(1)

    else:
        print("=> 不使用预训练模型，直接开始训练 '{}'".format(args.resume))

    # 调整学习率
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    # 数据迭代器

    for epoch in range(args.start_epoch, args.maxepoch):
        train_avg = train(model=model, optimizer=optimizer, dataParser=trainDataLoader, epoch=epoch)
        # val_avg = val(model=model, dataParser=valDataLoader, epoch=epoch)

        # 保存模型
        output_name = 'epoch{}_loss{}.pth'.format(epoch,train_avg['loss_avg'])

        if epoch % 1 == 0:
            save_model_name = os.path.join(args.model_save_dir, output_name)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       save_model_name)

        scheduler.step(epoch=epoch)

    print('训练已完成!')


""""""""""""""""""""""""""""""
"           训练              "
""""""""""""""""""""""""""""""


def train(model, optimizer, dataParser, epoch):
    # 读取数据的迭代器

    train_epoch = len(dataParser)
    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()

    # switch to train mode
    model.train()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        # 准备输入数据
        images = input_data['tamper_image'].cuda()
        gt_cls = input_data['gt_cls'].long().cuda()
        gt_cls = gt_cls.view(-1)

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
        else:
            loss = torch.zeros(1)

        with torch.set_grad_enabled(True):
            images.requires_grad = True
            optimizer.zero_grad()
            # 网络输出

            output = model(images)

            """"""""""""""""""""""""""""""
            "         Loss 函数           "
            """"""""""""""""""""""""""""""

            loss = torch.nn.CrossEntropyLoss()(output,gt_cls)

            # 记录总loss值
            wandb.log({
                'loss': loss.item()
            })
            loss.backward()
            optimizer.step()


        # 将各种数据记录到专门的对象中
        losses.update(loss.item())


        torch.cuda.empty_cache()

        batch_time.update(time.time() - end)
        end = time.time()


        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, train_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses)

            print(info)

        if batch_index >= train_epoch:
            break

    return {'loss_avg': losses.avg}






if __name__ == '__main__':
    main()