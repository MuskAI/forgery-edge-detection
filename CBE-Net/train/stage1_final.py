#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：forgery-edge-detection 
@File    ：stage1-final.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/8 3:36 PM
最新的网络模型
'''
import math

import torch
import torch.optim as optim
import torch.utils.data.dataloader
import os, sys
import asyncio
import numpy as np
import wandb

sys.path.append('.')
sys.path.append('../')
sys.path.append('../utils')
import argparse
import time, datetime
from functions import my_f1_score, my_acc_score, my_precision_score, weighted_cross_entropy_loss, wce_huber_loss, \
    map8_loss_ce, my_recall_score, cross_entropy_loss, wce_dice_huber_loss
from datasets.dataloader_final import TamperDataset
from model.model_final import UNetStage1 as Net1
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from utils import Logger, Averagvalue, weights_init, load_pretrained, save_mid_result, send_msn
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

""""""""""""""""""""""""""""""
"       1 参数设置区           "
""""""""""""""""""""""""""""""

work_dirs = '0417-stage1'

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=28, type=int, metavar='BT',
                    help='batch size')

# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default=['', ''], type=list, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--maxepoch', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--model_save_dir', type=str, help='model_save_dir',
                    default='../save_model/' + work_dirs)

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=10, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')

parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc

parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')

# ================ dataset

parser.add_argument('--save_mid_result', help='weather save mid result', type=bool, default=False)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_save_dir = abspath(dirname(__file__))
model_save_dir = join(model_save_dir, args.model_save_dir)

if not isdir(model_save_dir):
    os.makedirs(model_save_dir)

""""""""""""""""""""""""""""""
"         2程序入口           "
""""""""""""""""""""""""""""""


class Trainer:
    """
    用来统一训练和测试，一键实现测试。
    version1：实现训练、验证、记录训练数据
    version2：测试（跟踪一个指定的数据、满足某一条件后自动测试指标）
    version3：GPU实现指标计算
    """

    def __init__(self):
        """
        配置训练的参数
        """
        # 1 数据相关的参数
        using_data = {'my_sp': True,
                      'my_cm': True,
                      'template_casia_casia': True,
                      'template_coco_casia': True,
                      'cod10k': True,
                      'casia': False,
                      'copy_move': False,
                      'texture_sp': True,
                      'texture_cm': False,
                      'columb': False,
                      'negative': True,
                      'negative_casia': False,
                      }
        using_data_test = {'my_sp': False,
                           'my_cm': False,
                           'template_casia_casia': False,
                           'template_coco_casia': False,
                           'cod10k': False,
                           'casia': False,
                           'coverage': True,
                           'columb': False,
                           'negative': False,
                           'negative_casia': False,
                           }

        trainData = TamperDataset(using_data=using_data, train_val_test_mode='train')
        valData = TamperDataset(using_data=using_data, train_val_test_mode='val')
        testData = TamperDataset(using_data=using_data_test, train_val_test_mode='test')

        self.trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=args.batch_size, shuffle=True,
                                                           num_workers=8)
        self.valDataLoader = torch.utils.data.DataLoader(valData, batch_size=args.batch_size, shuffle=True,
                                                         num_workers=4)

        # testDataLoader = torch.utils.data.DataLoader(testData, batch_size=1, num_workers=0)
        ######################

        # 2 模型训练相关的参数
        self.model1 = Net1()
        if torch.cuda.is_available():
            self.model1.cuda()
        else:
            self.model1.cpu()
        # 模型初始化
        self.model1.apply(weights_init)

        # 优化器
        self.optimizer1 = optim.Adam(self.model1.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)

        # 加载模型
        if args.resume[0] == '':
            print('不导入模型')
        elif isfile(args.resume[0]):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint1 = torch.load(args.resume[0])
            self.model1.load_state_dict(checkpoint1['state_dict'])
            print("=> loaded checkpoint '{}'".format(args.resume[0]))
        else:
            print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))
            raise '模型导入失败！'

        # 调整学习率
        self.scheduler1 = lr_scheduler.StepLR(self.optimizer1, step_size=args.stepsize, gamma=args.gamma)

        try:
            wandb.init(project="forgery-edge-detection", entity="muskai")
            pass

        except:
            pass

    def check_data(self):
        pass

    def check_pipeline(self):
        """
        使用一个小批量的数据检查数据通路
        @return:
        """
        pass

    def train_val(self):
        """
        train+val
        @return:
        """
        # 数据迭代器
        for epoch in range(args.start_epoch, args.maxepoch):
            train_avg = self.train(epoch=epoch)

            val_avg = self.val(epoch=epoch)
            # self.test(model1=model1, dataParser=testDataLoader, epoch=epoch)

            """"""""""""""""""""""""""""""
            "          写入图             "
            """"""""""""""""""""""""""""""
            try:
                pass
            except Exception as e:
                print(e)

            """"""""""""""""""""""""""""""
            "          写入图            "
            """"""""""""""""""""""""""""""

            output_name_file_name = 'epoch_%d-%.4f-f.4f-precision%.4f-acc%.4f-recall%.4f.pth'
            output_name1 = output_name_file_name % \
                           (epoch,
                            val_avg['f1_avg_stage1'],
                            val_avg['precision_avg_stage1'],
                            val_avg['accuracy_avg_stage1'],
                            val_avg['recall_avg_stage1'])
            if epoch % 1 == 0:
                save_model_name_stage1 = os.path.join(args.model_save_dir, 'stage1_' + output_name1)
                torch.save(
                    {'epoch': epoch, 'state_dict': self.model1.state_dict(), 'optimizer': self.optimizer1.state_dict()},
                    save_model_name_stage1)
            self.scheduler1.step(epoch=epoch)
        print('训练已完成!')

    async def writer(self, write_type='loss'):
        """
        实验数据记录等
        @return:
        """

        pass

    @staticmethod
    def __show(img, gt):
        """
        显示输入网络的图片 和 gt
        @return:
        """
        _img = img.cpu().numpy()
        _gt = gt.cpu().numpy()

        batch_size = _gt.shape[0]
        print(batch_size)
        subplot_len = batch_size*2
        count = 1
        for idx,i in enumerate(_img):
            _i = np.transpose(i, (1, 2, 0))
            _g = np.transpose(_gt[idx],(1, 2, 0))
            plt.subplot(int(math.sqrt(subplot_len)+1),int(math.sqrt(subplot_len)+1),count)
            plt.imshow(_i)
            count += 1
            plt.subplot(int(math.sqrt(subplot_len) + 1), int(math.sqrt(subplot_len) + 1), count)
            plt.imshow(_g)
            count += 1
        plt.show()

    def train(self, epoch):
        dataloader = self.trainDataLoader
        train_epoch = len(dataloader)

        # 变量保存
        batch_time = Averagvalue()
        data_time = Averagvalue()
        losses = Averagvalue()
        loss_stage1 = Averagvalue()

        # switch to train mode
        self.model1.train()
        end = time.time()
        for batch_index, input_data in enumerate(dataloader):
            # 读取数据的时间
            data_time.update(time.time() - end)

            # 准备输入数据
            images = input_data['tamper_image'].cuda()
            labels_band = input_data['gt_band'].cuda()

            with torch.set_grad_enabled(True):
                # images.requires_grad = True
                self.optimizer1.zero_grad()
                one_stage_outputs = self.model1(images)
                # 建立loss
                loss_stage_1 = wce_dice_huber_loss(one_stage_outputs['logits'], labels_band)
                # self.__show(images, labels_band)
                loss = loss_stage_1
                loss.backward()
                self.optimizer1.step()

            # 将各种数据记录到专门的对象中
            losses.update(loss.item())
            wandb.log({'loss': loss_stage_1.item()})
            loss_stage1.update(loss_stage_1.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % args.print_freq == 0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, train_epoch) + \
                       'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                       '第一阶段Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=loss_stage1)
                print(info)

        return losses.avg

    @torch.no_grad()
    def val(self, epoch):
        # 读取数据的迭代器
        dataloader = self.valDataLoader
        val_epoch = len(dataloader)
        # 变量保存
        batch_time = Averagvalue()
        data_time = Averagvalue()
        losses = Averagvalue()
        loss_stage1 = Averagvalue()

        f1_value_stage1 = Averagvalue()
        acc_value_stage1 = Averagvalue()
        recall_value_stage1 = Averagvalue()
        precision_value_stage1 = Averagvalue()

        # switch to train mode
        self.model1.eval()

        end = time.time()

        for batch_index, input_data in enumerate(dataloader):
            # 读取数据的时间
            data_time.update(time.time() - end)
            # 准备输入数据
            images = input_data['tamper_image'].cuda()
            labels_band = input_data['gt_band'].cuda()

            with torch.set_grad_enabled(False):
                images.requires_grad = False
                # 网络输出
                one_stage_outputs = self.model1(images)

                # deal with one stage issue
                # 建立loss
                loss_stage_1 = wce_dice_huber_loss(one_stage_outputs['logits'], labels_band)

                loss = loss_stage_1
                #######################################

            # 将各种数据记录到专门的对象中
            losses.update(loss.item())
            loss_stage1.update(loss_stage_1.item())
            # loss_stage2.update(loss_stage_2.item())

            batch_time.update(time.time() - end)
            end = time.time()

            f1score_stage1 = my_f1_score(one_stage_outputs['logits'], labels_band)
            precisionscore_stage1 = my_precision_score(one_stage_outputs['logits'], labels_band)
            accscore_stage1 = my_acc_score(one_stage_outputs['logits'], labels_band)
            recallscore_stage1 = my_recall_score(one_stage_outputs['logits'], labels_band)
            f1_value_stage1.update(f1score_stage1)
            precision_value_stage1.update(precisionscore_stage1)
            acc_value_stage1.update(accscore_stage1)
            recall_value_stage1.update(recallscore_stage1)

            if batch_index % args.print_freq == 0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, val_epoch) + \
                       'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                       '两阶段总Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                       '第一阶段Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=loss_stage1) + \
                       '第一阶段:f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_stage1) + \
                       '第一阶段:precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(
                           precision=precision_value_stage1) + \
                       '第一阶段:acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_stage1) + \
                       '第一阶段:recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_stage1)
                print(info)

            if batch_index >= val_epoch:
                break

        return {'loss_avg': losses.avg,
                'f1_avg_stage1': f1_value_stage1.avg,
                'precision_avg_stage1': precision_value_stage1.avg,
                'accuracy_avg_stage1': acc_value_stage1.avg,
                'recall_avg_stage1': recall_value_stage1.avg,
                }

    # @staticmethod
    # @torch.no_grad()
    # def test(model1, dataParser, epoch):
    #     # 读取数据的迭代器
    #     # 变量保存
    #     batch_time = Averagvalue()
    #     data_time = Averagvalue()
    #
    #     # switch to train mode
    #     model1.eval()
    #     end = time.time()
    #
    #     for batch_index, input_data in enumerate(dataParser):
    #         # 读取数据的时间
    #         data_time.update(time.time() - end)
    #         # 准备输入数据
    #
    #         images = input_data['tamper_image'].cuda()
    #         # labels_dou_edge = input_data['gt_dou_edge'].cuda()
    #         # relation_map = input_data['relation_map']
    #         with torch.set_grad_enabled(False):
    #             images.requires_grad = False
    #             # 网络输出
    #             one_stage_outputs = model1(images)
    #
    #             """"""""""""""""""""""""""""""
    #             "         Loss 函数           "
    #             """"""""""""""""""""""""""""""
    #             ##########################################
    #             # deal with one stage issue
    #             # 建立loss
    #             writer.add_images('one&two_stage_image_batch:%d' % (batch_index),
    #                               one_stage_outputs[0], global_step=epoch)
    #


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_val()
