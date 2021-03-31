import torch
import torch.optim as optim
import torch.utils.data.dataloader
import os, sys
sys.path.append('../')
sys.path.append('../utils')
import argparse
import time, datetime
from functions import my_f1_score, my_acc_score, my_precision_score, weighted_cross_entropy_loss, wce_huber_loss, \
    wce_huber_loss_8, my_recall_score, cross_entropy_loss, wce_dice_huber_loss
from torch.nn import init
from datasets.dataloader import TamperDataset
from model.unet_two_stage_model_0306_2 import UNetStage1 as Net1
from model.unet_two_stage_model_0306_2 import UNetStage2 as Net2
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from utils import Logger, Averagvalue, weights_init, load_pretrained, save_mid_result, send_msn
from send_email import SendMail as send_email
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
"""
Created by HaoRan
time: 2021/01/29
description:
1. 单独训练stage 2
"""

""""""""""""""""""""""""""""""
"          参数               "
""""""""""""""""""""""""""""""

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=6, type=int, metavar='BT',
                    help='batch size')

# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default=['',
                                         ''], type=list, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=4, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=1000, type=int, metavar='N',
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

# parser.add_argument('--tmp', help='tmp folder', default='tmp/HED')
parser.add_argument('--mid_result_root', type=str, help='mid_result_root', default='./save')
parser.add_argument('--model_save_dir', type=str, help='model_save_dir',
                    default='../save_model/0306_stage1&2_后缀为0306_2的模型')
parser.add_argument('--mid_result_index', type=list, help='mid_result_index', default=[0])
parser.add_argument('--per_epoch_freq', type=int, help='per_epoch_freq', default=50)

parser.add_argument('--fuse_loss_weight', type=int, help='fuse_loss_weight', default=12)
# ================ dataset

# parser.add_argument('--dataset', help='root folder of dataset', default='dta/HED-BSD')
parser.add_argument('--band_mode', help='weather using band of normal gt', type=bool, default=True)
parser.add_argument('--save_mid_result', help='weather save mid result', type=bool, default=False)
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

# tensorboard 使用
# writer = SummaryWriter(
#     'runs/' + '0105_%d-%d_tensorboard' % (datetime.datetime.now().month, datetime.datetime.now().day))
writer = SummaryWriter(
    '/data-output/' + '0306_stage1&2_后缀为0306_2的模型')
email_header = 'Python'
output_name_file_name = '0306_stage1&2_后缀为0306_2的模型_checkpoint%d-two_stage-%f-f1%f-precision%f-acc%f-recall%f.pth'
""""""""""""""""""""""""""""""
"    ↑↑↑↑需要修改的参数↑↑↑↑     "
""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""
"          程序入口            "
""""""""""""""""""""""""""""""
email_list = []


def main():
    args.cuda = True
    # 1 choose the data you want to use
    using_data = {'my_sp': True,
                  'my_cm': True,
                  'template_casia_casia': True,
                  'template_coco_casia': True,
                  'cod10k': True,
                  'casia': False,
                  'copy_move': False,
                  'texture_sp': True,
                  'texture_cm': True,
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
    # 2 define 3 types
    trainData = TamperDataset(stage_type='stage2', using_data=using_data, train_val_test_mode='train',device='jly')
    valData = TamperDataset(stage_type='stage2', using_data=using_data, train_val_test_mode='val',device='jly')
    testData = TamperDataset(stage_type='stage2', using_data=using_data_test, train_val_test_mode='test',device='jly')

    # 3 specific dataloader
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=args.batch_size, num_workers=6, shuffle=True,
                                                  pin_memory=False)
    valDataLoader = torch.utils.data.DataLoader(valData, batch_size=args.batch_size, num_workers=6)

    testDataLoader = torch.utils.data.DataLoader(testData, batch_size=1, num_workers=0)
    # model
    model1 = Net1()
    model2 = Net2()
    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()
    else:
        model1.cpu()
        model2.cpu()

    # 模型初始化
    # 如果没有这一步会根据正态分布自动初始化
    model1.apply(weights_init)
    model2.apply(weights_init)

    # 模型可持续化
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
    if args.resume[0]:
        if isfile(args.resume[0]):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint1 = torch.load(args.resume[0])
            # checkpoint2 = torch.load(args.resume[1])
            model1.load_state_dict(checkpoint1['state_dict'])
            # optimizer1.load_state_dict(checkpoint1['optimizer'])
            ################################################
            # model2.load_state_dict(checkpoint2['state_dict'])
            # optimizer2.load_state_dict(checkpoint2['optimizer'])
            print("=> loaded checkpoint '{}'".format(args.resume))


        else:
            print("=> Error!!!! checkpoint found at '{}'".format(args.resume))

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # 调整学习率
    scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.stepsize, gamma=args.gamma)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=args.stepsize, gamma=args.gamma)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    # 数据迭代器

    for epoch in range(args.start_epoch, args.maxepoch):
        train_avg = train(model1=model1, model2=model2,optimizer1=optimizer1, optimizer2=optimizer2,
                          dataParser=trainDataLoader, epoch=epoch)

        val_avg = val(model1=model1, model2=model2, dataParser=valDataLoader, epoch=epoch)
        try:
            test_avg = test(model1=model1, model2=model2, dataParser=testDataLoader, epoch=epoch)
        except Exception as e:
            print(e)
            pass
        """"""""""""""""""""""""""""""
        "          写入图             "
        """"""""""""""""""""""""""""""
        try:
            writer.add_scalars('lr_per_epoch', {'stage1':scheduler1.get_lr(),
                                               'stage2':scheduler2.get_lr()}, global_step=epoch)
            writer.add_scalars('tr/val/test_avg_loss_per_epoch', {'train': train_avg['loss_avg'],
                                                                  'val': val_avg['loss_avg'],
                                                                  'test': test_avg['loss_avg']},
                               global_step=epoch)
            writer.add_scalars('tr/val/test_avg_f1_per_epoch', {'train': train_avg['f1_avg_stage2'],
                                                                'val': val_avg['f1_avg_stage2'],
                                                                'test': test_avg['f1_avg_stage2']}, global_step=epoch)

            writer.add_scalars('tr/val/test_avg_precision_per_epoch', {'train': train_avg['precision_avg_stage2'],
                                                                       'val': val_avg['precision_avg_stage2'],
                                                                       'test': test_avg['precision_avg_stage2']},
                               global_step=epoch)
            writer.add_scalars('tr/val/test_avg_acc_per_epoch', {'train': train_avg['accuracy_avg_stage2'],
                                                                 'val': val_avg['accuracy_avg_stage2'],
                                                                 'test': test_avg['accuracy_avg_stage2']},
                               global_step=epoch)
            writer.add_scalars('tr/val/test_avg_recall_per_epoch', {'train': train_avg['recall_avg_stage2'],
                                                                    'val': val_avg['recall_avg_stage2'],
                                                                    'test': test_avg['recall_avg_stage2']},
                               global_step=epoch)


        except Exception as e:
            print(e)

        """"""""""""""""""""""""""""""
        "          写入图            "
        """"""""""""""""""""""""""""""

        output_name = output_name_file_name % \
                      (epoch, val_avg['loss_avg'],
                       val_avg['f1_avg_stage2'],
                       val_avg['precision_avg_stage2'],
                       val_avg['accuracy_avg_stage2'],
                       val_avg['recall_avg_stage2'])

        try:
            # # send_msn(epoch, f1=val_avg['f1_avg'])
            # email_output_train = 'The train epoch:%d,f1:%f,loss:%f,precision:%f,accuracy:%f,recall:%f' % \
            #                      (epoch, train_avg['loss_avg'], train_avg['f1_avg'], train_avg['precision_avg'],
            #                       train_avg['accuracy_avg'], train_avg['recall_avg'])
            # email_output_val = 'The val epoch:%d,f1:%f,loss:%f,precision:%f,accuracy:%f,recall:%f' % \
            #                    (epoch, val_avg['loss_avg'], val_avg['f1_avg'], val_avg['precision_avg'],
            #                     val_avg['accuracy_avg'], val_avg['recall_avg'])
            # email_output_test = 'The test epoch:%d,f1:%f,loss:%f,precision:%f,accuracy:%f,recall:%f' % \
            #                     (epoch, test_avg['loss_avg'], test_avg['f1_avg'], test_avg['precision_avg'],
            #                      test_avg['accuracy_avg'], test_avg['recall_avg'])
            #
            # email_output = email_output_train + '\n' + email_output_val + '\n' + email_output_test + '\n\n\n'
            # email_list.append(email_output)
            # send_email(str(email_header), context=str(email_list))
            pass

        except:
            pass
        if epoch % 1 == 0:
            save_model_name_stage1 = os.path.join(args.model_save_dir, 'stage1' + output_name)
            save_model_name_stage2 = os.path.join(args.model_save_dir, 'stage2' + output_name)
            torch.save({'epoch': epoch, 'state_dict': model1.state_dict(), 'optimizer': optimizer1.state_dict()},
                       save_model_name_stage1)
            torch.save({'epoch': epoch, 'state_dict': model2.state_dict(), 'optimizer': optimizer2.state_dict()},
                       save_model_name_stage2)

        scheduler1.step(epoch=epoch)
        scheduler2.step(epoch=epoch)
    print('训练已完成!')


""""""""""""""""""""""""""""""
"           训练              "
""""""""""""""""""""""""""""""


def train(model1, model2, optimizer1,optimizer2, dataParser, epoch):
    # 读取数据的迭代器

    train_epoch = len(dataParser)
    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    loss_stage1 = Averagvalue()
    loss_stage2 = Averagvalue()

    f1_value_stage1 = Averagvalue()
    acc_value_stage1 = Averagvalue()
    recall_value_stage1 = Averagvalue()
    precision_value_stage1 = Averagvalue()

    f1_value_stage2 = Averagvalue()
    acc_value_stage2 = Averagvalue()
    recall_value_stage2 = Averagvalue()
    precision_value_stage2 = Averagvalue()
    map8_loss_value = Averagvalue()

    # switch to train mode
    model1.train()
    model2.train()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        # check_4dim_img_pair(input_data['tamper_image'],input_data['gt_band'])
        # 准备输入数据
        images = input_data['tamper_image'].cuda()
        labels_band = input_data['gt_band'].cuda()
        labels_dou_edge = input_data['gt_dou_edge'].cuda()
        relation_map = input_data['relation_map']

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
            loss_8t = torch.zeros(()).cuda()
        else:
            loss = torch.zeros(1)
            loss_8t = torch.zeros(())

        with torch.set_grad_enabled(True):
            images.requires_grad = True
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            if images.shape[1] != 3 or images.shape[2] != 320:
                continue
            # 网络输出

            try:
                one_stage_outputs = model1(images)
            except Exception as e:
                print(e)
                print(images.shape)
                continue

            rgb_pred = images * one_stage_outputs[0]
            rgb_pred_rgb = torch.cat((rgb_pred, images), 1)
            two_stage_outputs = model2(rgb_pred_rgb, one_stage_outputs[1], one_stage_outputs[2], one_stage_outputs[3])
            """"""""""""""""""""""""""""""
            "         Loss 函数           "
            """"""""""""""""""""""""""""""
            ##########################################
            # deal with one stage issue
            # 建立loss
            _loss_stage_1 = wce_dice_huber_loss(one_stage_outputs[0], labels_band)
            loss_stage_1 = _loss_stage_1
            ##############################################
            # deal with two stage issues
            _loss_stage_2 = wce_dice_huber_loss(two_stage_outputs[0], labels_dou_edge) * 12

            for c_index, c in enumerate(two_stage_outputs[1:9]):
                one_loss_t = cross_entropy_loss(c, relation_map[c_index].cuda())
                loss_8t += one_loss_t
                writer.add_scalar('stage2_%d_map_loss' % (c_index), one_loss_t.item(),
                                  global_step=epoch * train_epoch + batch_index)

            _loss_stage_2 += loss_8t
            loss_stage_2 = _loss_stage_2 / 20
            loss = loss_stage_2
            #######################################
            # 总的LOSS
            # print(type(loss_stage_2.item()))
            writer.add_scalars('loss_gather', {'stage_one_loss':loss_stage_1.item(),
                                             'stage_two_pred_loss':_loss_stage_2.item(),
                                             'stage_two_fuse_loss':loss_stage_2.item(),
                                             'fuse_loss_per_epoch':loss.item()}, global_step=epoch * train_epoch + batch_index)
            ##########################################

            loss.backward()
            optimizer1.step()
            optimizer2.step()

        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        loss_stage1.update(loss_stage_1.item())
        loss_stage2.update(loss_stage_2.item())

        map8_loss_value.update(loss_8t.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score_stage2 = my_f1_score(two_stage_outputs[0], labels_dou_edge)
        precisionscore_stage2 = my_precision_score(two_stage_outputs[0], labels_dou_edge)
        accscore_stage2 = my_acc_score(two_stage_outputs[0], labels_dou_edge)
        recallscore_stage2 = my_recall_score(two_stage_outputs[0], labels_dou_edge)

        f1score_stage1 = my_f1_score(one_stage_outputs[0], labels_band)
        precisionscore_stage1 = my_precision_score(one_stage_outputs[0], labels_band)
        accscore_stage1 = my_acc_score(one_stage_outputs[0], labels_band)
        recallscore_stage1 = my_recall_score(one_stage_outputs[0], labels_band)

        writer.add_scalars('f1_score_stage', {'stage1':f1score_stage1,
                                             'stage2':f1score_stage2}, global_step=epoch * train_epoch + batch_index)
        writer.add_scalars('precision_score_stage', {'stage1':precisionscore_stage1,
                                                     'stage2':precisionscore_stage2},global_step=epoch * train_epoch + batch_index)
        writer.add_scalars('acc_score_stage', {'stage1':accscore_stage1,
                                              'stage2':accscore_stage2}, global_step=epoch * train_epoch + batch_index)
        writer.add_scalars('recall_score_stage', {'stage1':recallscore_stage1,
                                                  'stage2':recallscore_stage2}, global_step=epoch * train_epoch + batch_index)
        ################################

        f1_value_stage1.update(f1score_stage1)
        precision_value_stage1.update(precisionscore_stage1)
        acc_value_stage1.update(accscore_stage1)
        recall_value_stage1.update(recallscore_stage1)

        f1_value_stage2.update(f1score_stage2)
        precision_value_stage2.update(precisionscore_stage2)
        acc_value_stage2.update(accscore_stage2)
        recall_value_stage2.update(recallscore_stage2)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, train_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   '两阶段总Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   '第一阶段Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=loss_stage1) + \
                   '第二阶段Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=loss_stage2) + \
                   '第一阶段:f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_stage1) + \
                   '第一阶段:precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(
                       precision=precision_value_stage1) + \
                   '第一阶段:acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_stage1) + \
                   '第一阶段:recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_stage1) + \
                   '第二阶段:f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_stage2) + \
                   '第二阶段:precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(
                       precision=precision_value_stage2) + \
                   '第二阶段:acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_stage2) + \
                   '第二阶段:recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_stage2)

            print(info)

        if batch_index >= train_epoch:
            break

    return {'loss_avg': losses.avg,
            'f1_avg_stage1': f1_value_stage1.avg,
            'precision_avg_stage1': precision_value_stage1.avg,
            'accuracy_avg_stage1': acc_value_stage1.avg,
            'recall_avg_stage1': recall_value_stage1.avg,

            'f1_avg_stage2': f1_value_stage2.avg,
            'precision_avg_stage2': precision_value_stage2.avg,
            'accuracy_avg_stage2': acc_value_stage2.avg,
            'recall_avg_stage2': recall_value_stage2.avg
            }


@torch.no_grad()
def val(model1, model2, dataParser, epoch):
    # 读取数据的迭代器

    val_epoch = len(dataParser)
    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    loss_stage1 = Averagvalue()
    loss_stage2 = Averagvalue()

    f1_value_stage1 = Averagvalue()
    acc_value_stage1 = Averagvalue()
    recall_value_stage1 = Averagvalue()
    precision_value_stage1 = Averagvalue()

    f1_value_stage2 = Averagvalue()
    acc_value_stage2 = Averagvalue()
    recall_value_stage2 = Averagvalue()
    precision_value_stage2 = Averagvalue()
    map8_loss_value = Averagvalue()

    # switch to train mode
    model1.eval()
    model2.eval()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        # check_4dim_img_pair(input_data['tamper_image'],input_data['gt_band'])
        # 准备输入数据
        images = input_data['tamper_image'].cuda()
        labels_band = input_data['gt_band'].cuda()
        labels_dou_edge = input_data['gt_dou_edge'].cuda()
        relation_map = input_data['relation_map']

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
            loss_8t = torch.zeros(()).cuda()
        else:
            loss = torch.zeros(1)
            loss_8t = torch.zeros(())

        with torch.set_grad_enabled(False):
            images.requires_grad = False
            # 网络输出
            one_stage_outputs = model1(images)

            rgb_pred = images * one_stage_outputs[0]
            rgb_pred_rgb = torch.cat((rgb_pred, images), 1)
            two_stage_outputs = model2(rgb_pred_rgb, one_stage_outputs[1], one_stage_outputs[2], one_stage_outputs[3])
            """"""""""""""""""""""""""""""
            "         Loss 函数           "
            """"""""""""""""""""""""""""""
            ##########################################
            # deal with one stage issue
            # 建立loss
            _loss_stage_1 = wce_dice_huber_loss(one_stage_outputs[0], labels_band)
            loss_stage_1 = _loss_stage_1
            ##############################################
            # deal with two stage issues
            _loss_stage_2 = wce_dice_huber_loss(two_stage_outputs[0], labels_dou_edge) * 12

            for c_index, c in enumerate(two_stage_outputs[1:9]):
                one_loss_t = cross_entropy_loss(c, relation_map[c_index].cuda())
                loss_8t += one_loss_t
                writer.add_scalar('stage2_%d_map_loss' % (c_index), one_loss_t.item(),
                                  global_step=epoch * val_epoch + batch_index)

            _loss_stage_2 += loss_8t
            loss_stage_2 = _loss_stage_2 / 20
            loss = (loss_stage_1 + loss_stage_2) / 2

            #######################################
            # 总的LOSS
            writer.add_scalar('val_stage_one_loss', loss_stage_1.item(), global_step=epoch * val_epoch + batch_index)
            writer.add_scalar('val_stage_two_pred_loss', _loss_stage_2.item(),
                              global_step=epoch * val_epoch + batch_index)
            writer.add_scalar('val_stage_two_fuse_loss', loss_stage_2.item(),
                              global_step=epoch * val_epoch + batch_index)

            writer.add_scalar('val_fuse_loss_per_epoch', loss.item(), global_step=epoch * val_epoch + batch_index)
            ##########################################

        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        loss_stage1.update(loss_stage_1.item())
        loss_stage2.update(loss_stage_2.item())

        map8_loss_value.update(loss_8t.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score_stage2 = my_f1_score(two_stage_outputs[0], labels_dou_edge)
        precisionscore_stage2 = my_precision_score(two_stage_outputs[0], labels_dou_edge)
        accscore_stage2 = my_acc_score(two_stage_outputs[0], labels_dou_edge)
        recallscore_stage2 = my_recall_score(two_stage_outputs[0], labels_dou_edge)

        f1score_stage1 = my_f1_score(one_stage_outputs[0], labels_band)
        precisionscore_stage1 = my_precision_score(one_stage_outputs[0], labels_band)
        accscore_stage1 = my_acc_score(one_stage_outputs[0], labels_band)
        recallscore_stage1 = my_recall_score(one_stage_outputs[0], labels_band)

        writer.add_scalar('val_f1_score_stage1', f1score_stage1, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_precision_score_stage1', precisionscore_stage1,
                          global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_acc_score_stage1', accscore_stage1, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_recall_score_stage1', recallscore_stage1, global_step=epoch * val_epoch + batch_index)

        writer.add_scalar('val_f1_score_stage2', f1score_stage2, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_precision_score_stage2', precisionscore_stage2,
                          global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_acc_score_stage2', accscore_stage2, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_recall_score_stage2', recallscore_stage2, global_step=epoch * val_epoch + batch_index)
        ################################

        f1_value_stage1.update(f1score_stage1)
        precision_value_stage1.update(precisionscore_stage1)
        acc_value_stage1.update(accscore_stage1)
        recall_value_stage1.update(recallscore_stage1)

        f1_value_stage2.update(f1score_stage2)
        precision_value_stage2.update(precisionscore_stage2)
        acc_value_stage2.update(accscore_stage2)
        recall_value_stage2.update(recallscore_stage2)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, val_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   '两阶段总Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   '第一阶段Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=loss_stage1) + \
                   '第二阶段Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=loss_stage2) + \
                   '第一阶段:f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_stage1) + \
                   '第一阶段:precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(
                       precision=precision_value_stage1) + \
                   '第一阶段:acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_stage1) + \
                   '第一阶段:recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_stage1) + \
                   '第二阶段:f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_stage2) + \
                   '第二阶段:precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(
                       precision=precision_value_stage2) + \
                   '第二阶段:acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_stage2) + \
                   '第二阶段:recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_stage2)

            print(info)

        if batch_index >= val_epoch:
            break

    return {'loss_avg': losses.avg,
            'f1_avg_stage1': f1_value_stage1.avg,
            'precision_avg_stage1': precision_value_stage1.avg,
            'accuracy_avg_stage1': acc_value_stage1.avg,
            'recall_avg_stage1': recall_value_stage1.avg,

            'f1_avg_stage2': f1_value_stage2.avg,
            'precision_avg_stage2': precision_value_stage2.avg,
            'accuracy_avg_stage2': acc_value_stage2.avg,
            'recall_avg_stage2': recall_value_stage2.avg
            }


@torch.no_grad()
def test(model1, model2, dataParser, epoch):
    # 读取数据的迭代器

    test_epoch = len(dataParser)
    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    loss_stage1 = Averagvalue()
    loss_stage2 = Averagvalue()

    f1_value_stage1 = Averagvalue()
    acc_value_stage1 = Averagvalue()
    recall_value_stage1 = Averagvalue()
    precision_value_stage1 = Averagvalue()

    f1_value_stage2 = Averagvalue()
    acc_value_stage2 = Averagvalue()
    recall_value_stage2 = Averagvalue()
    precision_value_stage2 = Averagvalue()
    map8_loss_value = Averagvalue()

    # switch to train mode
    model1.eval()
    model2.eval()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        # 准备输入数据
        images = input_data['tamper_image'].cuda()
        labels_band = input_data['gt_band'].cuda()
        labels_dou_edge = input_data['gt_dou_edge'].cuda()
        relation_map = input_data['relation_map']

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
            loss_8t = torch.zeros(()).cuda()
        else:
            loss = torch.zeros(1)
            loss_8t = torch.zeros(())

        with torch.set_grad_enabled(False):
            images.requires_grad = False
            # 网络输出
            one_stage_outputs = model1(images)

            rgb_pred = images * one_stage_outputs[0]
            rgb_pred_rgb = torch.cat((rgb_pred, images), 1)
            two_stage_outputs = model2(rgb_pred_rgb, one_stage_outputs[1], one_stage_outputs[2], one_stage_outputs[3])
            """"""""""""""""""""""""""""""
            "         Loss 函数           "
            """"""""""""""""""""""""""""""
            ##########################################
            # deal with one stage issue
            # 建立loss
            _loss_stage_1 = wce_dice_huber_loss(one_stage_outputs[0], labels_band)
            loss_stage_1 = _loss_stage_1
            ##############################################
            # deal with two stage issues
            _loss_stage_2 = wce_dice_huber_loss(two_stage_outputs[0], labels_dou_edge) * 12

            for c_index, c in enumerate(two_stage_outputs[1:9]):
                one_loss_t = cross_entropy_loss(c, relation_map[c_index].cuda())
                loss_8t += one_loss_t
                writer.add_scalar('stage2_%d_map_loss' % (c_index), one_loss_t.item(),
                                  global_step=epoch * test_epoch + batch_index)

            _loss_stage_2 += loss_8t
            loss_stage_2 = _loss_stage_2 / 20
            loss = (loss_stage_1 + loss_stage_2) / 2

            #######################################
            # 总的LOSS
            writer.add_scalar('test_stage_one_loss', loss_stage_1.item(), global_step=epoch * test_epoch + batch_index)
            writer.add_scalar('test_stage_two_pred_loss', _loss_stage_2.item(),
                              global_step=epoch * test_epoch + batch_index)
            writer.add_scalar('test_stage_two_fuse_loss', loss_stage_2.item(),
                              global_step=epoch * test_epoch + batch_index)

            z = torch.cat((one_stage_outputs[0], two_stage_outputs[0]), 0)
            writer.add_image('one&two_stage_image_batch:%d' % (batch_index),
                             make_grid(z,nrow=2), global_step=epoch)
            # writer.add_images('test_image_batch:%d_stage1' % (batch_index), one_stage_outputs[0], global_step=epoch)
            # writer.add_images('test_image_batch:%d_stage2' % (batch_index), two_stage_outputs[0], global_step=epoch)
            writer.add_scalar('test_fuse_loss_per_epoch', loss.item(), global_step=epoch * test_epoch + batch_index)
            ##########################################

        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        loss_stage1.update(loss_stage_1.item())
        loss_stage2.update(loss_stage_2.item())

        map8_loss_value.update(loss_8t.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score_stage2 = my_f1_score(two_stage_outputs[0], labels_dou_edge)
        precisionscore_stage2 = my_precision_score(two_stage_outputs[0], labels_dou_edge)
        accscore_stage2 = my_acc_score(two_stage_outputs[0], labels_dou_edge)
        recallscore_stage2 = my_recall_score(two_stage_outputs[0], labels_dou_edge)

        f1score_stage1 = my_f1_score(one_stage_outputs[0], labels_band)
        precisionscore_stage1 = my_precision_score(one_stage_outputs[0], labels_band)
        accscore_stage1 = my_acc_score(one_stage_outputs[0], labels_band)
        recallscore_stage1 = my_recall_score(one_stage_outputs[0], labels_band)

        writer.add_scalar('test_f1_score_stage1', f1score_stage1, global_step=epoch * test_epoch + batch_index)
        writer.add_scalar('test_precision_score_stage1', precisionscore_stage1,
                          global_step=epoch * test_epoch + batch_index)
        writer.add_scalar('test_acc_score_stage1', accscore_stage1, global_step=epoch * test_epoch + batch_index)
        writer.add_scalar('test_recall_score_stage1', recallscore_stage1, global_step=epoch * test_epoch + batch_index)

        writer.add_scalar('test_f1_score_stage2', f1score_stage2, global_step=epoch * test_epoch + batch_index)
        writer.add_scalar('test_precision_score_stage2', precisionscore_stage2,
                          global_step=epoch * test_epoch + batch_index)
        writer.add_scalar('test_acc_score_stage2', accscore_stage2, global_step=epoch * test_epoch + batch_index)
        writer.add_scalar('test_recall_score_stage2', recallscore_stage2, global_step=epoch * test_epoch + batch_index)
        ################################

        f1_value_stage1.update(f1score_stage1)
        precision_value_stage1.update(precisionscore_stage1)
        acc_value_stage1.update(accscore_stage1)
        recall_value_stage1.update(recallscore_stage1)

        f1_value_stage2.update(f1score_stage2)
        precision_value_stage2.update(precisionscore_stage2)
        acc_value_stage2.update(accscore_stage2)
        recall_value_stage2.update(recallscore_stage2)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, test_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   '两阶段总Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   '第一阶段Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=loss_stage1) + \
                   '第二阶段Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=loss_stage2) + \
                   '第一阶段:f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_stage1) + \
                   '第一阶段:precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(
                       precision=precision_value_stage1) + \
                   '第一阶段:acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_stage1) + \
                   '第一阶段:recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_stage1) + \
                   '第二阶段:f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value_stage2) + \
                   '第二阶段:precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(
                       precision=precision_value_stage2) + \
                   '第二阶段:acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value_stage2) + \
                   '第二阶段:recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value_stage2)

            print(info)

        if batch_index >= test_epoch:
            break

    return {'loss_avg': losses.avg,
            'f1_avg_stage1': f1_value_stage1.avg,
            'precision_avg_stage1': precision_value_stage1.avg,
            'accuracy_avg_stage1': acc_value_stage1.avg,
            'recall_avg_stage1': recall_value_stage1.avg,

            'f1_avg_stage2': f1_value_stage2.avg,
            'precision_avg_stage2': precision_value_stage2.avg,
            'accuracy_avg_stage2': acc_value_stage2.avg,
            'recall_avg_stage2': recall_value_stage2.avg
            }


if __name__ == '__main__':
    main()
