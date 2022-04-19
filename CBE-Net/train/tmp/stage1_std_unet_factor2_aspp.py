"""
@author: haoran
tine : 2021-02-22
description：
unet factor=2
没有使用aspp，3月10日在54上运行

"""
import os,sys
sys.path.append('../..')
sys.path.append('../../utils')

import torch
import torch.optim as optim
import torch.utils.data.dataloader
import argparse
import time
from functions import my_f1_score, my_acc_score, my_precision_score, weighted_cross_entropy_loss, wce_huber_loss, \
    wce_huber_loss_8, my_recall_score, cross_entropy_loss, wce_dice_huber_loss
from datasets.dataloader import TamperDataset
from model.unet_two_stage_model_0306_3 import UNetStage1 as Net
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import Averagvalue, weights_init, save_mid_result, send_msn
from os.path import join, isdir, isfile, splitext, split, abspath, dirname

"""
Created by HaoRan
time: 2021/01/29
description:
1. stage one training
"""

""""""""""""""""""""""""""""""
"          参数               "
""""""""""""""""""""""""""""""
name = '0317_stage1_后缀为0306_3的模型aspp第二阶段_全部数据'
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=6, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--model_save_dir', type=str, help='model_save_dir',
                    default='../save_model/'+name)
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
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
#####################resume##########################
parser.add_argument('--resume', default='/data-tmp/Mymodel/save_model/0310_stage1_后缀为0306_3的模型aspp第一阶段_全部数据/0310_stage1_后缀为0306_3的模型aspp第一阶段_checkpoint76-stage1-0.154597-f10.770403-precision0.886410-acc0.970743-recall0.685445.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--mid_result_index', type=list, help='mid_result_index', default=[0])
parser.add_argument('--per_epoch_freq', type=int, help='per_epoch_freq', default=50)

parser.add_argument('--fuse_loss_weight', type=int, help='fuse_loss_weight', default=12)
# ================ dataset

parser.add_argument('--band_mode', help='weather using band of normal gt', type=bool, default=True)
parser.add_argument('--save_mid_result', help='weather save mid result', type=bool, default=False)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
email_list = []

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
writer = SummaryWriter('../runs/' + name)
output_name_file_name = name+'_checkpoint%d-stage1-%f-f1%f-precision%f-acc%f-recall%f.pth'
email_header = 'Python'
""""""""""""""""""""""""""""""
"    ↑↑↑↑需要修改的参数↑↑↑↑     "
""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""
"          程序入口            "
""""""""""""""""""""""""""""""


def main():
    args.cuda = True
    # 1 choose the data you want to use
    using_data = {'my_sp': True,
                  'my_cm': True,
                  'template_casia_casia': True,
                  'template_coco_casia': False,
                  'cod10k': True,
                  'casia': False,
                  'coverage': False,
                  'columb': False,
                  'negative': True,
                  'negative_casia': False,
                  'texture_sp': True,
                  'texture_cm': True,
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
    trainData = TamperDataset(stage_type='stage1', using_data=using_data, train_val_test_mode='train',device='jly')
    valData = TamperDataset(stage_type='stage1', using_data=using_data, train_val_test_mode='val',device='jly')
    testData = TamperDataset(stage_type='stage1', using_data=using_data_test, train_val_test_mode='test',device='jly')

    # 3 specific dataloader
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                                  pin_memory=True)
    valDataLoader = torch.utils.data.DataLoader(valData, batch_size=args.batch_size, num_workers=4)

    testDataLoader = torch.utils.data.DataLoader(testData, batch_size=1, num_workers=1)
    # model
    model = Net()
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
            # optimizer.load_state_dict(checkpoint['optimizer'])
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
        val_avg = val(model=model, dataParser=valDataLoader, epoch=epoch)
        test(model=model, dataParser=testDataLoader, epoch=epoch)

        """"""""""""""""""""""""""""""
        "          写入图            "
        """"""""""""""""""""""""""""""
        try:

            writer.add_scalar('lr_per_epoch', scheduler.get_lr(), global_step=epoch)
            writer.add_scalars('tr-val-test_avg_loss_per_epoch', {'train': train_avg['loss_avg'],
                                                                  'val': val_avg['loss_avg']},
                               global_step=epoch)
            writer.add_scalars('tr-val-test_avg_f1_per_epoch', {'train': train_avg['f1_avg'],
                                                                'val': val_avg['f1_avg']}, global_step=epoch)

            writer.add_scalars('tr-val-test_avg_precision_per_epoch', {'train': train_avg['precision_avg'],
                                                                       'val': val_avg['precision_avg']},
                               global_step=epoch)
            writer.add_scalars('tr-val-test_avg_acc_per_epoch', {'train': train_avg['accuracy_avg'],
                                                                 'val': val_avg['accuracy_avg']},
                               global_step=epoch)
            writer.add_scalars('tr-val-test_avg_recall_per_epoch', {'train': train_avg['recall_avg'],
                                                                    'val': val_avg['recall_avg']},
                               global_step=epoch)


        except Exception as e:
            print(e)

        """"""""""""""""""""""""""""""
        "          写入图            "
        """"""""""""""""""""""""""""""

        # 保存模型
        """
                    info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, dataParser.steps_per_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
                   'precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value) + \
                   'acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) +\
                   'recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)

        """
        output_name = output_name_file_name % \
                      (epoch, val_avg['loss_avg'],
                       val_avg['f1_avg'],
                       val_avg['precision_avg'],
                       val_avg['accuracy_avg'],
                       val_avg['recall_avg'])


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
    f1_value = Averagvalue()
    acc_value = Averagvalue()
    recall_value = Averagvalue()
    precision_value = Averagvalue()
    map8_loss_value = Averagvalue()

    # switch to train mode
    model.train()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        # check_4dim_img_pair(input_data['tamper_image'],input_data['gt_band'])
        # 准备输入数据
        images = input_data['tamper_image'].cuda()
        labels = input_data['gt_band'].cuda()
        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
            loss_8t = torch.zeros(()).cuda()
        else:
            loss = torch.zeros(1)
            loss_8t = torch.zeros(())

        with torch.set_grad_enabled(True):
            images.requires_grad = True
            optimizer.zero_grad()
            # 网络输出
            outputs = model(images)[0]
            # 这里放保存中间结果的代码
            """"""""""""""""""""""""""""""
            "         Loss 函数           "
            """"""""""""""""""""""""""""""

            loss = wce_dice_huber_loss(outputs, labels)
            writer.add_scalar('loss_per_batch', loss.item(),
                              global_step=epoch * train_epoch + batch_index)

            loss.backward()
            optimizer.step()

        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        map8_loss_value.update(loss_8t.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score = my_f1_score(outputs, labels)
        precisionscore = my_precision_score(outputs, labels)
        accscore = my_acc_score(outputs, labels)
        recallscore = my_recall_score(outputs, labels)

        writer.add_scalar('f1_score', f1score, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('precision_score', precisionscore, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('acc_score', accscore, global_step=epoch * train_epoch + batch_index)
        writer.add_scalar('recall_score', recallscore, global_step=epoch * train_epoch + batch_index)
        ################################

        f1_value.update(f1score)
        precision_value.update(precisionscore)
        acc_value.update(accscore)
        recall_value.update(recallscore)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, train_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
                   'precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value) + \
                   'acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) + \
                   'recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)

            print(info)

        if batch_index >= train_epoch:
            break

    return {'loss_avg': losses.avg,
            'f1_avg': f1_value.avg,
            'precision_avg': precision_value.avg,
            'accuracy_avg': acc_value.avg,
            'recall_avg': recall_value.avg}


@torch.no_grad()
def val(model, dataParser, epoch):
    # 读取数据的迭代器
    val_epoch = len(dataParser)

    # 变量保存
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    f1_value = Averagvalue()
    acc_value = Averagvalue()
    recall_value = Averagvalue()
    precision_value = Averagvalue()
    map8_loss_value = Averagvalue()

    # switch to test mode
    model.eval()
    end = time.time()

    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)

        images = input_data['tamper_image']
        labels = input_data['gt_band']

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # 对读取的numpy类型数据进行调整

        if torch.cuda.is_available():
            loss = torch.zeros(1).cuda()
            loss_8t = torch.zeros(()).cuda()
        else:
            loss = torch.zeros(1)
            loss_8t = torch.zeros(())
        # 网络输出
        outputs = model(images)[0]
        # 这里放保存中间结果的代码
        if args.save_mid_result:
            if batch_index in args.mid_result_index:
                save_mid_result(outputs, labels, epoch, batch_index, args.mid_result_root, save_8map=True,
                                train_phase=True)
            else:
                pass
        else:
            pass
        """"""""""""""""""""""""""""""
        "         Loss 函数           "
        """"""""""""""""""""""""""""""
        loss = wce_dice_huber_loss(outputs, labels)
        writer.add_scalar('val_fuse_loss_per_epoch', loss.item(),
                          global_step=epoch * val_epoch + batch_index)
        # 将各种数据记录到专门的对象中
        losses.update(loss.item())
        map8_loss_value.update(loss_8t.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # 评价指标
        f1score = my_f1_score(outputs, labels)
        precisionscore = my_precision_score(outputs, labels)
        accscore = my_acc_score(outputs, labels)
        recallscore = my_recall_score(outputs, labels)

        writer.add_scalar('val_f1_score', f1score, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_precision_score', precisionscore, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_acc_score', accscore, global_step=epoch * val_epoch + batch_index)
        writer.add_scalar('val_recall_score', recallscore, global_step=epoch * val_epoch + batch_index)
        ################################

        f1_value.update(f1score)
        precision_value.update(precisionscore)
        acc_value.update(accscore)
        recall_value.update(recallscore)

        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, val_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'vla_Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
                   'val_f1_score {f1.val:f} (avg:{f1.avg:f}) '.format(f1=f1_value) + \
                   'val_precision_score: {precision.val:f} (avg:{precision.avg:f}) '.format(precision=precision_value) + \
                   'val_acc_score {acc.val:f} (avg:{acc.avg:f})'.format(acc=acc_value) + \
                   'val_recall_score {recall.val:f} (avg:{recall.avg:f})'.format(recall=recall_value)

            print(info)

        if batch_index >= val_epoch:
            break

    return {'loss_avg': losses.avg,
            'f1_avg': f1_value.avg,
            'precision_avg': precision_value.avg,
            'accuracy_avg': acc_value.avg,
            'recall_avg': recall_value.avg}


@torch.no_grad()
def test(model, dataParser, epoch):
    # switch to test mode
    model.eval()
    for batch_index, input_data in enumerate(dataParser):
        images = input_data['tamper_image']
        if torch.cuda.is_available():
            images = images.cuda()
        # 网络输出
        outputs = model(images)[0]
        writer.add_images('test_image_batch:%d' % (batch_index), outputs, global_step=epoch)
        ################################




if __name__ == '__main__':
    main()
