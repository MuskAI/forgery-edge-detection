#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：forgery-edge-detection
@File    ：dataloader_final.py
@IDE     ：PyCharm
@Author  ：haoran
@Date    ：2022/4/5 9:44 PM

最终模型的数据处理，有如下功能：
1. 合并数据集
2. 差错、容错
3. 双边缘、条带、区域、类别标签均支持
4. 对公开数据集的处理特别优化
'''
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
import torchvision
from PIL import Image
import time
import os, sys
import traceback
from sklearn.model_selection import train_test_split
from scipy import ndimage
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# from check_image_pair import check_4dim_img_pair
from PIL import ImageFilter
import random


class TamperDataset(Dataset):
    def __init__(self, transform=None, train_val_test_mode='train', device='413', using_data=None,
                 val_percent=0.1):
        """
        The only data loader for train val test dataset
        using_data = {'my_sp':True,'my_cm':True,'casia':True,'copy_move':True,'columb':True}
        :param transform: only for src transform
        :param train_val_test_mode: the type is string
        :param device: using this to debug, 413, laptop for choose
        :param using_data: a dict, e.g.
        """

        # train val test mode
        self.train_val_test_mode = train_val_test_mode
        self.transform = transform
        # if the mode is train then split it to get val
        """train or test mode"""
        if train_val_test_mode == 'train' or train_val_test_mode == 'val':
            train_val_src_list, train_val_gt_list = \
                MixData(train_mode=True, using_data=using_data, device=device).gen_dataset()

            # 划分验证集
            self.train_src_list, self.val_src_list, self.train_gt_list, self.val_gt_list = \
                train_test_split(train_val_src_list, train_val_gt_list, test_size=val_percent,
                                 train_size=1 - val_percent, random_state=1234)

            self.transform = transform
            # TODO if there is a check function would be better
        elif train_val_test_mode == 'test':
            self.test_src_list, self.test_gt_list = \
                MixData(train_mode=False, using_data=using_data, device=device).gen_dataset()
        else:
            raise EOFError

    def __gen_band(self, gt, dilate_window=5):
        """
        :param gt: PIL type
        :param dilate_window:
        :return:
        """

        _gt = gt.copy()

        # input required
        if len(_gt.split()) == 3:
            _gt = _gt.split()[0]
        else:
            pass

        _gt = np.array(_gt, dtype='uint8')

        if max(_gt.reshape(-1)) == 255:
            _gt = np.where((_gt == 255) | (_gt == 100), 1, 0)
            _gt = np.array(_gt, dtype='uint8')
        else:
            pass

        _gt = cv.merge([_gt])
        kernel = np.ones((dilate_window, dilate_window), np.uint8)
        _band = cv.dilate(_gt, kernel)
        _band = np.array(_band, dtype='uint8')
        _band = np.where(_band == 1, 255, 0)
        _band = Image.fromarray(np.array(_band, dtype='uint8'))
        if len(_band.split()) == 3:
            _band = np.array(_band,dtype='uint8')[:, :, 0]
        else:
            _band = np.array(_band,dtype='uint8')
        _band = Image.fromarray(_band)
        return _band
    def __gen_cls(self,gt):
        """
        通过gt得到类别标签
        @param gt:
        @return: cls_label ,True = 有篡改
        """
        _gt = gt.copy()
        cls_label = False

        if sum(np.array(_gt).reshape(-1)) > 0:
            cls_label = True

        return cls_label
    def __gen_area(self, gt):
        """
        得到篡改区域
        @param gt:标准Gt
        @return: 0-255的mask
        """
        _gt = gt.copy()
        _gt = np.array(_gt, dtype='uint8')
        _gt = np.where(_gt == 50, 255, _gt)
        _gt = np.where(_gt == 100, 0, _gt)
        _gt = Image.fromarray(_gt)
        return _gt

    def __gen_dou_edge(self,gt):
        """
        得到双边缘
        @param gt:
        @return:
        """
        _gt = gt.copy()
        _gt = np.array(_gt, dtype='uint8')
        _gt = np.where(_gt == 50, 0, _gt)
        _gt = np.where(_gt == 100, 255, _gt)
        _gt = Image.fromarray(_gt)
        return _gt
        pass

    def check_pipeline(self, data):
        """
        检查到错误，然后纠正错误。如果出现错误则返回纠正后的结果，纠正不了的就返回None交给上一级处理
        @param data:
        @return:
        """
        for idx, item in enumerate(data):
            # 检查输入的RGB图
            if item == 'img':
                _ = data['img']
                # 检查有没有出错,错误一般有这几种情况
                # 1. 文件被损坏，打不开 (不在这考虑
                # 2. 不是三通道的图 (直接统统转为RGB
                # 3. ps-battle中161×81尺寸的图 (在这里处理

                if (_.size[0], _.size[1]) == (161, 81):
                    data[item] = None



            elif item == 'gt':
                _ = data['gt']
                # check the gt dim
                if len(_.split()) == 3:
                    _ = _.split()[0]
                else:
                    pass
                data[item] = _

            else:
                raise '暂时还没有实现这种{}类型的检查'.format(data)

        return data


    def __getitem__(self, index):
        """
        train val test 区别对待
        :param index:
        :return:
        """
        # train mode
        # val mode
        # test mode
        mode = self.train_val_test_mode

        isforgery = True

        # read img,考虑容错处理

        try:
            checked_data = None

            for i in range(3):
                if mode == 'train':
                    tamper_path = self.train_src_list[index]
                    gt_path = self.train_gt_list[index]
                    length = len(tamper_path)
                elif mode == 'val':
                    tamper_path = self.val_src_list[index]
                    gt_path = self.val_gt_list[index]
                    length = len(tamper_path)
                elif mode == 'test':
                    tamper_path = self.test_src_list[index]
                    gt_path = self.test_gt_list[index]
                    length = len(tamper_path)
                else:
                    raise '模式选择错误，你的选择是{}'.format(mode)

                img = Image.open(tamper_path).convert('RGB')
                gt = Image.open(gt_path)
                # if 'negative' in tamper_path:
                #     isforgery = False

                # 需要检查的数据
                need_checked = {
                    'img': img,
                    'gt': gt,
                }

                need_checked = self.check_pipeline(data=need_checked)

                # 开始进行容错处理
                if need_checked['img'] is not None:
                    pass
                else:
                    # 随机3次，如果不能解决问题则停止程序
                    print('数据错误,{}进行容错处理'.format(tamper_path))
                    index = np.random.randint(0, length)
                    continue

                if need_checked['gt'] is not None:
                    pass
                else:
                    print('数据错误,{}进行容错处理'.format(gt_path))
                    index = np.random.randint(0, length)
                    continue

                checked_data = need_checked
                break

            if checked_data['img'] is None or checked_data['gt'] is None:
                raise 'image load error'
            else:
                img = checked_data['img']
                gt = checked_data['gt']
                # dou edge
                dou_edge = self.__gen_dou_edge(gt)

                # band
                gt_band = self.__gen_band(gt)

                # area
                gt_area = self.__gen_area(gt)

                # cls
                gt_cls = self.__gen_cls(gt)



        except Exception as e:
            print('数据加载错误，开始容错处理')
            raise '暂时还没些容错处理'


        if mode == 'train' or mode == 'val':
            # if transform src
            if self.transform:
                img = self.transform(img)
            else:
                p = np.random.choice([0, 1])
                t1_img = transforms.Compose([
                    # transforms.RandomCrop(320),
                    transforms.Resize(320),
                    transforms.RandomHorizontalFlip(p=p),
                    transforms.RandomVerticalFlip(p=p),
                    # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),

                ])


                t2_img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])

                t3_img = transforms.Compose([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),

                ])

                t1_gt = transforms.Compose([
                    transforms.ToTensor()
                ])

                # 获取增广后的img,band,dou_edge
                if p > 0.8:
                    img = t2_img(t3_img(t1_img(img)))
                else:
                    img = t2_img(t1_img(img))

                gt_band = t1_gt(t1_img(gt_band))


        elif mode == 'test':
            # if transform src
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)
            dou_edge = transforms.ToTensor()(dou_edge)
            gt_area = transforms.ToTensor()(gt_area)
            gt_band = transforms.ToTensor()(gt_band)
        else:
            traceback.print_exc('the train_val_test mode is error')

        sample = {'tamper_image': img,
                  # 'dou_edge': dou_edge,
                  'gt_band': gt_band,
                  # 'gt_area' : gt_area,
                  'gt_cls': torch.Tensor([1]) if isforgery else torch.Tensor([0]),
                  'path': {'src': tamper_path, 'gt': gt_path}}

        return sample

    def __len__(self):
        mode = self.train_val_test_mode
        if mode == 'train':
            length = len(self.train_src_list)
        elif mode == 'val':
            length = len(self.val_src_list)
        elif mode == 'test':
            length = len(self.test_src_list)
        else:
            traceback.print_exc('an error occur')
        return length


class MixData:
    def __init__(self, train_mode=True, using_data=None, device='413'):
        """
        :param train_mode:
        :param using_data:
        :param device:
        """
        if device == '413':
            self.data_root = '/home/liu/haoran/3月最新数据'
        elif device == 'jly':
            self.data_root = '/data-tmp/3月最新数据'

        # data_path_gather的逻辑是返回一个字典，该字典包含了需要使用的src 和 gt
        data_dict = MixData.__data_path_gather(self, train_mode=train_mode, using_data=using_data)
        # src

    def gen_dataset(self):
        """
        通过输入的src & gt的路径生成train_list 列表
        并通过check方法，检查是否有误
        :return:
        """
        dataset_type_num = len(self.src_path_list)
        train_list = []
        gt_list = []
        unmatched_list = []
        # 首先开始遍历不同类型的数据集路径
        for index1, item1 in enumerate(self.src_path_list):
            for index2, item2 in enumerate(os.listdir(item1)):
                t_img_path = os.path.join(item1, item2)
                # print(t_img_path)
                t_gt_path = MixData.__switch_case(self, t_img_path)
                if t_gt_path != '':
                    train_list.append(t_img_path)
                    gt_list.append(t_gt_path)
                else:
                    print(t_gt_path, t_gt_path, 'unmatched')
                    unmatched_list.append([t_img_path, t_gt_path])
                    print('The process: %d/%d : %d/%d' % (
                        index1 + 1, len(self.src_path_list), index2 + 1, len((os.listdir(item1)))))
        print('The number of unmatched data is :', len(unmatched_list))
        print('The unmatched list is : ', unmatched_list)

        return train_list, gt_list

    def __switch_case(self, path):
        """
        针对不同类型的数据集做处理
        :return: 返回一个路径，这个路径是path 所对应的gt路径，并且需要检查该路径是否存在
        """
        # 0 判断路径的合法性
        if os.path.exists(path):
            pass
        else:
            print('The path :', path, 'does not exist')
            return ''
        # 1 分析属于何种类型
        # there are
        # 1.  sp generate data
        # 2. cm generate data
        # 3. negative data
        # 4. CASIA data

        sp_type = ['Sp']
        cm_type = ['Default', 'poisson']
        negative_type = ['negative']
        CASIA_type = ['casia']
        COLUMBIA_type = ['columbia']
        COVERAGE_type = ['coverage']
        debug_type = ['debug']
        template_coco_casia = ['coco_casia_template_after_divide']
        template_casia_casia = ['casia_au_and_casia_template_after_divide']
        COD10K_type = ['COD10K']
        texture_sp_type = ['texture_and_casia_template_divide']
        texture_cm_type = ['periodic_texture']
        type = []
        name = path.split('/')[-1]
        # name = path.split('\\')[-1]
        for sp_flag in sp_type:
            if sp_flag in name[:2] and 'texture' not in path:
                type.append('sp')
                break

        for cm_flag in cm_type:
            if cm_flag in name[:7] and 'texture' not in path:
                type.append('cm')
                break

        for negative_flag in negative_type:
            if negative_flag in name:
                type.append('negative')
                break

        for CASIA_flag in CASIA_type:
            if CASIA_flag in path and 'template' not in path:
                type.append('CASIA')
                break

        for template_flag in template_casia_casia:
            if template_flag in path:
                type.append('TEMPLATE_CASIA_CASIA')
                break
        for template_flag in template_coco_casia:
            if template_flag in path:
                type.append('TEMPLATE_COCO_CASIA')
                break

        for COD10K_flag in COD10K_type:
            if COD10K_flag in path:
                type.append('COD10K')
                break

        for COVERAGE_flag in COVERAGE_type:
            if COVERAGE_flag in path:
                type.append('COVERAGE')
                break
        for COLUMBIA_flag in COLUMBIA_type:
            if COLUMBIA_flag in path:
                type.append('COLUMBIA')
                break
        for TEXUTURE_flag in texture_cm_type:
            if TEXUTURE_flag in path:
                type.append('TEXTURE_CM')
                break
        for TEXUTURE_flag in texture_sp_type:
            if TEXUTURE_flag in path:
                type.append('TEXTURE_SP')
                break
        # 判断正确性

        if len(type) != 1:
            print('The type len is ', len(type))
            return ''

        if type[0] == 'sp':
            gt_path = name.replace('Default', 'Gt').replace('.jpg', '.bmp').replace('.png', '.bmp').replace('poisson',
                                                                                                            'Gt')
            gt_path = os.path.join(self.sp_gt_path, gt_path)
            pass
        elif type[0] == 'cm':
            gt_path = name.replace('Default', 'Gt').replace('.jpg', '.bmp').replace('.png', '.bmp').replace('poisson',
                                                                                                            'Gt')
            gt_path = os.path.join(self.cm_gt_path, gt_path)
            pass
        elif type[0] == 'negative':
            gt_path = 'negative_gt.bmp'
            gt_path = os.path.join(self.negative_gt_path, gt_path)
            pass
        elif type[0] == 'CASIA':
            gt_path = name.split('.')[0] + '_gt' + '.png'
            gt_path = os.path.join(self.casia_gt_path, gt_path)
            pass
        elif type[0] == 'TEMPLATE_CASIA_CASIA':
            gt_path = name.split('.')[0] + '.bmp'
            gt_path = os.path.join(self.template_casia_casia_gt_path, gt_path)

        elif type[0] == 'TEMPLATE_COCO_CASIA':
            gt_path = name.split('.')[0] + '.bmp'
            gt_path = os.path.join(self.template_coco_casia_gt_path, gt_path)

        elif type[0] == 'COD10K':
            gt_path = name.split('.')[0] + '.bmp'
            gt_path = gt_path.replace('tamper', 'Gt')
            gt_path = os.path.join(self.COD10K_gt_path, gt_path)

        elif type[0] == 'COVERAGE':
            # gt_path = name.split('.')[0] + '_gt.bmp'
            gt_path = name.replace('t', 'forged')
            gt_path = os.path.join(self.coverage_gt_path, gt_path)




        elif type[0] == 'TEXTURE_CM':
            gt_path = name.split('.')[0] + '.bmp'
            gt_path = os.path.join(self.texture_cm_gt_path, gt_path)
        elif type[0] == 'TEXTURE_SP':
            gt_path = name.split('.')[0] + '.bmp'
            gt_path = os.path.join(self.texture_sp_gt_path, gt_path)

        else:
            traceback.print_exc()
            print('Error')
            sys.exit()
        # 判断gt是否存在
        if os.path.exists(gt_path):
            pass
        else:
            return ''

        return gt_path

    def __data_path_gather(self, train_mode=True, using_data=None):
        """
        using_data = {'my_sp':True,'my_cm':True,'casia':True,'copy_move':True,'columb':True,'negative':True}
        :param device:
        :param using_data:
        :return:
        """

        src_path_list = []
        if using_data:
            pass
        else:
            traceback.print_exc('using_data input None error')
            print(
                "using_data = {'my_sp':True,'my_cm':True,'casia':True,'copy_move':True,'columb':True,'negative':True}")
            sys.exit(1)

        # sp cm
        try:
            if using_data['my_sp']:
                if train_mode:
                    path = os.path.join(self.data_root, 'coco_sp/train_src')
                    src_path_list.append(path)
                    self.sp_gt_path = os.path.join(self.data_root, 'coco_sp/train_gt')
                else:
                    traceback.print_exc('trying to use sp data to test error')
        except Exception as e:
            print(e)
        #
        try:
            if using_data['my_cm']:
                if train_mode:
                    path = os.path.join(self.data_root, 'coco_cm/train_src')
                    # path = '/home/liu/chenhaoran/Tamper_Data/0222/coco_cm_after_divide/train_src'
                    src_path_list.append(path)
                    self.cm_gt_path = os.path.join(self.data_root, 'coco_cm/train_gt')
                else:
                    path = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
                    src_path_list.append(path)
                    self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'
        except Exception as e:
            print(e)
        ###########################################
        # template
        try:
            if using_data['template_casia_casia']:
                if train_mode:
                    path = os.path.join(self.data_root, 'casia_au_and_casia_template_after_divide/train_src')

                    src_path_list.append(path)
                    self.template_casia_casia_gt_path = os.path.join(self.data_root,
                                                                     'casia_au_and_casia_template_after_divide/train_gt')

                else:

                    path = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
                    src_path_list.append(path)
                    self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'
        except Exception as e:
            print('template_casia_casia', 'error')

        try:
            if using_data['template_coco_casia']:
                if train_mode:
                    path = os.path.join(self.data_root, 'coco_casia_template_after_divide/train_src')
                    src_path_list.append(path)
                    self.template_coco_casia_gt_path = os.path.join(self.data_root,
                                                                    'coco_casia_template_after_divide/train_gt')
                else:
                    path = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
                    src_path_list.append(path)
                    self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'

        except Exception as e:
            print(e)
        ###########################################

        # cod10k
        try:
            if using_data['cod10k']:
                if train_mode:

                    path = os.path.join(self.data_root, 'COD10K/train_src')
                    src_path_list.append(path)
                    self.COD10K_gt_path = os.path.join(self.data_root, 'COD10K/train_gt')

                else:
                    path = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
                    src_path_list.append(path)
                    self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'
        except Exception as e:
            print(e)

        # casia
        try:
            if using_data['casia']:
                if train_mode:
                    path = os.path.join(self.data_root, 'public_dataset/casia/src')
                    src_path_list.append(path)
                    self.casia_gt_path = os.path.join(self.data_root, 'public_dataset/casia/gt')

                else:
                    path = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
                    src_path_list.append(path)
                    self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'
        except Exception as e:
            print(e)
        #############################################

        # negative
        try:
            if using_data['negative']:
                if train_mode:
                    path = os.path.join(self.data_root, 'negative')
                    src_path_list.append(path)
                    self.negative_gt_path = '/home/liu/haoran/HDG_ImgTamperDetection/utils'
                else:
                    path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/src'
                    src_path_list.append(path)
                    self.negative_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'
        except Exception as e:
            print(e)

        # public dataset
        try:

            if using_data['coverage']:
                if train_mode:
                    pass
                else:
                    path = os.path.join(self.data_root, 'public_dataset/coverage/src')
                    src_path_list.append(path)
                    self.coverage_gt_path = os.path.join(self.data_root, 'public_dataset/coverage/gt')

        except Exception as e:
            print(e)

        try:
            if using_data['columb']:
                path = os.path.join(self.data_root, 'public_dataset/columbia/src')
                src_path_list.append(path)
                self.columbia_gt_path = os.path.join(self.data_root, 'public_dataset/columbia/gt')
        except Exception as e:
            print(e)
        ##############################

        self.src_path_list = src_path_list


class AddGlobalBlur(object):
    """
    增加全局模糊t
    """

    def __init__(self, p=1.0):
        """
        :param p: p的概率会加模糊
        """
        kernel_size = random.randint(0, 10) / 10
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.random() < self.p:  # 概率判断
            img_ = np.array(img).copy()
            img_ = Image.fromarray(img_)
            img_ = img_.filter(ImageFilter.GaussianBlur(radius=self.kernel_size))
            img_ = np.array(img_)
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


if __name__ == '__main__':
    mytestdataset = TamperDataset(using_data={'my_sp': True,
                                              'my_cm': True,
                                              'template_casia_casia': False,
                                              'template_coco_casia': False,
                                              'cod10k': False,
                                              'casia': False,
                                              'coverage': False,
                                              'columb': False,
                                              'negative_coco': False,
                                              'negative_casia': False,
                                              'texture_sp': False,
                                              'texture_cm': False,
                                              }, train_val_test_mode='train')

    dataloader = torch.utils.data.DataLoader(mytestdataset, batch_size=1, num_workers=0)
    start = time.time()
    try:
        for idx, item in enumerate(dataloader):
            print(item)
            plt.show()
            if item['tamper_image'].shape[2:] != item['gt_band'].shape[2:]:
                pass
                # print(item['gt_band'])
                # plt.imshow(item['gt_band'])

    except Exception as e:
        print(e)
    end = time.time()
    print('time :', end - start)
