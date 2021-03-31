"""
created by HaoRan
time: 1114
description:
the only data reader
input: dataset path
output: a iterator
"""
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
import torchvision
from PIL import Image
import time
import os, sys
import traceback
from sklearn.model_selection import train_test_split
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# from check_image_pair import check_4dim_img_pair
from PIL import ImageFilter
import random
from rich.progress import track


class TamperDataset(Dataset):
    def __init__(self, stage_type='stage1', transform=None, train_val_test_mode='train', device='413', using_data=None,
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
        self.stage_type = stage_type
        self.transform = transform
        # if the mode is train then split it to get val
        """train or test mode"""
        if train_val_test_mode == 'train' or train_val_test_mode == 'val':

            train_val_src_list, train_val_gt_list = \
                MixData(train_mode=True, using_data=using_data, device=device).gen_dataset()

            self.train_src_list, self.val_src_list, self.train_gt_list, self.val_gt_list = \
                train_test_split(train_val_src_list, train_val_gt_list, test_size=val_percent,
                                 train_size=1 - val_percent, random_state=1000)

            _train_src_list, _val_src_list = \
                train_test_split(train_val_src_list, test_size=val_percent,
                                 train_size=1 - val_percent, random_state=1000)

            self.transform = transform
            # if there is a check function would be better
        elif train_val_test_mode == 'test':
            self.test_src_list, self.test_gt_list = \
                MixData(train_mode=False, using_data=using_data, device=device).gen_dataset()
        else:
            raise EOFError

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

        if mode == 'train':
            tamper_path = self.train_src_list[index]
            gt_path = self.train_gt_list[index]
        elif mode == 'val':
            tamper_path = self.val_src_list[index]
            gt_path = self.val_gt_list[index]
        elif mode == 'test':
            tamper_path = self.test_src_list[index]
            gt_path = self.test_gt_list[index]
        else:
            traceback.print_exc('an error occur')

        # read img
        try:
            img = Image.open(tamper_path)
            gt = Image.open(gt_path)
        except Exception as e:
            traceback.print_exc(e)
        # check the src dim
        if mode=='test':
            if len(img.split()) != 3:
                print(tamper_path, 'error')
                print(gt_path)
        else:
            if len(img.split()) != 3 or img.size !=(320,320) or gt.size !=(320,320):
                print(tamper_path, 'error')
                print(gt_path)
        ##############################################

        # check the gt dim
        if len(gt.split()) == 3:
            gt = gt.split()[0]
        elif len(gt.split()) == 1:
            pass
        else:
            traceback.print_exc('gt dim error! please check it ')
        ##################################################
        try:
            # 全部将下面的1 换成255
            gt_band = self.__gen_band(gt)
            if self.stage_type == 'stage2':
                gt_dou_edge = self.__to_dou_edge(gt)
                # 下面的relation_map还是01
                relation_map = self.__gen_8_2_map(np.array(gt,dtype='uint8'))
        except Exception as e:
            print(e)

        if mode == 'train' or mode == 'val':
            # if transform src
            if self.transform:
                img = self.transform(img)
            else:

                img = transforms.Compose([
                    # AddGlobalBlur(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)

            # transform
            gt_band = transforms.ToTensor()(gt_band)
            if self.stage_type == 'stage2':
                gt_dou_edge = transforms.ToTensor()(gt_dou_edge)

        elif mode == 'test':
            # if transform src
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)

            # transform
            gt_band = transforms.ToTensor()(gt_band)

            if self.stage_type == 'stage2':
                gt_dou_edge = transforms.ToTensor()(gt_dou_edge)

        else:
            traceback.print_exc('the train_val_test mode is error')
        if self.stage_type == 'stage1':
            sample = {'tamper_image': img, 'gt_band': gt_band, 'path': {'src': tamper_path, 'gt': gt_path}}
        elif self.stage_type == 'stage2':
            sample = {'tamper_image': img, 'gt_band': gt_band, 'gt_dou_edge': gt_dou_edge,'relation_map':relation_map,
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
            _band = np.array(_band)[:, :, 0]
        else:
            _band = np.array(_band)
        return _band


    def __to_dou_edge(self,dou_em):
        # 转化为无类别的GT 100 255 为边缘
        dou_em = np.array(dou_em)
        dou_em = np.where(dou_em == 50, 0, dou_em)
        dou_em = np.where(dou_em == 100, 255, dou_em)
        return dou_em
    def __gen_8_2_map(self, mask, mask_area=50, mask_edge=255, not_mask_edge=100):
        """
        输入mask，先按照固定参数标好，篡改区域、篡改区域边缘，非篡改区域边缘，非篡改区域
        :param mask: 255 的图，channel数为1
        :return: 从左上角的点开始按照顺时针方向的8张二通道的图
        """
        # 在输入mask 之前对mask进行检查
        if type(mask) is np.ndarray:
            if mask.ndim == 2:
                pass
            else:
                # 如果输入的维度不是3二维的，则转化成2 dim
                print('Notice: when using the function gen_8_2_map, the mask ndim is not 2 but', mask.ndim)
                mask = mask[:,:,0]
        else:
            print('Notice: when using function gen_8_2_map, the input mask not numpy array')
            traceback.print_exc()
            sys.exit()

        # 开始进行8张图的生成
        relation_8_map = []
        edge_loc_ = [1, 1]
        # 找到内侧和外侧边缘
        mask_pad = np.pad(mask, (1, 1), mode='constant')
        mask_pad = np.where(mask_pad == 50, 0, mask_pad)
        edge_loc = np.where(mask_pad == mask_edge)
        edge_loc_1 = np.where(mask_pad == not_mask_edge)
        edge_loc_[0] = np.append(edge_loc[0], edge_loc_1[0])
        edge_loc_[1] = np.append(edge_loc[1], edge_loc_1[1])

        del edge_loc_1
        del edge_loc
        edge_loc = edge_loc_
        mask_shape = mask_pad.shape
        # 生成8张结果图
        for i in range(8):
            relation_8_map.append(np.ones((mask_shape[0], mask_shape[1], 2)))

        for j in range(len(edge_loc[0])):
            row = edge_loc[0][j]
            col = edge_loc[1][j]
            if mask_pad[row - 1, col - 1] != 0:
                relation_8_map[0][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row - 1, col - 1]:
                    relation_8_map[0][row, col, 1] = 1
                else:
                    relation_8_map[0][row, col, 1] = 0
            else:
                relation_8_map[0][row, col, 0] = 0
                relation_8_map[0][row, col, 1] = 0

            if mask_pad[row - 1, col] != 0:
                relation_8_map[1][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row - 1, col]:
                    relation_8_map[1][row, col, 1] = 1
                else:
                    relation_8_map[1][row, col, 1] = 0
            else:
                relation_8_map[1][row, col, 0] = 0
                relation_8_map[1][row, col, 1] = 0

            if mask_pad[row - 1, col + 1] != 0:
                relation_8_map[2][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row - 1, col + 1]:
                    relation_8_map[2][row, col, 1] = 1
                else:
                    relation_8_map[2][row, col, 1] = 0
            else:
                relation_8_map[2][row, col, 0] = 0
                relation_8_map[2][row, col, 1] = 0

            if mask_pad[row, col + 1] != 0:
                relation_8_map[3][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row, col + 1]:
                    relation_8_map[3][row, col, 1] = 1
                else:
                    relation_8_map[3][row, col, 1] = 0
            else:
                relation_8_map[3][row, col, 0] = 0
                relation_8_map[3][row, col, 1] = 0

            if mask_pad[row + 1, col + 1] != 0:
                relation_8_map[4][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row + 1, col + 1]:
                    relation_8_map[4][row, col, 1] = 1
                else:
                    relation_8_map[4][row, col, 1] = 0
            else:
                relation_8_map[4][row, col, 0] = 0
                relation_8_map[4][row, col, 1] = 0

            if mask_pad[row + 1, col] != 0:
                relation_8_map[5][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row + 1, col]:
                    relation_8_map[5][row, col, 1] = 1
                else:
                    relation_8_map[5][row, col, 1] = 0
            else:
                relation_8_map[5][row, col, 0] = 0
                relation_8_map[5][row, col, 1] = 0

            if mask_pad[row + 1, col - 1] != 0:
                relation_8_map[6][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row + 1, col - 1]:
                    relation_8_map[6][row, col, 1] = 1
                else:
                    relation_8_map[6][row, col, 1] = 0
            else:
                relation_8_map[6][row, col, 0] = 0
                relation_8_map[6][row, col, 1] = 0

            if mask_pad[row, col - 1] != 0:
                relation_8_map[7][row, col, 0] = 1
                if mask_pad[row, col] == mask_pad[row, col - 1]:
                    relation_8_map[7][row, col, 1] = 1
                else:
                    relation_8_map[7][row, col, 1] = 0
            else:
                relation_8_map[7][row, col, 0] = 0
                relation_8_map[7][row, col, 1] = 0

        for i in range(8):
            relation_8_map[i] = relation_8_map[i][1:-1, 1:-1, :]
            relation_8_map[i] = torch.from_numpy(relation_8_map[i].transpose((2,0,1)))
        return relation_8_map
class MixData:
    def __init__(self, train_mode=True, using_data=None, device='413'):
        """

        :param train_mode:
        :param using_data:
        :param device:
        """
        if device=='413':
            self.data_root = '/home/liu/chenhaoran/Tamper_Data/3月最新数据'
        elif device=='jly':
            self.data_root = '/data-tmp/3月最新数据'
        elif device=='libiao':
            self.data_root = '/media/libiao/1385DB3ECD51D467/chenhaoran/3月最新数据'
        elif device =='hy':
            self.data_root = '/hy-tmp/3月最新数据'
        elif device == 'wkl':
            self.data_root = 'D:\\chenhaoran\\data'
        elif device == 'ai500':
            self.data_root = '/home/dc2-user/chenhaoran/3月份最新数据'
        elif device == '413blur':
            self.data_root = '/home/liu/chenhaoran/Tamper_Data/edge_blur'
        # data_path_gather的逻辑是返回一个字典，该字典包含了需要使用的src 和 gt
        data_dict = MixData.__data_path_gather(self, train_mode=train_mode, using_data=using_data)
        # src

        # if True:
        #     self.src_path_list = ['/media/liu/File/debug_data/tamper_result']
        #     self.cm_gt_path = '/media/liu/File/debug_data/ground_truth_result'

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

        # if MixData.__check(self,train_list=, gt_list=):
        #     pass
        # else:
        #     print('check error, please redesign')
        #     traceback.print_exc()
        #     sys.exit()

        return train_list, gt_list

    def __check(self, train_list, gt_list):
        """
        检查train_list 和 gt_list 是否有问题
        :return:
        """

        pass

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

        sp_type = ['coco_sp']
        cm_type = ['coco_cm']
        negative_type = ['negative']
        CASIA_type = ['Tp']
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
            if sp_flag in path:
                type.append('sp')
                break
        for cm_flag in cm_type:
            if cm_flag in path:
                type.append('cm')
                break

        for negative_flag in negative_type:
            if negative_flag in path:
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
            gt_path = name.replace('Default', '').replace('Sp', name.split('.')[-1]).replace('.jpg', '.bmp').replace('.png', '.bmp').replace('poisson',
                                                                                                            'Gt')
            gt_path = os.path.join(self.sp_gt_path, gt_path)
            pass
        elif type[0] == 'cm':
            gt_path = name.replace('Default', name.split('.')[-1]+'_').replace('.jpg', '.bmp').replace('.png', '.bmp').replace('poisson',
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
            gt_path = name.split('.')[-1]+name.replace('COD10K','').replace('tamper', '').split('.')[0] + '.bmp'
            gt_path = os.path.join(self.COD10K_gt_path, gt_path)

        elif type[0] == 'COVERAGE':
            # gt_path = name.split('.')[0] + '_gt.bmp'
            gt_path = name.replace('t','forged')
            gt_path = os.path.join(self.coverage_gt_path, gt_path)
        elif type[0] == 'TEXTURE_CM':
            gt_path = name.split('.')[-1] + '_'+name.split('.')[0] + '.bmp'
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
                    path = os.path.join(self.data_root,'coco_sp/train_src_blur')
                    src_path_list.append(path)
                    self.sp_gt_path =os.path.join(self.data_root, 'coco_sp/train_gt_blur')
                else:
                    path = '/media/liu/File/Sp_320_dataset/tamper_result_320'
                    src_path_list.append(path)
                    self.sp_gt_path = '/media/liu/File/Sp_320_dataset/ground_truth_result_320'
        except Exception as e:
            print(e)

        try:
            if using_data['my_cm']:
                if train_mode:
                    path = os.path.join(self.data_root, 'coco_cm/train_src_blur')
                    # path = '/home/liu/chenhaoran/Tamper_Data/0222/coco_cm_after_divide/train_src_blur'
                    src_path_list.append(path)
                    self.cm_gt_path = os.path.join(self.data_root, 'coco_cm/train_gt_blur')
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
                    path = os.path.join(self.data_root,'casia_au_and_casia_template_after_divide/train_src_blur')

                    src_path_list.append(path)
                    self.template_casia_casia_gt_path = os.path.join(self.data_root, 'casia_au_and_casia_template_after_divide/train_gt_blur')

                else:

                    path = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
                    src_path_list.append(path)
                    self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'
        except Exception as e:
            print('template_casia_casia', 'error')

        try:
            if using_data['template_coco_casia']:
                if train_mode:
                    path = os.path.join(self.data_root,'coco_casia_template_after_divide/train_src_blur')
                    src_path_list.append(path)
                    self.template_coco_casia_gt_path = os.path.join(self.data_root, 'coco_casia_template_after_divide/train_gt_blur')
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

                    path = os.path.join(self.data_root, 'COD10K/train_src_blur')
                    src_path_list.append(path)
                    self.COD10K_gt_path = os.path.join(self.data_root, 'COD10K/train_gt_blur')

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
                    self.negative_gt_path = os.path.join(self.data_root, 'negative_gt')
                else:
                    path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/src'
                    src_path_list.append(path)
                    self.negative_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'
        except Exception as e:
            print(e)

        # texture
        try:
            if using_data['texture_sp']:
                if train_mode:
                    path = os.path.join(self.data_root, '0108_texture_and_casia_template_divide/train_src_blur')
                    src_path_list.append(path)
                    self.texture_sp_gt_path = os.path.join(self.data_root, '0108_texture_and_casia_template_divide/train_gt_blur')
                else:
                    path = '/home/liu/chenhaoran/Tamper_Data/0222/0108_texture_and_casia_template_divide/test_src'
                    src_path_list.append(path)
                    self.texture_sp_gt_path = '/home/liu/chenhaoran/Tamper_Data/0222/0108_texture_and_casia_template_divide/test_gt'
        except Exception as e:
            print(e)

        try:
            if using_data['texture_cm']:
                if train_mode:
                    path = os.path.join(self.data_root, 'periodic_texture/divide/train_src_blur')
                    src_path_list.append(path)
                    self.texture_cm_gt_path = os.path.join(self.data_root,
                                                           'periodic_texture/divide/train_gt_blur')
                else:
                    path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/src'
                    src_path_list.append(path)
                    self.texture_cm_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'
        except Exception as e:
            print(e)

        try:
            if using_data['casia']:
                if train_mode:
                    pass
                else:
                    path = '/home/liu/chenhaoran/Tamper_Data/0222/casia/src'
                    src_path_list.append(path)
                    self.casia_gt_path = '/home/liu/chenhaoran/Tamper_Data/0222/casia/gt'

        except Exception as e:
            print(e)
        ##################################################################################

        # public dataset
        try:
            if using_data['coverage']:
                if train_mode:
                    pass
                else:
                    path = os.path.join('/home/liu/chenhaoran/Tamper_Data/3月最新数据', 'public_dataset/coverage/src')
                    src_path_list.append(path)
                    self.coverage_gt_path = os.path.join('/home/liu/chenhaoran/Tamper_Data/3月最新数据', 'public_dataset/coverage/gt')

        except Exception as e:
            print(e)

        try:
            if using_data['columb']:
                path = '/media/liu/File/Sp_320_dataset/tamper_result_320'
                src_path_list.append(path)
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

    print('start')
    # mydataset = TamperDataset(using_data={'my_sp': True,
    #                                       'my_cm': True,
    #                                       'template_casia_casia': True,
    #                                       'template_coco_casia': True,
    #                                       'cod10k': True,
    #                                       'casia': False,
    #                                       'copy_move': False,
    #                                       'texture_sp':True,
    #                                       'texture_cm': True,
    #                                       'columb': False,
    #                                       'negative': True,
    #                                       'negative_casia': False,
    #                                       }, train_val_test_mode='train',stage_type='stage2',device='jly')
    mytestdataset = TamperDataset(using_data={'my_sp': False,
                                              'my_cm': False,
                                              'template_casia_casia': False,
                                              'template_coco_casia': False,
                                              'cod10k': False,
                                              'casia': False,
                                              'coverage': True,
                                              'columb': False,
                                              'negative_coco': False,
                                              'negative_casia': False,
                                              'texture_sp': False,
                                              'texture_cm': False,
                                              }, train_val_test_mode='test',stage_type='stage1',device='413blur')
    dataloader = torch.utils.data.DataLoader(mytestdataset, batch_size=1, num_workers=8)
    start = time.time()
    try:
        for idx, item in enumerate(track(dataloader)):
            # print(idx, type(item))
            # print(item['relation_map'][0].shape)
            # if item['tamper_image'].shape[1] !=3 or item['tamper_image'].shape[2]!=320:
            #     rich.print(item['path'])

            # check_4dim_img_pair(item['tamper_image'], item['gt_band'])
            # if idx == 3000:
            #     break


            if item['tamper_image'].shape[2:] !=item['gt_band'].shape[2:]:
                # print('++++++')
                # print(item['tamper_image'].shape)
                # print(item['gt_band'].shape)
                print(item['path'])

    except Exception as e:
        # print(item['path'])
        print(e)
    end = time.time()
    print('time :', end - start)
