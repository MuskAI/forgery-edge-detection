"""
created by HDG
time: 2022-1-7
description:
唯一的数据加载器，后续所有代码都使用这个数据加载器

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

            self.train_src_list, self.val_src_list, self.train_gt_list, self.val_gt_list = \
                train_test_split(train_val_src_list, train_val_gt_list, test_size=val_percent,
                                 train_size=1 - val_percent, random_state=1234)

            _train_src_list, _val_src_list = \
                train_test_split(train_val_src_list, test_size=val_percent,
                                 train_size=1 - val_percent, random_state=1234)

            self.transform = transform
            # if there is a check function would be better
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
            _band = np.array(_band)[:, :, 0]
        else:
            _band = np.array(_band)
        return _band

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
            if 'negative' in tamper_path:
                isforgery = False
        except Exception as e:
            traceback.print_exc(e)
        # check the src dim
        if mode=='test':
            if len(img.split()) != 3:
                print(tamper_path, 'error')
                print(gt_path)
        else:
            if len(img.split()) != 3 or img.size !=(320,320) or gt.size !=(320,320):
                # print(tamper_path, 'error')
                # print(gt_path)
                pass
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
            # 将std gt换成篡改区域
            gt_band = self.__gen_band(gt)
            gt = np.array(gt, dtype='uint8')
            # 转化为无类别的GT 100 255 为边缘
            # dou_em = np.array(gt)
            # dou_em = np.where(dou_em == 50, 0, dou_em)
            # dou_em = np.where(dou_em == 100, 255, dou_em)

            gt = np.where(gt==50, 255,gt)
            gt = np.where(gt==100,0,gt)
            gt = Image.fromarray(gt)
            gt_band = Image.fromarray(gt_band)


        except Exception as e:
            traceback.print_exc(e)

        if mode == 'train' or mode == 'val':
            # if transform src
            if self.transform:
                img = self.transform(img)
            else:

                img = transforms.Compose([
                    # AddGlobalBlur(p=0.5),
                    transforms.Resize((320, 320)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)
            gt = transforms.ToTensor()(gt)
            gt_band = transforms.ToTensor()(gt_band)

        elif mode == 'test':
            # if transform src
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)
            gt = transforms.ToTensor()(gt)
            gt_band = transforms.ToTensor()(gt_band)
        else:
            traceback.print_exc('the train_val_test mode is error')

        sample = {'tamper_image': img,
                  'gt' : gt,
                  'gt_band' : gt_band,
                  'gt_cls':torch.Tensor([1]) if isforgery else torch.Tensor([0]),
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
        if device=='413':
            self.data_root = '/home/liu/haoran/3月最新数据'
        elif device=='jly':
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
            gt_path = name.replace('t','forged')
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
                    path = os.path.join(self.data_root,'coco_sp/train_src')
                    src_path_list.append(path)
                    self.sp_gt_path =os.path.join(self.data_root, 'coco_sp/train_gt')
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
                    path = os.path.join(self.data_root,'casia_au_and_casia_template_after_divide/train_src')

                    src_path_list.append(path)
                    self.template_casia_casia_gt_path = os.path.join(self.data_root, 'casia_au_and_casia_template_after_divide/train_gt')

                else:

                    path = '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26'
                    src_path_list.append(path)
                    self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'
        except Exception as e:
            print('template_casia_casia', 'error')

        try:
            if using_data['template_coco_casia']:
                if train_mode:
                    path = os.path.join(self.data_root,'coco_casia_template_after_divide/train_src')
                    src_path_list.append(path)
                    self.template_coco_casia_gt_path = os.path.join(self.data_root, 'coco_casia_template_after_divide/train_gt')
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

        # # texture
        # try:
        #     if using_data['texture_sp']:
        #         if train_mode:
        #             path = os.path.join(self.data_root, '0108_texture_and_casia_template_divide/train_src')
        #             src_path_list.append(path)
        #             self.texture_sp_gt_path = os.path.join(self.data_root, '0108_texture_and_casia_template_divide/train_gt')
        #         else:
        #             path = '/home/liu/chenhaoran/Tamper_Data/0222/0108_texture_and_casia_template_divide/test_src'
        #             src_path_list.append(path)
        #             self.texture_sp_gt_path = '/home/liu/chenhaoran/Tamper_Data/0222/0108_texture_and_casia_template_divide/test_gt'
        # except Exception as e:
        #     print(e)
        #
        # try:
        #     if using_data['texture_cm']:
        #         if train_mode:
        #             path = os.path.join(self.data_root, 'periodic_texture/divide/train_src')
        #             src_path_list.append(path)
        #             self.texture_cm_gt_path = os.path.join(self.data_root,
        #                                                    'periodic_texture/divide/train_gt')
        #         else:
        #             path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/src'
        #             src_path_list.append(path)
        #             self.texture_cm_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'
        # except Exception as e:
        #     print(e)
        #
        # try:
        #     if using_data['casia']:
        #         if train_mode:
        #             pass
        #         else:
        #             path = '/home/liu/chenhaoran/Tamper_Data/0222/casia/src'
        #             src_path_list.append(path)
        #             self.casia_gt_path = '/home/liu/chenhaoran/Tamper_Data/0222/casia/gt'
        #
        # except Exception as e:
        #     print(e)
        # ##################################################################################

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

    dataloader = torch.utils.data.DataLoader(mytestdataset, batch_size=1, num_workers=1)
    start = time.time()
    try:
        for idx, item in enumerate(dataloader):
            print(item['gt_band'])
            _ = np.array(item['gt_band'])
            _ = _.squeeze(0)
            print(_.shape)
            print(item['gt_band'])
            plt.imshow(_)
            plt.show()
            if item['tamper_image'].shape[2:] !=item['gt_band'].shape[2:]:
                pass
                # print(item['gt_band'])
                # plt.imshow(item['gt_band'])

    except Exception as e:
        print(e)
    end = time.time()
    print('time :', end - start)