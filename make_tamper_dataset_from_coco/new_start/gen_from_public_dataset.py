"""
Created by Haoran
Time : 10/12
description: 从公开数据生成数据集
1. CASIA 1.0 2.0
2. Coverage

在你需要使用公开数据集训练的时候，用这部分代码
"""
import os, sys
import cv2 as cv
from PIL import Image
import traceback
import random
import numpy as np
import shutil
import skimage.morphology as dilation
import matplotlib.pyplot as plt

def cv_imread(file_path):
    cv_img = cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img
class PublicDataset():
    def __init__(self):
        pass

    def casia1(self, in_path_src=None, in_path_gt=None, save_path=None):
        """
        casia1 tamper 数据文件夹目录：1.CM 2.Sp
        CM文件组成： src: name ;gt: name_gt
        :param in_path_src:
        :param in_path_gt:
        :param save_path:
        :return:
        """
        # 0 有效性判断
        PublicDataset.__casia_path_check(self, in_path_src, in_path_gt, save_path)

        # 1 check src and gt
        src_list = os.listdir(in_path_src)
        gt_list = os.listdir(in_path_gt)
        unmatched_list = []
        matched_list = []
        for src_name in src_list:
            gt_name = src_name.split('.')[0] + '_gt' + '.png'
            print('The gt_name is :', gt_name)
            if gt_name not in gt_list:
                print('src:[%s] with gt:[%s] not match' % (src_name, gt_name))
                unmatched_list.append(gt_name)

        if len(unmatched_list) == 0:
            print('All the data match successfully!')
        else:
            print('There are some data not match, the unmatched number is : %d, the percent is %d/%d'
                  % (len(unmatched_list), len(unmatched_list), len(src_list)))
            print(unmatched_list)
            print(unmatched_list[:][1])
            for item in src_list:
                if item.split('.')[0] + '_gt' + '.png' not in unmatched_list:
                    matched_list.append(item)
                    shutil.copyfile(src=os.path.join(in_path_src, item), dst=os.path.join(save_path + '\\src', item))

        # 2 generate gt from gt_list and save it
        for index, gt_name in enumerate(matched_list):
            gt = Image.open(os.path.join(in_path_gt, gt_name.split('.')[0] + '_gt' + '.png'))
            gt = np.array(gt)
            if gt.shape[-1] == 3 or gt.shape[-1] == 4:
                gt = gt[:, :, 0]
                gt = np.where(gt > 0, 1, 0)
            else:
                pass
            gt = np.where(gt > 0, 1, 0)
            out_gt = PublicDataset.__mask_to_double_edge(self, gt)
            # check out_gt
            # print('The out_gt shape is :', out_gt.shape)
            # plt.figure('out_gt')
            # plt.imshow(out_gt)
            # plt.show()
            ################
            out_gt = Image.fromarray(out_gt)
            out_gt.save(os.path.join(save_path + '\\gt', gt_name.split('.')[0] + '_gt' + '.png'))
            print('\r', 'gt generate process: {:d}/{:d}'.format(index, len(gt_list)), end='')

        return True

    def size_statistics(self, statistics_path=None, target_size=(320, 320)):
        if statistics_path == None:
            print('Please input a useful path')
            sys.exit()
        else:
            required_img_list = []
            print('Start statistics')
            for item in os.listdir(statistics_path):
                if '.png' or '.jpg' or '.tif' in item:
                    pass
                else:
                    print('The format not required', item)
                    continue
                img = Image.open(os.path.join(statistics_path, item))
                if img.size[0] >= target_size[0] and img.size[1] >= target_size[1]:
                    required_img_list.append(item)
            print('End statistics')
            print('The statistics result: (required:all) = (%d:%d)' % (
            len(required_img_list), len(os.listdir(statistics_path))))

        # 开始搬运
        dst = r'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA2.0_SELECTED\gt'
        for item in required_img_list:
            shutil.copyfile(os.path.join(statistics_path, item), os.path.join(dst, item))

    def __mask_to_double_edge(self, orignal_mask):
        """
        :param orignal_mask:
        :return: 255 100 50 mask 图
        """
        # print('We are in mask_to_outeedge function:')
        try:
            mask = orignal_mask
            # print('the shape of mask is :', mask.shape)
            selem = np.ones((3, 3))
            dst_8 = dilation.binary_dilation(mask, selem=selem)
            dst_8 = np.where(dst_8 == True, 1, 0)
            difference_8 = dst_8 - orignal_mask

            difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
            difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
            double_edge_candidate = difference_8_dilation + mask
            double_edge = np.where(double_edge_candidate == 2, 1, 0)
            ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
                mask == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
            ground_truth = np.where(ground_truth == 305, 255, ground_truth)
            ground_truth = np.array(ground_truth, dtype='uint8')
            return ground_truth

        except Exception as e:
            print(e)

    def __casia_path_check(self, in_path_src, in_path_gt, save_path):
        if in_path_src == None:
            print('You should giving a useful casia1 src path')
            traceback.print_exc()
            sys.exit()
        else:
            pass
        if in_path_gt == None:
            print('You should giving a useful casia1 gt path')
            traceback.print_exc()
            sys.exit()
        else:
            pass

        if save_path == None:
            print('You should giving a useful save path')
            traceback.print_exc()
            sys.exit()

        if os.path.exists(save_path):
            if len(os.listdir(save_path)) != 0:
                print('保存路径文件夹不为空，请重新输入')
                sys.exit()
            else:
                if os.path.exists(os.path.join(save_path, 'src')) and os.path.exists(os.path.join(save_path, 'gt')):
                    print('保存路径存在，满足要求')
        else:
            print('保存路径不存在，开始创建')
            os.mkdir(save_path)
            os.mkdir(os.path.join(save_path, 'src'))
            os.mkdir(os.path.join(save_path, 'gt'))

    def casia_crop(self, in_path_src, in_path_gt, save_path):
        """
        :param in_path:该目录存在两个文件夹src gt
        :param save_path: 该目录需要自动创建两个文件夹src_after_320crop gt_after_320crop
        :return:
        """
        # 0 check path
        PublicDataset.__casia_path_check(self, in_path_src=in_path_src, in_path_gt=in_path_gt, save_path=save_path)
        in_src_list = os.listdir(in_path_src)
        in_gt_list = os.listdir(in_path_gt)

        for index, src_name in enumerate(in_src_list):
            gt_name = src_name.split('.')[0] + '_gt' + '.png'
            if gt_name in in_gt_list:
                pass
            else:
                print('match error,please check the file')
            try:
                img = Image.open(os.path.join(in_path_src, src_name))
                gt = Image.open(os.path.join(in_path_gt, gt_name))
            except:
                print('error!')
                continue
                traceback.print_exc()
            # 1 check img and gt
            img = np.array(img)
            gt = np.array(gt)
            # print(img.shape, gt.shape)
            crop320_src, crop320_gt = PublicDataset.__crop(self, img=img, gt=gt)
            crop320_src = Image.fromarray(crop320_src)
            crop320_gt = Image.fromarray(crop320_gt)
            # plt.figure('check crop320_src')
            # plt.imshow(crop320_src)
            # plt.show()
            #
            # plt.figure('check crop320_gt')
            # plt.imshow(crop320_gt)
            # plt.show()
            t_save_src = os.path.join(save_path + '\\\\src', src_name.split('.')[0] + '.png')

            crop320_gt.save(os.path.join(save_path + '\\\\gt', gt_name.split('.')[0] + '.png'))
            crop320_src.save(os.path.join(save_path + '\\\\src', src_name.split('.')[0] + '.png'))
            print('\r', 'The process of crop is :%d/%d' % (index, len(in_src_list)), end='')

    def __crop(self, img, gt, target_shape=(320, 320)):
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]
        random_height_range = height - target_shape[0]
        random_width_range = width - target_shape[1]

        if random_width_range < 0 or random_height_range < 0:
            return img
        if random_height_range == 0:
            random_height = 0
        else:
            random_height = np.random.randint(0, random_height_range)
        if random_width_range == 0:
            random_width = 0
        else:
            random_width = np.random.randint(0, random_width_range)

        return img[random_height:random_height + target_shape[0], random_width:random_width + target_shape[1]], \
               gt[random_height:random_height + target_shape[0], random_width:random_width + target_shape[1]]

class DataCheck:
    def __init__(self,dir_path):
        self.dir_path = dir_path
    def channel(self):
        for idx, item in enumerate(os.listdir(self.dir_path)):
            img_path = os.path.join(self.dir_path, item)
            # I = Image.open(img_path)
            I = cv_imread(img_path)


            # print(idx,' ',item)
            # print(len(I.split()))

            if I.shape[-1]!= 3:
                print('the size of ',item,'is ',I.shape)

if __name__ == '__main__':
    #
    in_path_src = r'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA\CASIA 2.0\CASIA 2.0\Tp'
    in_path_gt = r'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA\casia2groundtruth-master\CASIA 2 Groundtruth'
    save_path = r'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA2.0_GT'
    check_path = 'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA2.0_DATA_FOR_TRAIN\src'
    # # The input save_path is a root path ,which contain src dir and gt dir
    # PublicDataset().casia1(in_path_src, in_path_gt, save_path)
    DataCheck(check_path).channel()
