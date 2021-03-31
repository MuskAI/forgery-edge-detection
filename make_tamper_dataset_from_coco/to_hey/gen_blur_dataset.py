"""
@author:haoran
time:0316
"""
import cv2 as cv
from PIL import Image
from PIL import ImageFilter
import traceback
import warnings
import numpy as np
import matplotlib.pylab as plt
import skimage.morphology as dilation
import hashlib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import os
import random


class MyBlur:
    """
    1. 输入进来的是一个Image类型的图
    """

    def __init__(self):
        self.radius_set = [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0]
        self.random_radius = [0, 3]

    def __filter(self, img, blur_mode=None):
        """

        :param blur_mode:
        :return: the list , 3 numpy array
        """
        if blur_mode == None:
            warnings.warn('You are not choose a blur_mode, So i will using default')
        elif blur_mode == 'gaussian':
            img_blur_list = []
            for idx, radius in enumerate(self.radius_set):
                # big to small
                img_blur_list.append(img.filter(ImageFilter.GaussianBlur(radius=radius)))
                img_blur_list[idx] = np.array(img_blur_list[idx], dtype='uint8')

        else:
            print('done nothing')

        if len(img_blur_list) != 0:
            return img_blur_list
        else:
            traceback.print_exc('an unknown error occur')
            exit(1)

    def edge_blur(self, img, blur_area):

        blur_list = self.__filter(img, blur_mode='gaussian')
        if len(blur_area.shape) == 3:
            blur_area = np.array(blur_area, dtype='uint8')
        elif len(blur_area.shape) == 2:
            blur_area = np.array(blur_area, dtype='uint8')
            _gt = np.zeros((blur_area.shape[0], blur_area.shape[1], 3))
            _gt[:, :, 0] = blur_area
            _gt[:, :, 1] = blur_area
            _gt[:, :, 2] = blur_area
            blur_area = _gt
        else:
            print('the input gt error!')

        weight_blur = np.where(blur_area == 255, 1, 0) * blur_list[
            random.randint(self.random_radius[0], self.random_radius[1])]
        img = img * np.where(blur_area == 255, 0, 1) + weight_blur
        return img


class EdgeBlur:
    def __init__(self):
        pass

    def gen_blur_data(self, src_path, gt_path):
        src = Image.open(src_path)
        gt = Image.open(gt_path)

        # TODO 1 :get_blur_area
        blur_area = self.get_blur_area(gt)

        # TODO 2:start_edge_blur
        random_edge_blur_img = self.__start_edge_blur(src, blur_area)

        # TODO 3:更新GT
        update_gt = self.__update_gt(gt, blur_area)
        # self.visualize([random_edge_blur_img,blur_area,update_gt],mode='result')
        return {'edge_blur_img': random_edge_blur_img,
                'blur_area': blur_area,
                'update_gt':update_gt}

    def get_blur_area(self, gt):

        # 找到mask 的矩形区域
        mask = gt.copy()
        mask = np.array(mask)
        mask = np.where(mask != 0, 1, 0)
        a = np.where(mask != 0)
        bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
        not_cut = False
        if bbox[3] - bbox[1] > 100:
            cut_point1 = random.randint(bbox[1] + 20, bbox[3] - 20)
        elif bbox[2] - bbox[0] > 100:
            cut_point1 = random.randint(bbox[0] + 20, bbox[2] - 20)
        else:
            not_cut = True
        dilate_windows = random.randint(5, 10)
        blur_band = self.__gen_band(gt, dilate_window=dilate_windows)
        blur_area = np.zeros_like(blur_band)
        if not_cut == False:
            blur_area[:, :cut_point1] = blur_band[:, :cut_point1]
        else:
            blur_area = blur_band

        # self.visualize([blur_band, blur_area],mode='blur_area',title='模糊窗口大小为:'+str(dilate_windows))
        # There we get the blur area : blur_area
        return blur_area

    def __start_edge_blur(self, src, blur_area):
        random_edge_blur_img = MyBlur().edge_blur(img=src, blur_area=blur_area)
        # self.visualize([src,random_edge_blur_img,blur_area,abs(random_edge_blur_img-src)],mode='blur_img')
        return random_edge_blur_img

    def __update_gt(self, original_gt, blur_area):
        # 生成band_area的双边缘结果
        original_gt = np.array(original_gt, dtype='uint8')
        blur_area_dou_edge = self.__mask_to_double_edge(np.where(blur_area == 255, 1, 0))


        # 解决重叠问题
        # 有些模糊像素的双边缘与未模糊像素的双边缘重叠了,这个时候应该选择未模糊像素的双边缘
        # 先求交
        update_gt = blur_area_dou_edge + original_gt*np.where((blur_area_dou_edge==255)|(blur_area_dou_edge==100)|(blur_area_dou_edge==50),0,1)
        return update_gt
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

    def __mask_to_double_edge(self, orignal_mask):
        """
        :param orignal_mask: 输入的是 01 mask图
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

            return np.where(ground_truth == 305, 255, ground_truth)

        except Exception as e:
            print(e)

    def visualize(self, img_list, mode, title=''):
        if mode == 'blur_area':
            plt.subplot(121)

            plt.imshow(img_list[0])
            plt.subplot(122)
            plt.imshow(img_list[1])
            plt.title(title)
            plt.show()
        elif mode == 'blur_img':
            plt.subplot(221)
            plt.xlabel('原图', )
            plt.imshow(img_list[0])
            plt.subplot(222)
            plt.xlabel('模糊图')
            plt.imshow(img_list[1])
            plt.subplot(223)
            plt.xlabel('模糊区域')
            plt.imshow(img_list[2])
            plt.subplot(224)
            plt.xlabel('相减绝对值')
            plt.imshow(img_list[3])
            plt.show()
        elif mode == 'new_gt':
            plt.subplot(121)
            plt.xlabel('原来的GT')
            plt.imshow(img_list[0], cmap='gray')
            plt.subplot(122)
            plt.xlabel('模糊后的双边缘GT')
            plt.imshow(img_list[1], cmap='gray')
            plt.show()
        elif mode=='result':
            plt.subplot(131)
            plt.imshow(img_list[0],resample=False)
            plt.subplot(132)
            plt.imshow(img_list[1],resample=False)
            plt.subplot(133)
            plt.imshow(img_list[2],resample=False)
            plt.show()


if __name__ == '__main__':
    img_list = os.listdir(r'C:\Users\musk\Desktop\mantranet_output\or')
    gt_list = os.listdir(r'C:\Users\musk\Desktop\mantranet_output\gt')
    eb = EdgeBlur()
    for idx, item in enumerate(img_list):
        img_path = os.path.join(r'C:\Users\musk\Desktop\mantranet_output\or', item)
        gt_path = os.path.join(r'C:\Users\musk\Desktop\mantranet_output\gt', gt_list[idx])
        # 返回的result是个字典，需要保存下来更新后的gt,命名不变
        result = eb.gen_blur_data(src_path=img_path, gt_path=gt_path)
