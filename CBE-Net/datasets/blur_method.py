"""
created by haoran
time : 20201208
"""
import cv2 as cv
from PIL import Image
from PIL import ImageFilter
import traceback
import warnings
import numpy as np
import matplotlib.pylab as plt

class MyBlur:
    """
    1. 输入进来的是一个Image类型的图
    """
    def __init__(self):
        self.radius_set = [1.5, 1, 0.5, 0]
    def filter(self,img,blur_mode=None):
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
                # plt.figure()
                # plt.imshow(img_blur_list[idx])
                # plt.show()

            # img = img.filter(ImageFilter.MedianFilter())

        else:
            print('done nothing')

        if len(img_blur_list) !=0:
            return img_blur_list
        else:
            traceback.print_exc('an unknown error occur')
            exit(1)

    def edge_blur(self, img, gt_path):
        blur_list = MyBlur.filter(self,img,blur_mode='gaussian')
        gt = Image.open(gt_path)

        if len(gt.split()) == 3:
            gt = np.array(gt, dtype='uint8')
        elif len(gt.split()) == 1:
            gt = np.array(gt, dtype='uint8')
            _gt = np.zeros([320, 320, 3])
            _gt[:, :, 0] = gt
            _gt[:, :, 1] = gt
            _gt[:, :, 2] = gt
            gt = _gt
        else:
            print('the input gt error!')

        weight_blur = np.where((gt == 255) | (gt == 100), 1, 0) * blur_list[1]
        img = img * np.where((gt == 255) | (gt == 100), 0, 1) + weight_blur
        plt.imshow(img)
        plt.show()

class MyWeightBlur(MyBlur):
    def __init__(self,img,gt_path):
        super(MyWeightBlur, self).__init__(img)
        self.gt_path = gt_path
        self.size = 320

    def weight_gaussian_blur(self):
        blur_list = MyWeightBlur.filter(self, 'gaussian')
        gt = Image.open(self.gt_path)
        if len(gt.split()) == 3:
            pass
        elif len(gt.split()) == 1:
            _gt = np.zeros([self.size,self.size,3])
            _gt[:, :, 0] = gt
            _gt[:, :, 1] = gt
            _gt[:, :, 2] = gt
            gt = _gt
        else:
            print('the input gt error!')

        weight_blur = np.zeros([self.size, self.size, 3])
        for idx, item in enumerate(blur_list):
            weight_blur = np.where((gt == 255) | (gt == 100), 1, 0) * blur_list[idx]


if __name__ == '__main__':
    img_path = r'C:\Users\musk\Desktop\tamper_test\Default_78_398895_clock.png'
    gt_path = r'C:\Users\musk\Desktop\tamper_test\Gt_78_398895_clock.bmp'
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    MyBlur().edge_blur(img, gt_path)