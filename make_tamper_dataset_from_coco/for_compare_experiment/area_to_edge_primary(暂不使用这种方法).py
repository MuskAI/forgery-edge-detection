"""
@author: haoran
time: 2021/3/4
background:
我在进行对比实验的时候需要将别人算法区域转化成我的边缘
"""
import numpy as np
from PIL import Image
import cv2 as cv
import os
import traceback
import matplotlib.pyplot as plt

class AreaToEdge:
    def __init__(self, save_dir='', is_255=True, check=False):
        """
        这个类只是负责将一张图片转化为边缘
        """
        self.save_dir = save_dir
        if save_dir == '':
            need_save = False
        else:
            need_save = True

        self.is_255 = is_255
        self.need_save = need_save
        self.check = check

    def convert(self, area_img_path, gt_img_path):
        """
        这是入口函数，整个转化的任务从这开始
        :param area_img_path: 输出为区域的结果所在绝对路径
        :param gt_img_path: 该张图所对应GT所在的绝对路径
        :return: 两个选择，一种是不return直接保存在指定目录，一种是return为numpy
        """
        # TODO 0 :将路径中的\\都换为/
        area_img_path = area_img_path.replace('\\', '/')
        area_img_name = area_img_path.split('/')[-1]
        # TODO 1: 将输入转化为pillow的文件格式
        area_img, gt_img = self.deal_with_input(area_img_path=area_img_path, gt_img_path=gt_img_path)

        # TODO 2: 开始按照规则转化
        edge_img = self.start_convert_to_edge(area_img, gt_img, is_255=self.is_255)

        # TODO 3: 是否需要检查
        if self.check:
            self.check_convert_result(area_img, gt_img, edge_img)


        # TODO 4: 选择是否保存在指定文件下
        if self.need_save:
            self.save_convert_result(edge_img, area_img_name, need_save=self.need_save)
            return 'dont need save'
        else:
            return edge_img




    def deal_with_input(self, area_img_path, gt_img_path):
        """

        :param area_img_path: 预测结果所在路径
        :param gt_img_path:  对应gt所在路径
        :return: 返回numpy
        """
        try:
            area_img = Image.open(area_img_path)
            gt_img = Image.open(gt_img_path)
        except Exception as e:
            traceback.print_exc(e)
            print('In deal_with_input, raed image error, please check it !!')

        # TODO 输入通道什么的 ，如果输入通道有问题则改正
        if len(area_img.split()) != 1 and len(area_img.split()) == 3:
            area_img = np.array(area_img.split()[0])
        else:
            area_img = np.array(area_img)
        if len(gt_img.split()) != 1 and len(gt_img.split()) == 3:
            gt_img = np.array(gt_img.split()[0])
            gt_img = np.where((gt_img == 255) | (gt_img == 100), 1, 0)
        else:
            gt_img = np.array(gt_img)
            gt_img = np.where((gt_img == 255) | (gt_img == 100), 1, 0)


        return area_img, gt_img

    def start_convert_to_edge(self, area_img, gt_img, is_255=False):
        if is_255 == False:
            area_img = area_img * 255
        else:
            pass
        edge = area_img * gt_img
        edge = edge.astype('uint8')
        edge = Image.fromarray(edge)

        return edge

    def save_convert_result(self, edge_img, area_img_name):
        edge_img = Image.fromarray(edge_img)
        edge_img.save(os.path.join(self.save_dir, area_img_name.replace('.jpg', '.png')))

    def check_convert_result(self, area_img, gt_img, edge_img):
        """
        检查结果错了没
        :param area_img: numpy
        :param gt_img: numpy
        :param edge_img: numpy
        :return:
        """
        area_img = np.array(area_img)
        gt_img = np.array(gt_img)
        edge_img = np.array(edge_img)
        plt.subplot(221)
        plt.imshow(area_img)
        plt.subplot(222)
        plt.imshow(gt_img)
        plt.subplot(223)
        plt.imshow(edge_img)
        plt.show()



if __name__ == '__main__':
    ATE = AreaToEdge(check=False)
    area_img_path = r'C:\Users\musk\Desktop\mantranet_output\mask\1t.bmp'
    gt_path = r'C:\Users\musk\Desktop\mantranet_output\gt\1t.bmp'
    ATE.convert(area_img_path=area_img_path, gt_img_path=gt_path)
