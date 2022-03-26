"""
@author:haoran
time : 2021/03/05
background:
复现别人的代码时的指标计算工具
description:
1. 输入一张图片和gt 然后计算他们的四种指标
2. 按照规范保存测试结果

usage:
1. 使用这个代码之前，你需要先使用复现好的模型将指定数据得到测试结果，然后通过将区域转化为边缘的算法进行转化
2. 做完第一步之后才可以使用这里的代码进行处理

"""
import numpy as np
from PIL import Image
import cv2 as cv
import traceback
import matplotlib.pyplot as plt
# 指标计算方法
from .functions import my_f1_score,my_acc_score,my_recall_score,my_precision_score


class Evaluate:
    def __init__(self, edge_dir='', save_dir=''):
        """
        在使用之前需要先确定好下面和几个参数
        :param edge_dir: 复现-->预测结果-->转化为边缘结果-->边缘结果所在路径
        :param save_dir: 需要保存的路径
        """
        self.edge_dir = edge_dir
        self.save_dir = save_dir

    def eval(self, edge_pred_path, edge_gt_path):
        """
        程序入口
        :param edge_pred_path: 预测结果边缘图所在绝对路径
        :param edge_gt_path: 预测结果gt所在绝对路径
        :return:四个指标的值
        """
        try:
            gt = Image.open(edge_gt_path)
            pred = Image.open(edge_pred_path)
        except Exception as e:
            traceback.print_exc(e)

        if len(gt.split()) == 3:
            gt = np.array(gt.split()[0])

        if len(pred.split())==3:
            pred = np.array(pred.split()[0])


        gt = np.array(gt)
        pred = np.array(pred)/255
        # TODO 判断你输入的图是不是标准的gt(我的gt图)图

        # TODO 将标准GT 的边缘拿出来
        gt = np.where((gt == 255) | (gt == 100), 1, 0)

        # TODO 展示出来
        plt.subplot(121)
        plt.imshow(pred)
        plt.subplot(122)
        plt.imshow(gt)
        plt.show()
        # TODO 开始计算指标，如果需要设置阈值进入functions中修改
        f1 = my_f1_score(pred,gt)
        acc = my_acc_score(pred,gt)
        precision = my_precision_score(pred,gt)
        recall = my_recall_score(pred,gt)

        result = {'f1':f1,
                'acc':acc,
                'precision':precision,
                'recall':recall}
        print(result)
        return result

    def input_issues(self):
        pass

if __name__ == '__main__':
    edge_dir = r'C:\Users\musk\Desktop\mantranet_output\edge\1t.bmp'
    gt_dir = r'C:\Users\musk\Desktop\mantranet_output\gt\1t.bmp'

    evaluate = Evaluate().eval(edge_pred_path=edge_dir,edge_gt_path=gt_dir)