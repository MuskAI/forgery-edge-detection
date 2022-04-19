#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：forgery-edge-detection 
@File    ：reproduce.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/10 08:31

关于论文复现的一些基类

'''
import numpy as np
import os,sys,shutil
from abc import ABCMeta, abstractmethod


class BaseInfer:
    """
    论文复现时候的基类
    """
    def __init__(self,data_root,using_data=None):
        assert os.path.exists(data_root)
        self.data_root = data_root
        _ = {'columbia': True,
             'coverage': True,
             'casia': False,
             'ps-battle': False,
             'in-the-wild': False,
             }
        self.using_data = using_data if using_data is not None else _

        # 各种数据集的路径
        self.data_dict = {}
        for _ in self.using_data:
            if self.using_data[_]:
                pass
            else:
                continue

            self.data_dict[_] = {
                'pred': os.path.join(self.pred_root, _, 'pred'),
                'gt': os.path.join(self.data_root, _, 'gt'),
                'fpars': []
            }

        if self.data_dict == {}:
            return

    @abstractmethod
    def get_single_result(self) :
        """
        获取一张图片的结果，需要到子类中实现
        @return: numpy array
        """
        pass

    def get_batch_results(self) -> bool:
        """
        批量获取指定数据集的结果
        @return:
        """
        pass


    def read(self):
        """
        读取数据的接口
        @return:
        """
        assert self.data_dict is not {}, '请指定需要评测的数据集'

        # check file exist or not
        for item in self.data_dict:
            assert os.path.exists(self.data_dict[item]['pred']), '文件路径 {} 不存在'.format(self.data_dict[item]['pred'])
            assert os.path.exists(self.data_dict[item]['gt']), '文件路径 {} 不存在'.format(self.data_dict[item]['pred'])

            # TODO 检查pred 与 gt 匹配情况

        # 获取图片列表
        for item in self.data_dict:
            _ = []
            pred_list = os.listdir(self.data_dict[item]['pred'])
            gt_list = os.listdir(self.data_dict[item]['gt'])

            # matching
            for idx, name in enumerate(pred_list):
                pred_path = os.path.join(self.data_dict[item]['pred'], name)
                gt_path = self.__match(data_type=item, query_name=name, name_list=gt_list)
                if gt_path is None:
                    continue
                gt_path = os.path.join(self.data_dict[item]['gt'], gt_path)  # 获取gt的图片path
                _.append({
                    'image': '',
                    'pred': pred_path,
                    'gt': gt_path
                })
            self.data_dict[item]['image_pair'] = _

    @abstractmethod
    def write(self):
        """
        保存推理的结果
        @return:
        """
        pass


    def __repr__(self):
        return '进行对比实验时候的一些基类'