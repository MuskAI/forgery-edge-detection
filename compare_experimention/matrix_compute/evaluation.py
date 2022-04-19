"""
created by haoran
time :2022-3-26
version: beta v1
实现各种指标的计算，默认采用的方式是
mantranet-std_eval
    -columbia
        -pred
        -gt
    -coverage
        -pred
        -gt

"""
import os, sys

import cv2
import wandb
from functions import my_precision_score, my_f1_score, my_acc_score, my_recall_score
import matplotlib.pyplot as plt
import numpy as np
import re
from pprint import pprint
from tqdm import tqdm


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Eval:
    def __init__(self, data_root, pred_root, using_data=None):
        self.data_root = data_root
        self.pred_root = pred_root

        assert os.path.exists(self.data_root)
        assert os.path.exists(self.pred_root)
        _ = {'columbia': True,
             'coverage': True,
             'casia': False,
             'ps-battle': False,
             'in-the-wild': False,
             }
        self.using_data = using_data if using_data is not None else _  # 没有指定评测数据的时候使用默认的评测数据

        # 创建保存变量的类

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

    def read(self):
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

    @staticmethod
    def __match(data_type, query_name, name_list, match_type='pred2gt'):
        """
        通过pred的图片名称使用正则表达去匹配gt
        """
        assert data_type in ['columbia', 'coverage', 'casia', 'in-the-wild', 'ps-battle', 'realistic'], \
            'not support data type {} so far'.format(data_type)
        assert match_type in ['pred2gt']

        match_re = ''
        matched_list = []
        if data_type == 'columbia':
            # 希望两边名称都是一样的
            key_words = query_name.replace('area_', '').replace('band_', '').replace('_edgemask', '').replace('output_','').split('.')[0]
            for _ in name_list:
                if key_words in _:
                    matched_list.append(_)
        elif data_type == 'coverage':
            match_re = r'(\d)+'
            key_words = re.search(match_re, query_name).group()  # 匹配到的关键字
            for _ in name_list:
                if key_words == re.search(match_re, _).group():
                    matched_list.append(_)

        elif data_type == 'casia':
            key_words = query_name.replace('output_', '').split('.')[0]
            # key_words = re.match(match_re, query_name)  # 匹配到的关键字
            for _ in name_list:
                if key_words in _:
                    matched_list.append(_)

        elif data_type == 'in-the-wild':
            key_words = re.match(match_re, query_name)  # 匹配到的关键字
            for _ in name_list:
                if key_words in _:
                    matched_list.append(_)

        elif data_type == 'ps-battle':
            key_words = re.match(match_re, query_name)  # 匹配到的关键字
            for _ in name_list:
                if key_words in _:
                    matched_list.append(_)

        elif data_type == 'realistic':
            key_words = re.match(match_re, query_name)  # 匹配到的关键字
            for _ in name_list:
                if key_words in _:
                    matched_list.append(_)

        # 如果匹配到多个
        if len(matched_list) != 1:
            if len(matched_list) == 0:
                print('{} matching failed'.format(query_name))
                return None
            else:
                print('有多个匹配到的项 {}'.format(matched_list))

        # 默认返回一个
        return matched_list[0]

    def __filter_image(self, name_list):
        """
        list中可能有非图片的内容，去掉之
        """

    def writer(self, writer_type='wandb'):
        """
        如何记录实验数据

        """
        assert writer_type in ['wandb'], 'Not support {} so far'.format(writer_type)

        # init
        if writer_type == 'wandb':
            wandb.init(project="HDG-Test", entity="muskai")
            # 开始遍历
            for idx, item in enumerate(self.data_dict):
                fpars_list = self.data_dict[item]['fpars']
                # 开始写入
                for i in fpars_list:
                    _ = i.copy()
                    _.pop('pred_path')

                    wandb.log({
                        '{}-f1'.format(item): _['f1'],
                        '{}-precision'.format(item): _['precision'],
                        '{}-recall'.format(item): _['recall'],
                        '{}-accuracy'.format(item): _['accuracy']
                    })

        elif writer_type == 'csv':
            pass

    @staticmethod
    def std2area(narray):
        # 将std gt换成篡改区域
        narray = np.where(narray == 50, 255, narray)
        narray = np.where(narray == 100, 0, narray)
        return narray

    @staticmethod
    def get_one_dataset_fpar(dataset_pair=None):
        assert dataset_pair
        one_dataset_result = []

        for idx, item in enumerate(tqdm(dataset_pair)):
            # 获取每张图片的对应的预测结果和gt
            pred_path = item['pred']
            gt_path = item['gt']

            # 开始计算指标

            fpar_dict = Eval.get_single_fpar(pred=pred_path, label=gt_path)
            if fpar_dict is not {}:
                one_dataset_result.append(fpar_dict)

            else:
                print('计算出错')
                continue
        return one_dataset_result

    def get_all_dataset_fpar(self):

        # 开始遍历每一个数据集
        for idx1, datasets_item in enumerate(self.data_dict):
            # 开始遍历每张图片
            one_dataset_result = self.get_one_dataset_fpar(self.data_dict[datasets_item]['image_pair'])

            # 保存计算结果
            self.data_dict[datasets_item]['fpars'] = one_dataset_result

    @staticmethod
    def get_single_fpar(pred, label):
        """
        获取一张图的四个指标
        pred:网络预测结果的图片路径
        label:对应的gt所在路径
        """

        try:
            pred_img = cv2.imread(pred)
            gt_img = cv2.imread(label)
            if pred_img.shape != gt_img.shape:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

            if pred_img.shape[-1] == 3:
                pred_img = np.array(pred_img)[..., 0]
            if gt_img.shape[-1] == 3:
                gt_img = np.array(gt_img)[..., 0]

            # 都进行归一化
            pred_img = pred_img / 255
            gt_img = Eval.std2area(gt_img) / 255


            if False:
                plt.subplot(121)
                plt.imshow(pred_img)
                plt.subplot(122)
                plt.imshow(gt_img)
                plt.show()


            # TODO 全部转为灰度图

            p_score = my_precision_score(pred_img, gt_img)
            f1_score = my_f1_score(pred_img, gt_img)
            r_score = my_recall_score(pred_img, gt_img)
            a_score = my_acc_score(pred_img, gt_img)
            return {
                'pred_path': pred,
                'f1': f1_score,
                'precision': p_score,
                'recall': r_score,
                'accuracy': a_score
            }



        except Exception as e:

            print(e)
        return {}

    def __repr__(self):
        return '评测代码V1'


if __name__ == '__main__':
    pred_dict = {
        '纯unet': '/home/liu/haoran/test_results/纯unet结果',
        'band加权':'/home/liu/haoran/test_results/边缘加权监督',
        '最终的模型':'/home/liu/haoran/test_results/final-finetune2'
    }
    data_root = '/home/liu/haoran/3月最新数据/public_dataset'

    pred_root = pred_dict['最终的模型']
    using_data = {'columbia': False,
                  'coverage': True,
                  'casia': False,
                  'ps-battle': False,
                  'in-the-wild': False,
                  }
    evaler = Eval(data_root=data_root, pred_root=pred_root, using_data=using_data)
    evaler.read()
    evaler.get_all_dataset_fpar()
    # evaler.writer()
    pprint(evaler.data_dict)
