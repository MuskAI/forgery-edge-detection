
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：forgery-edge-detection 
@File    ：one_stage_test.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/3/30 8:41 PM 
'''
import os, sys, random
sys.path.append('../')
import numpy as np
from tqdm import tqdm
from prepare_datasets import Datasets
from PIL import Image
import torchvision
import cv2 as cv
import traceback
from importlib import import_module
import torch
from pprint import pprint

"""
选择模型
"""
from model.model_final import UNetStage1 as Net

class OneStageInfer:
    def __init__(self, model):
        """
        只负责输入
        @param model_name:
        @param src_data_dir:
        @param output_dir:
        """
        self.model = model
        self.error_log = []

    def read_test_data(self, src_data_dir, output_dir, name_list=None):
        test_data_path = src_data_dir
        image_name = os.listdir(test_data_path) if name_list is None else name_list
        try:
            for index, name in enumerate(tqdm(image_name)):
                img = Image.open(os.path.join(test_data_path, name)).convert('RGB')
                img = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)
                if device == 'cuda':

                    img = img[np.newaxis, :, :, :].cuda()
                else:
                    img = img[np.newaxis, :, :, :]
                output = self.model(img)


                # parse output
                if isinstance(output,dict):
                    output = output['logits']

                output = np.array(output.cpu().detach().numpy(), dtype='float32')
                output = output.squeeze(0)
                output = np.transpose(output, (1, 2, 0))
                output_ = output.squeeze(2)
                output = np.array(output_) * 255
                output = np.asarray(output, dtype='uint8')
                cv.imwrite(os.path.join(output_dir, '{}.png'.format(name.split('.')[0])), output)
        except Exception as e:
            traceback.print_exc()
            print(e)


class TestDatasets(Datasets):
    def __init__(self):
        self.using_data = {'columbia': True,
                           'coverage': True,
                           'casia': True,
                           'ps-battle': False,
                           'in-the-wild': False,
                           }

        self.datasets_path = {
            'root': '/home/liu/haoran/3月最新数据/public_dataset',
            'columbia': None,
            'coverage': None,
            'casia': None,
            'my-protocol': None
        }
        super(TestDatasets, self).__init__(model_name='final-naive2', using_data=self.using_data,
                                           datasets_path=self.datasets_path)
        self.datasets_dict = self.get_datasets_dict()
        # pprint(self.datasets_dict)
        self.one_stage_infer = OneStageInfer(model=self.get_model())
        self.infer_all_dataset()

    def infer_all_dataset(self):
        for idx, item in enumerate(tqdm(self.datasets_dict)):
            # 遍历每张图片
            self.one_stage_infer.read_test_data(src_data_dir=self.datasets_dict[item]['path'],
                                                output_dir=self.datasets_dict[item]['save_path'],
                                                name_list=self.datasets_dict[item]['names'])

    def get_model(self):
        model_path = self.model_zoo()['final-naive2']
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model = Net().to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def model_zoo(self):
        model_dict = {
            'final-naive2':'/home/liu/haoran/forgery-edge-detection/CBE-Net/save_model/0417-stage1/stage1_epoch_14-0.6170-f.4f-precision0.7652-acc0.9873-recall0.5241.pth'
            }

        return model_dict

if __name__ == '__main__':
    device = 'cuda'
    tester = TestDatasets()