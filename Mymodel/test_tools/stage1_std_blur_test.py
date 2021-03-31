#encoding=utf-8
"""
created by haoran
time 8-17
"""
import sys
sys.path.append('..')
import traceback
from model.unet_two_stage_model_0306 import UNetStage1 as Net1
import os
import numpy as np
from PIL import Image
import torchvision
import torch
from tqdm import tqdm

import random
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
from functions import sigmoid_cross_entropy_loss, cross_entropy_loss,l1_loss,wce_huber_loss
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from utils import to_none_class_map
device = torch.device("cuda")

class TestDataset:
    def __init__(self,src_data_dir=None,gt_data_dir=None,output_dir=None,save_percent=0.05):
        self.save_percent = save_percent
        self.src_data_dir = src_data_dir
        self.gt_data_dir = gt_data_dir
        self.model_dir = output_dir
        self.output_dir = output_dir
        self.read_test_data()

    def read_test_data(self):
        test_data_path = self.src_data_dir
        print('we are in :',test_data_path)
        output_path = self.output_dir
        try:
            image_name = os.listdir(test_data_path)
            length = int(len(image_name) * self.save_percent)
            random.seed(310)
            image_name = random.sample(image_name,length)
            for index, name in enumerate(tqdm(image_name)):
                image_path = os.path.join(test_data_path, name)
                img = Image.open(image_path)
                img = torchvision.transforms.Compose([
                    # AddGlobalBlur(p=0.5),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)


                img = img.unsqueeze(0)
                #
                # img = img[np.newaxis, :, :, :]
                img = img.cuda()
                output = model1(img)[0]
                del img
                output = np.array(output.cpu().detach().numpy(), dtype='float32')
                output = output.squeeze(0)
                output = np.transpose(output, (1, 2, 0))
                output_ = output.squeeze(2)

                # TODO 1 :这里开始计算loss 和相关的指标
                # 在这里计算一下loss
                # output_ = torch.from_numpy(output_)
                # gt_ = torch.from_numpy(gt_)
                # loss = wce_huber_loss(output_,gt_)
                # print(loss)
                # plt.figure('prediction')
                # plt.imshow(output_)
                # plt.show()
                output = np.array(output_) * 255
                output = np.asarray(output, dtype='uint8')
                cv.imwrite(os.path.join(output_path, 'output_' + name), output)

        except Exception as e:
            traceback.print_exc()
            print(e)


class SPTest(TestDataset):
    def __init__(self,src_data_dir=None, output_dir=None):
        self.src_data_dir = os.path.join(data_dir,'coco_sp')
        self.src_data_output_dir = os.path.join(output_dir,'coco_sp_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)
        try:
            super(SPTest,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src_blur'), output_dir=self.src_data_train_output_dir)
            super(SPTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src_blur'), output_dir=self.src_data_test_output_dir)
        except Exception as e:
            traceback.print_exc(e)

class CMTest(TestDataset):
    def __init__(self,src_data_dir = None, output_dir=None):
        self.src_data_dir = os.path.join(data_dir,'coco_cm')
        self.src_data_output_dir = os.path.join(output_dir,'coco_cm_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)
        try:
            super(CMTest,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src_blur'), output_dir=self.src_data_train_output_dir)
            super(CMTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src_blur'), output_dir=self.src_data_test_output_dir)
        except:
            pass
class CoverageTest(TestDataset):
    def __init__(self, src_data_dir=None, output_dir=None):
        self.src_data_dir = os.path.join(data_dir,'public_dataset/coverage')
        self.src_data_output_dir = os.path.join(output_dir, 'coverage_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)
        try:
            super(CoverageTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir, 'src'),
                                         output_dir=self.src_data_train_output_dir,save_percent=1)

        except:
            pass
class TextureTestSP(TestDataset):
    def __init__(self,src_data_dir = None, output_dir=None):
        self.src_data_dir = os.path.join(data_dir,'0108_texture_and_casia_template_divide')
        self.src_data_output_dir = os.path.join(output_dir,'texture_sp_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)
        try:
            super(TextureTestSP,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src_blur'), output_dir=self.src_data_train_output_dir,save_percent=0.01)
            super(TextureTestSP, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src_blur'), output_dir=self.src_data_test_output_dir,save_percent=0.01)
        except:
            pass
class TextureTestCM(TestDataset):
    def __init__(self,src_data_dir = None, output_dir=None):
        self.src_data_dir = os.path.join(data_dir,'periodic_texture/divide')
        self.src_data_output_dir = os.path.join(output_dir,'texture_cm_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)
        try:
            super(TextureTestCM,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src_blur'), output_dir=self.src_data_train_output_dir,save_percent=0.01)
            super(TextureTestCM, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src_blur'), output_dir=self.src_data_test_output_dir,save_percent=0.01)
        except:
            pass
class TemplateTest(TestDataset):
    def __init__(self,src_data_dir = None, output_dir=None):
        self.src_data_dir = os.path.join(data_dir,'coco_casia_template_after_divide')
        self.src_data_output_dir = os.path.join(output_dir,'coco_casia_template_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)

        super(TemplateTest,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src_blur'), output_dir=self.src_data_train_output_dir)
        super(TestDataset, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src_blur'), output_dir=self.src_data_test_output_dir)
class CasiaTest(TestDataset):
    def __init__(self, src_data_dir=None, output_dir=None):

        self.src_data_dir = os.path.join('/home/liu/chenhaoran/Tamper_Data/3月最新数据', 'public_dataset/casia')
        self.src_data_output_dir = os.path.join(output_dir, 'casia_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)
        try:
            super(CasiaTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir, 'src'),
                                         output_dir=self.src_data_train_output_dir,save_percent=1)

        except:
            pass
class Negative(TestDataset):
    def __init__(self, src_data_dir=None, output_dir=None):

        self.src_data_dir = data_dir
        self.src_data_output_dir = os.path.join(output_dir, 'negative_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)
        try:
            super(Negative, self).__init__(src_data_dir=os.path.join(self.src_data_dir, 'negative'),
                                         output_dir=self.src_data_train_output_dir,save_percent=0.1)

        except:
            pass
class COD10K(TestDataset):
    def __init__(self, src_data_dir=None, output_dir=None):

        self.src_data_dir = os.path.join(data_dir,'COD10K')
        self.src_data_output_dir = os.path.join(output_dir, 'COD10K_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)
        try:
            super(COD10K, self).__init__(src_data_dir=os.path.join(self.src_data_dir, 'train_src_blur'),
                                         output_dir=self.src_data_train_output_dir, save_percent=0.1)
        except:
            pass
if __name__ == '__main__':
    output_path = '/home/liu/chenhaoran/test/0321_边缘模糊测试_0321_stage1后缀为0306的模型，训练双边缘第一阶段,无八张图约束，边缘模糊数据'
    if os.path.exists(output_path):
        pass
    else:
        os.mkdir(output_path)
        print('mkdir :',output_path)
    model_path1 = '/home/liu/chenhaoran/Mymodel/save_model/0321_stage1后缀为0306的模型，训练双边缘第一阶段,无八张图约束，边缘模糊数据/stage1_0321_stage1后缀为0306的模型，训练双边缘第一阶段,无八张图约束，边缘模糊数据_checkpoint77-two_stage-0.150757-f10.784330-precision0.671825-acc0.991222-recall0.946185.pth'
    # model_path2 = '/home/liu/chenhaoran/Mymodel/save_model/0308_stage1&2_后缀为0306的模型,两阶段联合训练/stage2_0308_stage1&2_后缀为0306的模型,两阶段联合训练_checkpoint6-two_stage-0.316201-f10.334154-precision0.973132-acc0.982770-recall0.206252.pth'
    checkpoint1 = torch.load(model_path1, map_location=torch.device('cuda'))
    # checkpoint2 = torch.load(model_path2, map_location=torch.device('cpu'))
    model1 = Net1().to(device)
    model1.load_state_dict(checkpoint1['state_dict'])
    model1.eval()
    data_device='413'
    if data_device=='413':
        data_dir = '/home/liu/chenhaoran/Tamper_Data/edge_blur'
    elif data_device =='wkl':
        data_dir = r'D:\chenhaoran\data'


    try:
        # CasiaTest(output_dir=output_path)
        # CoverageTest(output_dir=output_path)
        # TextureTestSP(output_dir=output_path)
        # TextureTestCM(output_dir=output_path)
        COD10K(output_dir=output_path)
        # TemplateTest(output_dir=output_path)
        # Negative(output_dir=output_path)
        SPTest(output_dir=output_path)
        CMTest(output_dir=output_path)
    except Exception as e:
        traceback.print_exc(e)
