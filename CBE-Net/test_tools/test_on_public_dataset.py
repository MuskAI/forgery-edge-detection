"""
@author:haoran
tiem:3/27
在coverage casia Columbia上测试自己的算法
"""
#encoding=utf-8

import sys
sys.path.append('..')
import traceback
from model import model
from model.unet_two_stage_model_0306 import UNetStage1 as Net1
from model.unet_two_stage_model_0306 import UNetStage2 as Net2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
import matplotlib
import random
import matplotlib.pyplot as plt
import cv2 as cv
import gc
from functions import sigmoid_cross_entropy_loss, cross_entropy_loss,l1_loss,wce_huber_loss
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from utils import to_none_class_map
device = torch.device("cuda")

class TestDataset:
    def __init__(self,src_data_dir=None,gt_data_dir=None,output_dir=None,save_percent=1):
        self.save_percent = save_percent
        # self.src_data_dir = src_data_dir
        self.gt_data_dir = gt_data_dir
        self.model_dir = output_dir
        self.output_dir = output_dir

        TestDataset.read_test_data2(self,src_data_dir)

    def read_test_data(self):
        test_data_path = self.src_data_dir
        output_path = self.output_dir
        try:
            image_name = os.listdir(test_data_path)
            for index, name in enumerate(tqdm(image_name)):
                image_path = os.path.join(test_data_path, name)
                img = Image.open(image_path)
                img = torchvision.transforms.Compose([
                    # AddGlobalBlur(p=0.5),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)

                img = img[np.newaxis, :, :, :].cuda()
                output = model(img)
                output = output[0]

                output = np.array(output.cpu().detach().numpy(), dtype='float32')
                output = output.squeeze(0)

                output = np.transpose(output, (1, 2, 0))
                output_ = output.squeeze(2)

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
    def read_test_data2(self,src_data_dir):
        test_data_path =src_data_dir
        output_path = self.output_dir

        if os.path.exists(os.path.join(output_path,'stage1')):
            print('exists: ', os.path.join(output_path,'stage2'))
            pass
        else:
            os.mkdir(os.path.join(output_path, 'stage1'))
            os.mkdir(os.path.join(output_path, 'stage2'))
        output_path1 = os.path.join(output_path,'stage1')
        output_path2 = os.path.join(output_path,'stage2')
        try:
            image_name = os.listdir(test_data_path)
            length = int(len(image_name) * self.save_percent)
            random.seed(310)
            image_name = random.sample(image_name, length)
            for index, name in enumerate(tqdm(image_name)):
                image_path = os.path.join(test_data_path, name)
                src = Image.open(image_path)
                if len(src.split()) == 4:
                    src = src.convert('RGB')
                img = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(src)
                for i in range(2):
                    try:
                        img = img[np.newaxis, :, :, :].cuda()
                        output = model1(img)
                        stage1_ouput = output[0].detach()

                        model2_input = torch.cat((stage1_ouput, img), 1).detach()

                        output2 = model2(model2_input, output[1], output[2], output[3])
                        output2[0].detach()
                        break
                    except Exception as e:
                        print('The error:',name)
                        print(model2_input.shape)
                        img = torchvision.transforms.Compose([
                            torchvision.transforms.Resize((src.size[0]//2,src.size[1]//2)),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                        ])(src)
                    print('resize:',(src.size[0]//2,src.size[1]//2))

                output = np.array(stage1_ouput.cpu().detach().numpy(), dtype='float32')

                output = output.squeeze(0)
                output = np.transpose(output, (1, 2, 0))
                output_ = output.squeeze(2)

                output2 = np.array(output2[0].cpu().detach().numpy(), dtype='float32')
                output2 = output2.squeeze(0)
                output2 = np.transpose(output2, (1, 2, 0))
                output2_ = output2.squeeze(2)

                output = np.array(output_) * 255
                output = np.asarray(output, dtype='uint8')
                output2 = np.array(output2_) * 255
                output2 = np.asarray(output2, dtype='uint8')
                # print(name.split('.')[0]+'.bmp')
                cv.imwrite(os.path.join(output_path1,  (name.split('.')[0]+'.bmp')), output)
                cv.imwrite(os.path.join(output_path2, (name.split('.')[0]+'.bmp')), output2)
                del stage1_ouput,model2_input,output,output2
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            traceback.print_exc()
            print(e)


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

        super(CoverageTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir, 'src'),
                                     output_dir=self.src_data_train_output_dir,save_percent=1)



class CasiaTest(TestDataset):
    def __init__(self, src_data_dir=None, output_dir=None):

        self.src_data_dir = os.path.join(data_dir,'public_dataset/casia')
        self.src_data_output_dir = os.path.join(output_dir, 'casia_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)

        super(CasiaTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir, 'src'),
                                     output_dir=self.src_data_train_output_dir,save_percent=1)


class ColumbiaTest(TestDataset):
    def __init__(self, src_data_dir=None, output_dir=None):

        self.src_data_dir = os.path.join(data_dir,'public_dataset/columbia')
        self.src_data_output_dir = os.path.join(output_dir, 'columbia_test')
        if not os.path.exists(self.src_data_output_dir):
            os.mkdir(self.src_data_output_dir)

        self.src_data_train_output_dir = os.path.join(self.src_data_output_dir, 'pred_train')
        if not os.path.exists(self.src_data_train_output_dir):
            os.mkdir(self.src_data_train_output_dir)

        self.src_data_test_output_dir = os.path.join(self.src_data_output_dir, 'pred_test')
        if not os.path.exists(self.src_data_test_output_dir):
            os.mkdir(self.src_data_test_output_dir)

        super(ColumbiaTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir, 'src'),
                                     output_dir=self.src_data_train_output_dir,save_percent=1)



if __name__ == '__main__':

    try:
        output_path = [
            '/home/liu/chenhaoran/test/0322_stage1&2_后缀为0306的模型,只监督条带区域',
                       '/home/liu/chenhaoran/test/0322_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束',
                       '/home/liu/chenhaoran/test/0322_stage1&2_后缀为0306的模型,只监督条带区域',]

        model_path1 = [
            '/home/liu/chenhaoran/Mymodel/save_model/0322_stage1&2_后缀为0306的模型,只监督条带区域/stage1_0322_stage1&2_后缀为0306的模型,只监督条带区域_checkpoint20-two_stage-0.013404-f10.890148-precision0.928232-acc0.993772-recall0.860163.pth',
                       '/home/liu/chenhaoran/Mymodel/save_model/0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束/stage1_0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束_checkpoint6-two_stage-0.047892-f10.890272-precision0.923245-acc0.993727-recall0.864488.pth']
        model_path2 = [
            '/home/liu/chenhaoran/Mymodel/save_model/0322_stage1&2_后缀为0306的模型,只监督条带区域/stage2_0322_stage1&2_后缀为0306的模型,只监督条带区域_checkpoint20-two_stage-0.013404-f10.837347-precision0.888073-acc0.996832-recall0.797986.pth',
                       '/home/liu/chenhaoran/Mymodel/save_model/0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束/stage2_0323_stage1&2_后缀为0306的模型,只监督条带区域,无8张图约束_checkpoint6-two_stage-0.047892-f10.838588-precision0.884079-acc0.996823-recall0.802881.pth']

        for model_idx in range(1):
            if os.path.exists(output_path[model_idx]):
                pass
                # traceback.print_exc('The path is already exists ,please change it ')
            else:
                os.mkdir(output_path[model_idx])
                print('mkdir :', output_path[model_idx])
            checkpoint1 = torch.load(model_path1[model_idx], map_location=torch.device('cuda'))
            checkpoint2 = torch.load(model_path2[model_idx], map_location=torch.device('cuda'))
            model1 = Net1().to(device)
            model2 = Net2().to(device)
            # model = torch.load(model_path)
            model1.load_state_dict(checkpoint1['state_dict'])
            model2.load_state_dict(checkpoint2['state_dict'])
            model1.eval()
            model2.eval()
        data_device = '413'
        if data_device == '413':
            data_dir = '/home/liu/chenhaoran/Tamper_Data/3月最新数据'
        elif data_device == 'wkl':
            data_dir = r'D:\chenhaoran\data'

        try:
            CasiaTest(output_dir=output_path[model_idx])
            CoverageTest(output_dir=output_path[model_idx])
            ColumbiaTest(output_dir=output_path[model_idx])
        except Exception as e:
            print(e)
            traceback.print_exc(e)
    except Exception as e:
        traceback.print_exc(e)