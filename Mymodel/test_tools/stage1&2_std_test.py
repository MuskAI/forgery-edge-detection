#encoding=utf-8
"""
created by haoran
time 8-17
"""
import sys
sys.path.append('..')
import traceback
from model import model
from model.unet_two_stage_model_0306_3 import UNetStage1 as Net1
from model.unet_two_stage_model_0306_3 import UNetStage2 as Net2
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
from functions import sigmoid_cross_entropy_loss, cross_entropy_loss,l1_loss,wce_huber_loss
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from utils import to_none_class_map
device = torch.device("cuda")

class TestDataset:
    def __init__(self,src_data_dir=None,gt_data_dir=None,output_dir=None,save_percent=0.05):
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
            length = int(len(image_name) * self.save_percent)
            random.seed(310)
            image_name = random.sample(image_name, length)
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
                img = Image.open(image_path)
                if len(img.split()) == 4:
                    img = img.convert('RGB')
                img = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
                ])(img)

                img = img[np.newaxis, :, :, :].cuda()
                output = model1(img)
                stage1_ouput = output[0]
                # zero = torch.zeros_like(stage1_ouput)
                # one = torch.ones_like(stage1_ouput)
                #
                # rgb_pred = img * torch.where(stage1_ouput > 0.1, one, zero)
                #
                # _rgb_pred = rgb_pred.squeeze(0)
                # _rgb_pred = np.array(_rgb_pred.cpu().detach())
                # _rgb_pred = np.transpose(_rgb_pred, (1, 2, 0))

                model2_input = torch.cat((stage1_ouput, img), 1)

                output2 = model2(model2_input, output[1], output[2], output[3])

                output = np.array(output[0].cpu().detach().numpy(), dtype='float32')
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

                cv.imwrite(os.path.join(output_path1, 'output1_' + name), output)
                cv.imwrite(os.path.join(output_path2, 'output2_' + name), output2)
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
            super(SPTest,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src'), output_dir=self.src_data_train_output_dir)
            super(SPTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src'), output_dir=self.src_data_test_output_dir)
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
            super(CMTest,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src'), output_dir=self.src_data_train_output_dir)
            super(CMTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src'), output_dir=self.src_data_test_output_dir)
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
            super(TextureTestSP,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src'), output_dir=self.src_data_train_output_dir,save_percent=0.01)
            super(TextureTestSP, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src'), output_dir=self.src_data_test_output_dir,save_percent=0.01)
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
            super(TextureTestCM,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src'), output_dir=self.src_data_train_output_dir,save_percent=0.01)
            super(TextureTestCM, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src'), output_dir=self.src_data_test_output_dir,save_percent=0.01)
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

        # super(TemplateTest,self).__init__(src_data_dir=os.path.join(self.src_data_dir,'train_src'), output_dir=self.src_data_train_output_dir)
        super(TemplateTest, self).__init__(src_data_dir=os.path.join(self.src_data_dir,'test_src'), output_dir=self.src_data_test_output_dir)
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
            super(COD10K, self).__init__(src_data_dir=os.path.join(self.src_data_dir, 'train_src'),
                                         output_dir=self.src_data_train_output_dir, save_percent=0.1)
        except:
            pass
if __name__ == '__main__':
    try:
        output_path = '/home/liu/chenhaoran/test/0324_两阶段_0306_3模型_grayproblem'
        if os.path.exists(output_path):
            pass
            # traceback.print_exc('The path is already exists ,please change it ')
        else:
            os.mkdir(output_path)
            print('mkdir :',output_path)
        model_path1 = '/home/liu/chenhaoran/Mymodel/save_model/0317_stage1&2_后缀为0306_3的模型,先训练好第一阶段再训练第二阶段_grayproblem/stage1_0317_stage1&2_后缀为0306_3的模型,先训练好第一阶段再训练第二阶段_grayproblem_checkpoint14-two_stage-0.091904-f10.805107-precision0.950772-acc0.990281-recall0.712211.pth'
        model_path2 = '/home/liu/chenhaoran/Mymodel/save_model/0317_stage1&2_后缀为0306_3的模型,先训练好第一阶段再训练第二阶段_grayproblem/stage2_0317_stage1&2_后缀为0306_3的模型,先训练好第一阶段再训练第二阶段_grayproblem_checkpoint15-two_stage-0.092072-f10.702414-precision0.643135-acc0.993115-recall0.802206.pth'
        checkpoint1 = torch.load(model_path1, map_location=torch.device('cuda'))
        checkpoint2 = torch.load(model_path2, map_location=torch.device('cuda'))
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
            CasiaTest(output_dir=output_path)
            CoverageTest(output_dir=output_path)
            TextureTestSP(output_dir=output_path)
            TextureTestCM(output_dir=output_path)
            COD10K(output_dir=output_path)
            TemplateTest(output_dir=output_path)
            Negative(output_dir=output_path)
            SPTest(output_dir=output_path)
            CMTest(output_dir=output_path)
        except Exception as e:
            traceback.print_exc(e)
    except Exception as e:
        traceback.print_exc(e)