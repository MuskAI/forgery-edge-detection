from functions import my_precision_score, my_acc_score, my_recall_score, wce_huber_loss, my_f1_score,wce_dice_huber_loss
import numpy as np
import torch
import pandas as pd
import random
import os, sys
import traceback
import PIL.Image as Image
import cv2 as cv
import matplotlib.pyplot as plt
"""
created by haoran
time: 11/8
usage:
1. using band_pred to gen a combine image which shows that src gt band_gt band_pred
2. analyze l1 f1 precision acc and recall to gen a excel file
"""


class Analyze:
    def __init__(self):
        """
        pred_dir
        """
        self.pred_dir = '/media/liu/File/10月数据准备/1108_数据测试/sp_train_data/1120stage_2_pred'
        self.src_dir = '/media/liu/File/Sp_320_dataset/tamper_result_320'
        self.gt_dir = '/media/liu/File/Sp_320_dataset/ground_truth_result_320'
        """
        save_dir
        """
        self.save_band_dir = '/media/liu/File/10月数据准备/1108_数据测试/sp_train_two_stage/edge2'
        self.save_combine_dir = '/media/liu/File/10月数据准备/1108_数据测试/sp_train_two_stage/combineImgedge2'
        self.save_excel_dir = '/media/liu/File/10月数据准备/1108_数据测试/sp_train_two_stage/sp_1120edgetest2.xlsx'
        if os.path.exists(self.save_combine_dir):
            pass
        else:
            os.mkdir(self.save_combine_dir)

        if os.path.exists(self.save_band_dir):
            pass
        else:
            os.mkdir(self.save_band_dir)
    def analyze_all(self):
        """

        :return:
        """
        path_dir = self.pred_dir
        # 1. path check
        if os.path.exists(path_dir):
            print('path ok')
        else:
            print('Path Not find:', path_dir)
            sys.exit()

        pred_list = os.listdir(path_dir)
        fileNumber=len(pred_list)
        print('The number of images are: ', fileNumber)
        rate=1
        pickNumber=int(fileNumber*rate)
        sample=random.sample(pred_list,pickNumber)

        f1_score_list = []
        pred_name_list = []
        loss_list = []
        src_name_list = []
        gt_name_list = []
        band_gt_name_list = []
        precision_score_list = []
        acc_list = []
        recall_list = []

        combineArray = np.zeros((320, 4*320, 3))

        for index, name in enumerate(sample):
            print(index,'/',len(sample))
            src_path, gt_path = Analyze.__find_src_and_gt(self, name)
            pred_img = os.path.join(path_dir, name)
            pred_img = Image.open(pred_img)
            src_img = Image.open(src_path)
            gt_img = Image.open(gt_path)

            # check channel required 1 dim
            if len(pred_img.split()) == 3:
                pred_img = pred_img.split()[0]
            else:
                pass
            if len(gt_img.split()) == 3:
                gt_img = gt_img.split()[0]
            else:
                pass

            # convert to ndarray and normalize,then to tensor
            pred_ndarray = np.array(pred_img)
            pred_ndarray3D = np.expand_dims(pred_ndarray, axis=2)
            pred_ndarray = pred_ndarray / 255

            pred_ndarray4D = pred_ndarray[np.newaxis, np.newaxis, :, :]
            # convert numpy to tensor
            pred_img_tensor = torch.from_numpy(pred_ndarray4D)

            # compute loss
            gt_ndarray = np.array(gt_img)
            gt = gt_ndarray.copy()
            gt = np.where((gt == 100) | (gt == 255), 1, 0)
            gt_ndarray3D = np.expand_dims(gt_ndarray, axis=2)
            gt_ndarray4D = gt_ndarray[np.newaxis, np.newaxis, :, :]
            band_gt_np = Analyze.__gen_band_gt(self, gt_ndarray4D)
            band_gt_np3D = band_gt_np.squeeze(0)
            band_gt_np2D = band_gt_np3D.squeeze(0)
            band_gt_np3DLast1 = np.expand_dims(band_gt_np2D, axis=2)
            band_gt_np3DLast1=band_gt_np3DLast1*255
            band_gt_img = Image.fromarray(band_gt_np2D)
            band_gt_prefix = 'band5_'

            gt_name = gt_path.split('/')[-1]

            band_gt_name = band_gt_prefix + gt_name
            band_gt_img.save(os.path.join(self.save_band_dir, band_gt_name))

            band_gt_tensor = torch.from_numpy(band_gt_np)


            # compute loss,f1,acc,precision,recall
            gt = torch.from_numpy(gt)
            gt = gt.unsqueeze(0)
            gt = gt.unsqueeze(0)
            loss_tonsor = wce_dice_huber_loss(pred_img_tensor.float(), gt.float())

            # loss_tonsor = wce_dice_huber_loss(pred_img_tensor.float(), band_gt_tensor.float())
            loss = loss_tonsor.item()

            f1_score = my_f1_score(pred_img_tensor, band_gt_tensor)

            acc_score = my_acc_score(pred_img_tensor, band_gt_tensor)
            recall = my_recall_score(pred_img_tensor, band_gt_tensor)
            precision = my_precision_score(pred_img_tensor, band_gt_tensor)

            # output to csv
            f1_score_list.append(f1_score)
            pred_name_list.append(name)
            loss_list.append(loss)
            acc_list.append(acc_score)
            recall_list.append(recall)
            precision_score_list.append(precision)

            src_name = src_path.split('/')[-1]
            src_name_list.append(src_name)

            gt_name_list.append(gt_name)

            band_gt_name_list.append(band_gt_name)

            # combine_plot
            src_ndarray = np.array(src_img)
            combineArray[:, :320, :] = src_ndarray
            combineArray[:, 320:640, :] = gt_ndarray3D
            combineArray[:, 640:960, :] = band_gt_np3DLast1
            combineArray[:, 960:, :] = pred_ndarray3D

            combineImg = Image.fromarray(combineArray.astype(np.uint8))
            combineImg_prefix = 'comb_'
            combineImg_name =combineImg_prefix + src_name
            combineImg.save(os.path.join(self.save_combine_dir,combineImg_name))

            # difficult top-k
        data = {
            'srcName': src_name_list,
            'gtName': gt_name_list,
            'bandGtName': band_gt_name_list,
            'predName': pred_name_list,
            'loss': loss_list,
            'precision': precision_score_list,
            'recall': recall_list,
            'f1': f1_score_list,
            'acc': acc_list
        }
        test = pd.DataFrame(data)
        test.to_excel(self.save_excel_dir)

    def __find_src_and_gt(self, name):
        """
        using pred name to find src and gt
        :return:src_path, gt_path
        输出的名字：output_Sp_Default_34_445004_zebra -->
        """

        pred_name = name
        src_name = pred_name.replace('output1_', '')
        gt_name = pred_name.replace('output1_', '').replace('Default','Gt').replace('jpg', 'bmp').replace('png', 'bmp').replace('output_poisson', 'Gt')
        src_path = os.path.join(self.src_dir, src_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        if os.path.exists(gt_path and src_path):
            pass
        else:
            print(gt_path or src_path, ' not exists')
            traceback.print_exc()
            sys.exit()
        return src_path, gt_path

    def loss_analyze(self, gt, pred):
        """

        :return:
        """
        # input gt
        # input band_pred
        # Image -> numpy -> tensor
        # loss
        # save result(csv)
        # 1. ranking ---> difficult top-k & easy case top -n

    def __gen_band_gt(self, gt):
        """
        01 mask 图gen边缘条带图
        :param gt: 01 mask图 输入的是numpy dim 4 维，B C H W
        :return: numpy,01 边缘条带,list B C H W
        """
        gt = np.where((gt == 100) | (gt == 255), 1, 0)
        gt = np.array(gt, dtype='uint8')
        band_gt = gt
        for i in range(gt.shape[0]):
            _gt = gt[i, :, :, :]
            _gt = _gt.squeeze(0)
            _gt = cv.cvtColor(np.asarray(_gt), cv.COLOR_GRAY2BGR)
            _gt = np.array(_gt, dtype='uint8')
            cv2_gt = cv.cvtColor(_gt, cv.COLOR_RGB2GRAY)
            kernel = np.ones((5, 5), np.uint8)
            cv2_gt = cv.dilate(cv2_gt, kernel)
            _band = Image.fromarray(cv.cvtColor(cv2_gt, cv.COLOR_BGR2RGB))
            _band = np.array(_band)[:, :, 0]
            band_gt[i, :, :, :] = np.expand_dims(_band, 0)

        return band_gt


if __name__ == '__main__':
    analyze = Analyze()
    analyze.analyze_all()
