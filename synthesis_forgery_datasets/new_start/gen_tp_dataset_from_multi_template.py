"""
created by haoran
time : 20201209
"""
from PIL import Image
import cv2 as cv
import os
import pandas as pd
import numpy as np
import random
import traceback
import warnings
import matplotlib.pylab as plt
import skimage.morphology as dilation
from image_crop import crop as my_crop
class GenTpFromTemplate:
    def __init__(self, template_dir=None, image_dir=None):
        self.template_dir = template_dir
        self.image_dir = image_dir
        self.template_list = os.listdir(template_dir)
        self.image_list = os.listdir(image_dir)
        self.tp_image_save_dir = 'D:/Image_Tamper_Project/Lab_project_code/Image-Tamper-Detection/TempWorkShop/1219_texture_and_coco_template_src'
        self.tp_gt_save_dir = 'D:/Image_Tamper_Project/Lab_project_code/Image-Tamper-Detection/TempWorkShop/1219_texture_and_coco_template_gt'

        if os.path.exists(self.tp_image_save_dir):
            pass
        else:
            print('create dict:',self.tp_image_save_dir)
            os.mkdir(self.tp_image_save_dir)

        if os.path.exists(self.tp_gt_save_dir):
            pass
        else:
            print('create dict:', self.tp_gt_save_dir)
            os.mkdir(self.tp_gt_save_dir)


    def gen_method_both_fix_size(self,width = 320, height = 320):
        """
        1. resize template to 320
        2. using crop 320 data to generate
        :return:
        """
        # 1 resize template
        template_dir = self.template_dir
        image_dir = self.image_dir
        template_list = self.template_list
        image_list = self.image_list

        for idx,item in enumerate(template_list):
            gt_name = item
            print('%d / %d'%(idx,len(template_list)))
            I = Image.open(os.path.join(template_dir,item))
            # deal with channel issues
            if len(I.split()) != 2:
                I = I.split()[0]
            else:
                pass
            I = I.resize((width, height), Image.ANTIALIAS)
            I = np.array(I,dtype='uint8')
            I = np.where(I>128,1,0)
            I = np.array(I, dtype='uint8')

            # random choose two images from fix size coco dataset
            gt = I.copy()
            for i in range(999):
                img_1_name = random.sample(image_list,1)[0]
                img_2_name = random.sample(image_list,1)[0]
                _ = open
                if img_1_name == img_2_name:
                    if i == 998:
                        traceback.print_exc()
                    else:
                        continue
                else:
                    img_1 = Image.open(os.path.join(image_dir, img_1_name))
                    img_2 = Image.open(os.path.join(image_dir, img_2_name))
                    if len(img_1.split())!=3 or len(img_2.split()) != 3:
                        continue
                    else:
                        break

            try:
                img_1 = np.array(img_1, dtype='uint8')
                img_2 = np.array(img_2, dtype='uint8')

                tp_img_1 = img_1.copy()
                tp_img_1[:,:,0] = I * img_1[:,:,0]
                tp_img_1[:,:,1] = I * img_1[:,:,1]
                tp_img_1[:,:,2] = I * img_1[:,:,2]

                I_reverse = np.where(I == 1, 0, 1)
                tp_img_2 = img_2.copy()

                tp_img_2[:,:,0] = I_reverse * img_2[:,:,0]
                tp_img_2[:,:,1] = I_reverse * img_2[:,:,1]
                tp_img_2[:,:,2] = I_reverse * img_2[:,:,2]
            except Exception as e:
                print(img_1_name)
                print(img_2_name)
                print(e)
            tp_img = tp_img_1 + tp_img_2
            # GenTpFromTemplate.__show_img(self, tp_img)


            # prepare to save
            tp_img = np.array(tp_img,dtype='uint8')
            double_edge_gt = GenTpFromTemplate.__mask_to_double_edge(self,gt)
            tp_gt = np.array(double_edge_gt, dtype='uint8')

            tp_img = Image.fromarray(tp_img)
            tp_gt = Image.fromarray(tp_gt)

            tp_img.save(os.path.join(self.tp_image_save_dir,
                                     gt_name.split('.')[0]+'_'+img_1_name.split('.')[0]+'_'+img_2_name.split('.')[0])+'.png')
            tp_img.save(os.path.join(self.tp_image_save_dir,
                                     gt_name.split('.')[0]+'_'+img_1_name.split('.')[0] + '_' + img_2_name.split('.')[0]) + '.jpg')

            tp_gt.save(os.path.join(self.tp_gt_save_dir,
                                     gt_name.split('.')[0]+'_'+img_1_name.split('.')[0] + '_' + img_2_name.split('.')[0]) + '.bmp')


    def __mask_to_double_edge(self, orignal_mask):
            """
            :param orignal_mask: 输入的是 01 mask图
            :return: 255 100 50 mask 图
            """
            # print('We are in mask_to_outeedge function:')
            try:
                mask = orignal_mask
                # print('the shape of mask is :', mask.shape)
                selem = np.ones((3, 3))
                dst_8 = dilation.binary_dilation(mask, selem=selem)
                dst_8 = np.where(dst_8 == True, 1, 0)
                difference_8 = dst_8 - orignal_mask

                difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
                difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
                double_edge_candidate = difference_8_dilation + mask
                double_edge = np.where(double_edge_candidate == 2, 1, 0)
                ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
                    mask == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
                ground_truth = np.where(ground_truth == 305, 255, ground_truth)
                ground_truth = np.array(ground_truth, dtype='uint8')
                return ground_truth

            except Exception as e:
                print(e)

    def __show_img(self,img):
        try:
            plt.figure('show_img')
            plt.imshow(img)
            plt.show()
        except Exception as e:
            print(e)

    def __path_check(self):
        """
        进行输入路径的检查
        :return:
        """

    def __prepare_template(self):

        pass
    def __choose_image(self):
        pass
    def __match_to_tamper(self):
        pass



class GenMultiTpFromTemplate(GenTpFromTemplate):
    def __init__(self,template_dir=None, image_dir=None):
        template_dir1 = 'D:\Image_Tamper_Project\Lab_project_code\Image-Tamper-Detection\make_tamper_dataset_from_coco\TempWorkShop\gt'
        template_dir2 = 'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA\casia2groundtruth-master\CASIA 2 Groundtruth'
        template_dir3 = r'H:\8_20_dataset_after_divide\test_gt_train_percent_0.80@8_20'

        template_dir = template_dir3

        image_dir1 = 'D:\实验室\图像篡改检测\数据集\COCO_320_CROP6'
        image_dir2 = 'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA\CASIA 2.0\CASIA 2.0\Au'
        image_dir3 = r'C:\Users\musk\Desktop\smooth5\texture'
        image_dir = image_dir3

        super(GenMultiTpFromTemplate, self).__init__(template_dir,image_dir)

        # 对于大于320的图，是否选用resize操作
        self.resize_flag = False
    def gen_method_both_fix_size(self,width = 320, height = 320):
        """
        函数重载
        这个函数将使用多种数据集的模板和多种数据集的template 和多种数据集来生成合成数据
        对于小于320的图 进行resize操作， 对于 大于320的图采取crop操作
        :return:
        """
        # 1 resize template
        template_dir = self.template_dir
        image_dir = self.image_dir
        template_list = self.template_list
        image_list = self.image_list

        for idx,item in enumerate(template_list):
            gt_name = item
            print('%d / %d'%(idx,len(template_list)))
            I_gt = Image.open(os.path.join(template_dir,item))
            # deal with channel issues
            if len(I_gt.split()) != 2:
                I = I_gt.split()[0]
            else:
                pass

            # 父类的template 都是mask区域，但是我现在这里使用的mask都是标注清楚四个区域的My Gt
            # 所以下面需要进行一些操作使其转化成
            # my gt to 01 mask
            I = np.array(I,dtype='uint8')
            # read mygt is 0 50 100 255 ;so we need convert it to 01 mask;which tamper area is 1
            I = np.where((I == 0) | (I == 100), 0, 1)
            # self._GenTpFromTemplate__show_img(I)

            # random choose two images from fix size coco dataset
            gt = I.copy()
            for i in range(999):
                img_1_name = random.sample(image_list,1)[0]
                img_2_name = random.sample(image_list,1)[0]
                if img_1_name.split('.')[-1] == 'db' or img_2_name.split('.')[-1] == 'db':
                    continue
                if img_1_name == img_2_name:
                    if i == 998:
                        traceback.print_exc()
                    else:
                        continue
                else:

                    img_1 = Image.open(os.path.join(image_dir, img_1_name))
                    img_2 = Image.open(os.path.join(image_dir, img_2_name))
                    img_1 = self.resize_and_crop(img_1)
                    img_2 = self.resize_and_crop(img_2)
                    if img_1==False or img_2==False:
                        traceback.print_exc()
                        exit(1)
                    else:
                        pass

                    if len(img_1.split()) != 3 or len(img_2.split()) != 3:
                        continue
                    else:
                        break

            try:
                img_1 = np.array(img_1, dtype='uint8')
                img_2 = np.array(img_2, dtype='uint8')

                tp_img_1 = img_1.copy()
                tp_img_1[:,:,0] = I * img_1[:,:,0]
                tp_img_1[:,:,1] = I * img_1[:,:,1]
                tp_img_1[:,:,2] = I * img_1[:,:,2]

                I_reverse = np.where(I == 1, 0, 1)
                tp_img_2 = img_2.copy()

                tp_img_2[:,:,0] = I_reverse * img_2[:,:,0]
                tp_img_2[:,:,1] = I_reverse * img_2[:,:,1]
                tp_img_2[:,:,2] = I_reverse * img_2[:,:,2]
            except Exception as e:
                print(img_1_name)
                print(img_2_name)
                print(e)
            tp_img = tp_img_1 + tp_img_2
            # self.show_img(tp_img)
            # self._GenTpFromTemplate__show_img(tp_img)
            # prepare to save
            tp_img = np.array(tp_img,dtype='uint8')
            # double_edge_gt = GenTpFromTemplate.__mask_to_double_edge(self,gt)
            tp_img = Image.fromarray(tp_img)
            tp_gt = I_gt
            tp_img.save(os.path.join(self.tp_image_save_dir,
                                     gt_name.split('.')[0]+'_'+img_1_name.split('.')[0]+'_'+img_2_name.split('.')[0])+'.png')
            tp_img.save(os.path.join(self.tp_image_save_dir,
                                     gt_name.split('.')[0]+'_'+img_1_name.split('.')[0] + '_' + img_2_name.split('.')[0]) + '.jpg')

            tp_gt.save(os.path.join(self.tp_gt_save_dir,
                                     gt_name.split('.')[0]+'_'+img_1_name.split('.')[0] + '_' + img_2_name.split('.')[0]) + '.bmp')
    def resize_and_crop(self,img,width=320,height=320):
        """
        deal 2 kinds of situation
        1. the size of image both larger than 320, so we need choose crop or resize, in our implementation
           we just crop it
        2. the size of image smaller than 320 , so we only need to resize it
        3. the input img is Image type , and the output is also Image type
        :param img:
        :param width:
        :param height:
        :return:
        """
        # 判断是否需要resize
        img_size = img.size
        if img_size[0] < width or img_size[1] < height:
            # 这里开始resize
            img = img.resize((width, height), Image.ANTIALIAS)
        elif img_size[0] >= width or img_size[1] >= height:
            if self.resize_flag == True:
                # do resize process
                pass
            else:
                # do crop process
                # image_crop need imput is numpy
                img = np.array(img,dtype='uint8')
                img = my_crop(img)
                img = np.array(img, dtype='uint8')
                img = Image.fromarray(img)
        else:
            print('error')
        if img != 'error':
            return img
        else:
            return False



if __name__ == '__main__':
    template_dir = 'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA\casia2groundtruth-master\CASIA 2 Groundtruth'

    # you only need to using this cmd to gen
    GenMultiTpFromTemplate().gen_method_both_fix_size()