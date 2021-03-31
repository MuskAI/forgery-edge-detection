"""
created by haoran
根据GT生成条带
"""
import os,sys
from PIL import Image
import numpy as np
import cv2 as cv
class GenBandGt():
    def __init__(self):
        pass
    def gen_band_gt(self,gt_dir,save_dir):

        if GenBandGt.__path_check(self,gt_dir) and GenBandGt.__path_check(self,save_dir,none_create=True):
            gt_list = os.listdir(gt_dir)
            length = len(gt_list)
            for index, item in enumerate(gt_list):
                # gt = GenBandGt.__input_ruler(self, path=os.path.join(gt_dir,item) , open_type='cv2')
                print(os.path.join(gt_dir,item))
                gt = Image.open(os.path.join(gt_dir,item))
                gt = cv.cvtColor(np.asarray(gt), cv.COLOR_GRAY2BGR)
                gt = np.array(gt)
                print(gt.shape)
                gt = np.array(np.where((gt==255) | (gt==100),255,0),dtype='uint8')[:,:,0:3]
                cv2_gt = cv.cvtColor(gt,cv.COLOR_RGB2GRAY)

                kernel = np.ones((5, 5), np.uint8)
                cv2_gt = cv.dilate(cv2_gt, kernel)

                GenBandGt.__save_image(self,cv2_gt,os.path.join(save_dir,item.split('.')[0]+'_band_gt'+'.'+item.split('.')[1]))
                print('The process is %d / %d'%(index,length))




        pass
    def __path_check(self,path, none_create=False):
        if not none_create:
            if os.path.exists(path):
                    return True
            else:
                return False
        else:
            if os.path.exists(path):
                    return True
            else:
                os.mkdir(path)
                return True

    def __input_ruler(self,path, open_type = 'Image'):
        if open_type == 'Image':
            I = Image.open(path)
            if len(I.split()) == 3:
                I = I.split()[0]
            else:
                pass
            return I
        elif open_type == 'cv2':
            I = cv.imread(path)
            return I

    def __save_image(self,image,path):
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
            image.save(path)

if __name__ == '__main__':
    gt_dir = '/media/liu/File/10月数据准备/10月12日实验数据/cm/test_gt_train_percent_0.80@8_20'
    save = '/media/liu/File/10月数据准备/10月12日实验数据/cm/band_save'
    GenBandGt().gen_band_gt(gt_dir,save)