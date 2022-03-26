"""
@author:haoran
time:0327
处理Columbia数据集
"""
from PIL import Image
import numpy as np
import matplotlib.pylab as plt
import os,sys,time
from tqdm import tqdm
import skimage
from skimage import morphology
import skimage.morphology as dilation
class Columbia:
    def __init__(self):
        pass
    def deal_with_edge(self,gt_path):
        """
        处理数据集的gt，转化为我们std gt
        图像分为4个区域：鲜红色（255,0,0），鲜绿色（0,255,0），常规红色（200,0,0）和常规绿色（0,200,0）。
        其中：两种绿色为篡改区域
        :return:
        """

        gt = Image.open(gt_path)
        gt = gt.split()[0]
        gt = np.array(gt, dtype='uint8')
        # plt.imshow(gt, cmap='gray')
        # plt.show()
        mask = np.where((gt == 0)|(gt <200),255,0)
        mask = mask > 0
        mask = morphology.remove_small_objects(mask,min_size=500)
        mask = morphology.remove_small_holes(mask,area_threshold=500)
        mask = np.where(mask==True,1,0)
        dou_gt = self.area2edge(mask)
        # plt.subplot(121)
        # plt.imshow(mask,cmap='gray')
        # plt.subplot(122)
        # plt.imshow(dou_gt, cmap='gray')
        # plt.show()
        return dou_gt
    def area2edge(self, orignal_mask):
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

if __name__ == '__main__':
    columbia = Columbia()
    gt_dir = r'D:\lab\tamper_detection\public_data\Columbia\Columbia Uncompressed Image Splicing Detection\4cam_splc.tar\4cam_splc\edgemask'
    gt_list = os.listdir(gt_dir)
    save_dir = r'D:\lab\tamper_detection\public_data\Columbia\Columbia Uncompressed Image Splicing Detection\4cam_splc.tar\dou_gt500'
    print('The len of gt is:',len(gt_list))
    for idx,item in enumerate(tqdm(gt_list)):
        if 'edgemask_3' in item:
            continue
        try:
            dou_gt = columbia.deal_with_edge(os.path.join(gt_dir,item))
            dou_gt = Image.fromarray(dou_gt).convert('RGB')
            dou_gt.save(os.path.join(save_dir,item.split('.')[0]+'.bmp'))
        except Exception as e:
            print(e)
            continue














