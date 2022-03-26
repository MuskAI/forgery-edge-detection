"""
created by haoran
time :2020-8-10
将mask图和原图叠加在一起用来检查mask图是否准确
"""
import numpy as np
from PIL import  Image
import cv2 as cv
import os
import matplotlib.pyplot as plt
def mask_src_to_check(mask,src):
    plt.figure('mask')
    plt.imshow(mask)
    plt.show()
    plt.figure('src')
    plt.imshow(src)
    plt.show()

    # 开始处理
    # mask = np.array(mask)
    # src = np.array(src)-100
    #
    # mask_src = src
    # mask_src[:,:,0] += mask
    # mask_src[:, :, 1] += mask
    # mask_src[:, :, 2] += mask
    #
    # mask_src = Image.fromarray(mask_src)
    mask3 = Image.new(mode='RGB',size=(src.size))
    mask3 = np.array(mask3)
    mask3[:,:,0] = mask
    mask3[:, :, 1] = mask
    mask3[:, :, 2] = mask
    mask3 = Image.fromarray(mask3)
    mask_src = Image.blend(src, mask3, .7)
    plt.figure('mask_src')
    plt.imshow(mask_src)
    plt.show()

if __name__ == '__main__':
    src_path = r'D:\实验室\图像篡改检测\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages\2007_001299.jpg'
    mask_path = r'D:\实验室\图像篡改检测\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationObject\2007_001299.png'
    mask = Image.open(mask_path)
    src = Image.open(src_path)
    mask_src_to_check(mask,src)