from pycocotools.coco import COCO
import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib
import pylab
import os
import sys
import time
from PIL import Image
from PIL import ImageFilter
import argparse
import sys
import pdb
from get_double_edge import mask_to_outeedge
from image_Fusion import Possion
import poisson_image_editing
import skimage.morphology as dilation
import traceback
import myCalImage
import image_crop
matplotlib.use('Qt5Agg')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='input begin and end category')
    parser.add_argument('--begin', dest='begin',
                        help='begin type of cat', default=None, type=int)
    parser.add_argument('--end', dest='end',
                        help='begin type of cat',
                        default=None, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def splicing_tamper_one_image(input_foreground, input_background, mask, similar_judge=True):
    """
    在输入这个函数之前，输入的两个参数需要以及经过挑选了的,需要import poisson融合的代码
    :param foreground:
    :param background:
    :return: 返回两个参数：直接篡改图, poisson融合篡改图, GT
    """

    I = input_foreground
    I1 = I
    # mask 是 01 蒙版
    I1[:, :, 0] = np.array(I[:, :, 0] * mask)
    I1[:, :, 1] = np.array(I[:, :, 1] * mask)
    I1[:, :, 2] = np.array(I[:, :, 2] * mask)
    # differece_8是background的edge
    difference_8 = mask_to_outeedge(mask)

    difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
    difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
    double_edge_candidate = difference_8_dilation + mask
    double_edge = np.where(double_edge_candidate == 2, 1, 0)
    ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(mask == 1, 50,0)  # 所以内侧边缘就是100的灰度值
    b1 = input_background

    background = Image.fromarray(b1, 'RGB')
    foreground = Image.fromarray(I1, 'RGB').convert('RGBA')
    datas = foreground.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)
    foreground.putdata(newData)
    background = background.resize((foreground.size[0], foreground.size[1]), Image.ANTIALIAS)
    # input_background = np.resize(input_background,(foreground.size[1], foreground.size[0],3))
    background_area = np.array(background)

    try:

        if similar_judge:
            foreground_area = I1
            background_area[:, :, 0] = np.array(background_area[:, :, 0] * mask)
            background_area[:, :, 1] = np.array(background_area[:, :, 1] * mask)
            background_area[:, :, 2] = np.array(background_area[:, :, 2] * mask)

            foreground_area = Image.fromarray(foreground_area)
            background_area = Image.fromarray(background_area)

            if myCalImage.calc_similar(foreground_area, background_area) > 0.2:
                pass
            else:
                return False, False, False
    except Exception as e:
        print(e)
        traceback.print_exc()

    try:
        mask = Image.fromarray(mask)
    except Exception as e:
        print('mask to Image error', e)
    # 在这里的时候，mask foreground background 尺寸都是一致的了，poisson融合时，offset置为0
    try:
        poisson_foreground = cv2.cvtColor(np.asarray(foreground.convert('RGB')), cv2.COLOR_RGB2BGR)
        poisson_background = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)
        poisson_mask = np.asarray(mask)
        poisson_mask = np.where(poisson_mask == 1, 255, 0)
        poisson_fusion_image = poisson_image_editing.poisson_fusion(poisson_foreground, poisson_background,
                                                                    poisson_mask)
        poisson_fusion_image = Image.fromarray(cv2.cvtColor(poisson_fusion_image, cv2.COLOR_BGR2RGB))
        background.paste(foreground, (0, 0), mask=foreground.split()[3])
        return background, poisson_fusion_image, ground_truth
    except Exception as e:
        traceback.print_exc()


def judge_required_image(area=None, f_size=None, b_size=None, min_area=0, max_area=99999, size_threshold=0.5):
    try:
        if area == None or f_size == None or b_size == None:
            return True
        else:
            pass

        if area >= min_area and area <= max_area:
            return True
        else:
            return False

        if b_size[0] > f_size[0] * (1 - size_threshold) and b_size[0] < f_size[0] * (1 + size_threshold) \
                and b_size[1] > f_size[1] * (1 - size_threshold) and b_size[1] < f_size[1] * (1 + size_threshold):
            return True
        else:
            return False

    except Exception as e:
        traceback.print_exc()


def judge_area_similar(foreground_area, background_area, similar_threshold=0.5):
    """
    判断两个区域是否相似，这有利于poisson编辑的效果，这只是固定位置判断，输入的都是两个mask区域

    1.判断直方图
    2. 判断
    :param foreground_area:
    :param background_area:
    :param similar_threshold:有百分之多少的相似 默认 0.5
    :return: 返回一个bool数,表示该选择的预期ok否
    """

    # 直方图
    similar_score = myCalImage.calc_similar(foreground_area, background_area)

    print('直方图相似程度:', similar_score)
    if similar_score > 0.5 * 100:
        return True
    else:
        return False


def paste_object_to_background(foreground, background, mask, bbox, tamper_num=1, optimize=False):
    """
    拿一张前景图 和mask 图 将mask区域paste到background上
    :param foreground:
    :param background:
    :param mask:
    :param tamper_num: return 多少张篡改图——位置不一样 ，默认为1
    :param optimize: 是否使用优化算法，寻找最佳区域
    :return:
    """

    # 把特定区域的object给弄出来`
    orignal_bg = background
    object_area = foreground
    object_area[:, :, 0] = np.array(object_area[:, :, 0] * mask)
    object_area[:, :, 1] = np.array(object_area[:, :, 1] * mask)
    object_area[:, :, 2] = np.array(object_area[:, :, 2] * mask)


    a = mask
    a = np.where(a != 0)
    bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]),np.max(a[1])
    cut_mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    object_area = object_area[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    #
    # plt.figure('获取矩形区域')
    # plt.imshow(object_area)
    # plt.show()
    #

    # 以左上角的点作为参考点，计算可以paste的区域
    background_shape = background.shape
    object_area_shape = cut_mask.shape
    paste_area = [background_shape[0] - object_area_shape[0], background_shape[1] - object_area_shape[1]]
    print('the permit paste area is :', paste_area)
    row = np.random.randint(0, paste_area[0])
    col = np.random.randint(0, paste_area[1])

    # 判断object和bg的大小是否符合要求
    if paste_area[0] < 5 or paste_area[1] < 5:
        print('提醒：允许的粘贴区域太小')

    tamper_image = []
    tamper_poisson = []
    tamper_mask = []
    tamper_gt = []
    if optimize == False:
        for times in range(tamper_num):
            try:
                padding_mask = np.zeros((background_shape[0], background_shape[1]))
                padding_mask[row:row+object_area_shape[0] , col:col+object_area_shape[1] ] = cut_mask

                padding_object_area = np.zeros((background_shape[0], background_shape[1], 3),dtype='uint8')
                padding_object_area[row:row +object_area_shape[0] , col:col + object_area_shape[1] , 0] = object_area[:,:,0]
                padding_object_area[row:row + object_area_shape[0], col:col + object_area_shape[1], 1] = object_area[:,
                                                                                                         :, 1]
                padding_object_area[row:row + object_area_shape[0], col:col + object_area_shape[1], 2] = object_area[:,
                                                                                                         :, 2]

                difference = sum(sum(sum(object_area))) - sum(sum(sum(padding_object_area)))
                print(difference)


                t_mask = np.where(padding_mask == 1, 0, 1)


                background[:,:,0] = background[:,:,0] * t_mask
                background[:, :, 1] = background[:, :, 1] * t_mask
                background[:, :, 2] = background[:, :, 2] * t_mask

                # plt.figure('选出background上的区域')
                # plt.imshow(background)
                # plt.show()

                background = background + padding_object_area



                # plt.figure('贴好的区域')
                # plt.imshow(padding_object_area)
                # plt.show()
                background = Image.fromarray(background, 'RGB')
                tamper_image.append(background)
                tamper_mask.append(padding_mask)
                background = orignal_bg
            except Exception as e:
                traceback.print_exc()
    else:
        # 暂时先空着
        pass

    for index, item in enumerate(tamper_image):

        difference_8 = mask_to_outeedge(tamper_mask[index])
        difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
        difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
        double_edge_candidate = difference_8_dilation + tamper_mask[index]
        double_edge = np.where(double_edge_candidate == 2, 1, 0)
        ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
            tamper_mask[index] == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
        tamper_gt.append(ground_truth)



        try:
            mask = Image.fromarray(tamper_mask[index])
        except Exception as e:
            print('mask to Image error', e)
        # 在这里的时候，mask foreground background 尺寸都是一致的了，poisson融合时，offset置为0
        foreground = Image.fromarray(padding_object_area)

        background = Image.fromarray(orignal_bg)
        mask = padding_mask
        try:
            poisson_foreground = cv2.cvtColor(np.asarray(foreground.convert('RGB')), cv2.COLOR_RGB2BGR)
            poisson_background = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)
            poisson_mask = np.asarray(mask)
            poisson_mask = np.where(poisson_mask == 1, 255, 0)
            poisson_fusion_image = poisson_image_editing.poisson_fusion(poisson_foreground, poisson_background,
                                                                        poisson_mask)
            poisson_fusion_image = Image.fromarray(cv2.cvtColor(poisson_fusion_image, cv2.COLOR_BGR2RGB))
            tamper_poisson.append(poisson_fusion_image)
            return tamper_image, tamper_poisson, tamper_gt

        except Exception as e:
            traceback.print_exc()



def random_area_to_background(background, mask, tamper_num =1):
    """
    输入一张背景图和一张mask图，该mask图的size不应需要与背景图一致
    :param background:
    :param mask: mask任意一个前景图的mask
    :param tamper_num :篡改的数量，默认为1
    :return: 篡改好了的图像和一张GT 是list
    """
    # 找到mask 的矩形区域
    oringal_background = background.copy()


    a = mask
    a = np.where(a != 0 )
    bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]),np.max(a[1])
    cut_mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # 以左上角的点作为参考点，计算可以paste的区域
    background_shape = background.shape
    object_area_shape = cut_mask.shape
    paste_area = [background_shape[0] - object_area_shape[0], background_shape[1] - object_area_shape[1]]
    print('the permit paste area is :', paste_area)
    row1 = np.random.randint(0, paste_area[0])
    col1 = np.random.randint(0, paste_area[1])


    # 在background上获取mask的区域
    temp_background = background.copy()
    cut_area = temp_background[row1:row1+object_area_shape[0],col1:col1+object_area_shape[1],:]
    cut_area[:,:,0] = cut_area[:,:,0] * cut_mask
    cut_area[:, :, 1] = cut_area[:, :, 1] * cut_mask
    cut_area[:, :, 2] = cut_area[:, :, 2] * cut_mask


    for i in range(3):
        row2 = np.random.randint(0, paste_area[0])
        col2 = np.random.randint(0, paste_area[1])
        if abs(row1-row2) + abs(col1-col2) <50:
            print('随机选到的区域太近，重新选择')
        else:
            break


    # 判断object和bg的大小是否符合要求
    if paste_area[0] < 5 or paste_area[1] < 5:
        print('提醒：允许的粘贴区域太小')


    # 随机在background上贴上该mask的区域，并且保证与原区域有一定的像素偏移,然后生成新的mask图

    tamper_image = []
    tamper_mask = []
    tamper_gt = []
    tamper_poisson=[]
    for times in range(tamper_num):
        bk_mask = np.zeros((background_shape[0],background_shape[1]), dtype='uint8')
        bk_area = np.zeros((background_shape[0], background_shape[1],3), dtype='uint8')
        bk_mask[row2:row2+object_area_shape[0], col2:col2+object_area_shape[1]] = cut_mask
        bk_area[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1], :] = cut_area



        background[:,:,0] = background[:,:,0] * np.where(bk_mask==1,0,1)
        background[:, :, 1] = background[:, :, 1] * np.where(bk_mask==1,0,1)
        background[:, :, 2] = background[:, :, 2] * np.where(bk_mask==1,0,1)
        background = background + bk_area

        tamper_image.append(background)
        tamper_mask.append(bk_mask)
    # 调用save_method保存



    for index, item in enumerate(tamper_image):

        difference_8 = mask_to_outeedge(tamper_mask[index])
        difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
        difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
        double_edge_candidate = difference_8_dilation + tamper_mask[index]
        double_edge = np.where(double_edge_candidate == 2, 1, 0)
        ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
            tamper_mask[index] == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
        tamper_gt.append(ground_truth)



        try:
            mask = Image.fromarray(tamper_mask[index])
        except Exception as e:
            print('mask to Image error', e)
        # 在这里的时候，mask foreground background 尺寸都是一致的了，poisson融合时，offset置为0
        foreground = Image.fromarray(bk_area)

        background = Image.fromarray(oringal_background)
        mask = bk_mask
        try:
            poisson_foreground = cv2.cvtColor(np.asarray(foreground.convert('RGB')), cv2.COLOR_RGB2BGR)
            poisson_background = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)
            poisson_mask = np.asarray(mask)
            poisson_mask = np.where(poisson_mask == 1, 255, 0)
            poisson_fusion_image = poisson_image_editing.poisson_fusion(poisson_foreground, poisson_background,
                                                                        poisson_mask)
            poisson_fusion_image = Image.fromarray(cv2.cvtColor(poisson_fusion_image, cv2.COLOR_BGR2RGB))

            #
            # plt.figure('123')
            # plt.imshow(poisson_fusion_image)
            # plt.show()
            tamper_poisson.append(poisson_fusion_image)
            return tamper_image, tamper_poisson, tamper_gt

        except Exception as e:
            traceback.print_exc()

def random_area_to_fix_background(background, mask, tamper_num =2,bk_shape=(320,320)):
    """
       输入一张背景图和一张mask图，该mask图的size不应需要与背景图一致
       :param background:
       :param mask: mask任意一个前景图的mask
       :param tamper_num :篡改的数量，默认为1
       :return: 篡改好了的图像和一张GT 是list
       """
    background = image_crop.crop(background,target_shape=(320,320))
    try:
        if background == 'error':
            return False, False, False
    except Exception as e:
        traceback.print_exc()
        sys.exit()

    # 找到mask 的矩形区域
    oringal_background = background.copy()
    a = mask
    a = np.where(a != 0)
    bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    cut_mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # 以左上角的点作为参考点，计算可以paste的区域
    background_shape = background.shape
    object_area_shape = cut_mask.shape
    paste_area = [background_shape[0] - object_area_shape[0], background_shape[1] - object_area_shape[1]]
    print('the permit paste area is :', paste_area)
    row1 = np.random.randint(0, paste_area[0])
    col1 = np.random.randint(0, paste_area[1])

    # 在background上获取mask的区域
    temp_background = background.copy()
    cut_area = temp_background[row1:row1 + object_area_shape[0], col1:col1 + object_area_shape[1], :]
    cut_area[:, :, 0] = cut_area[:, :, 0] * cut_mask
    cut_area[:, :, 1] = cut_area[:, :, 1] * cut_mask
    cut_area[:, :, 2] = cut_area[:, :, 2] * cut_mask

    for i in range(3):
        row2 = np.random.randint(0, paste_area[0])
        col2 = np.random.randint(0, paste_area[1])
        if abs(row1 - row2) + abs(col1 - col2) < 10:
            print('随机选到的区域太近，最好重新选择')
        else:
            break

    # 判断object和bg的大小是否符合要求
    if paste_area[0] < 5 or paste_area[1] < 5:
        print('提醒：允许的粘贴区域太小')
    if paste_area[0] < 1 or paste_area[1] < 1:
        print('无允许粘贴的区域')
        return False,False,False
    # 随机在background上贴上该mask的区域，并且保证与原区域有一定的像素偏移,然后生成新的mask图

    tamper_image = []
    tamper_mask = []
    tamper_gt = []
    tamper_poisson = []
    for times in range(tamper_num):
        bk_mask = np.zeros((background_shape[0], background_shape[1]), dtype='uint8')
        bk_area = np.zeros((background_shape[0], background_shape[1], 3), dtype='uint8')
        bk_mask[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1]] = cut_mask
        bk_area[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1], :] = cut_area

        background[:, :, 0] = background[:, :, 0] * np.where(bk_mask == 1, 0, 1)
        background[:, :, 1] = background[:, :, 1] * np.where(bk_mask == 1, 0, 1)
        background[:, :, 2] = background[:, :, 2] * np.where(bk_mask == 1, 0, 1)
        background = background + bk_area

        tamper_image.append(background)
        tamper_mask.append(bk_mask)
    # 调用save_method保存

    for index, item in enumerate(tamper_image):

        difference_8 = mask_to_outeedge(tamper_mask[index])
        difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
        difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
        double_edge_candidate = difference_8_dilation + tamper_mask[index]
        double_edge = np.where(double_edge_candidate == 2, 1, 0)
        ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
            tamper_mask[index] == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
        tamper_gt.append(ground_truth)

        try:
            mask = Image.fromarray(tamper_mask[index])
        except Exception as e:
            print('mask to Image error', e)
        # 在这里的时候，mask foreground background 尺寸都是一致的了，poisson融合时，offset置为0
        foreground = Image.fromarray(bk_area)

        background = Image.fromarray(oringal_background)
        mask = bk_mask
        try:
            poisson_foreground = cv2.cvtColor(np.asarray(foreground.convert('RGB')), cv2.COLOR_RGB2BGR)
            poisson_background = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)
            poisson_mask = np.asarray(mask)
            poisson_mask = np.where(poisson_mask == 1, 255, 0)
            poisson_fusion_image = poisson_image_editing.poisson_fusion(poisson_foreground, poisson_background,
                                                                        poisson_mask)
            poisson_fusion_image = Image.fromarray(cv2.cvtColor(poisson_fusion_image, cv2.COLOR_BGR2RGB))

            #
            # plt.figure('123')
            # plt.imshow(poisson_fusion_image)
            # plt.show()
            tamper_poisson.append(poisson_fusion_image)
            return tamper_image, tamper_poisson, tamper_gt

        except Exception as e:
            traceback.print_exc()

def specific_object_to_fix_background(background, mask, tamper_num =2,bk_shape=(320,320)):
    """
       输入一张背景图和一张mask图，该mask图的size不应需要与背景图一致
       :param background:
       :param mask: mask任意一个前景图的mask
       :param tamper_num :篡改的数量，默认为1
       :return: 篡改好了的图像和一张GT 是list
       """
    background = image_crop.crop(background,target_shape=(320,320))
    try:
        if background == 'error':
            return False, False, False
    except Exception as e:
        traceback.print_exc()
        sys.exit()

    # 找到mask 的矩形区域
    oringal_background = background.copy()
    a = mask
    a = np.where(a != 0)
    bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    cut_mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # 以左上角的点作为参考点，计算可以paste的区域
    background_shape = background.shape
    object_area_shape = cut_mask.shape
    paste_area = [background_shape[0] - object_area_shape[0], background_shape[1] - object_area_shape[1]]
    print('the permit paste area is :', paste_area)
    row1 = np.random.randint(0, paste_area[0])
    col1 = np.random.randint(0, paste_area[1])

    # 在background上获取mask的区域
    temp_background = background.copy()
    cut_area = temp_background[row1:row1 + object_area_shape[0], col1:col1 + object_area_shape[1], :]
    cut_area[:, :, 0] = cut_area[:, :, 0] * cut_mask
    cut_area[:, :, 1] = cut_area[:, :, 1] * cut_mask
    cut_area[:, :, 2] = cut_area[:, :, 2] * cut_mask

    for i in range(3):
        row2 = np.random.randint(0, paste_area[0])
        col2 = np.random.randint(0, paste_area[1])
        if abs(row1 - row2) + abs(col1 - col2) < 10:
            print('随机选到的区域太近，最好重新选择')
        else:
            break

    # 判断object和bg的大小是否符合要求
    if paste_area[0] < 5 or paste_area[1] < 5:
        print('提醒：允许的粘贴区域太小')
    if paste_area[0] < 1 or paste_area[1] < 1:
        print('无允许粘贴的区域')
        return False,False,False
    # 随机在background上贴上该mask的区域，并且保证与原区域有一定的像素偏移,然后生成新的mask图

    tamper_image = []
    tamper_mask = []
    tamper_gt = []
    tamper_poisson = []
    for times in range(tamper_num):
        bk_mask = np.zeros((background_shape[0], background_shape[1]), dtype='uint8')
        bk_area = np.zeros((background_shape[0], background_shape[1], 3), dtype='uint8')
        bk_mask[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1]] = cut_mask
        bk_area[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1], :] = cut_area

        background[:, :, 0] = background[:, :, 0] * np.where(bk_mask == 1, 0, 1)
        background[:, :, 1] = background[:, :, 1] * np.where(bk_mask == 1, 0, 1)
        background[:, :, 2] = background[:, :, 2] * np.where(bk_mask == 1, 0, 1)
        background = background + bk_area

        tamper_image.append(background)
        tamper_mask.append(bk_mask)
    # 调用save_method保存

    for index, item in enumerate(tamper_image):

        difference_8 = mask_to_outeedge(tamper_mask[index])
        difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
        difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
        double_edge_candidate = difference_8_dilation + tamper_mask[index]
        double_edge = np.where(double_edge_candidate == 2, 1, 0)
        ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
            tamper_mask[index] == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
        tamper_gt.append(ground_truth)

        try:
            mask = Image.fromarray(tamper_mask[index])
        except Exception as e:
            print('mask to Image error', e)
        # 在这里的时候，mask foreground background 尺寸都是一致的了，poisson融合时，offset置为0
        foreground = Image.fromarray(bk_area)

        background = Image.fromarray(oringal_background)
        mask = bk_mask
        try:
            poisson_foreground = cv2.cvtColor(np.asarray(foreground.convert('RGB')), cv2.COLOR_RGB2BGR)
            poisson_background = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)
            poisson_mask = np.asarray(mask)
            poisson_mask = np.where(poisson_mask == 1, 255, 0)
            poisson_fusion_image = poisson_image_editing.poisson_fusion(poisson_foreground, poisson_background,
                                                                        poisson_mask)
            poisson_fusion_image = Image.fromarray(cv2.cvtColor(poisson_fusion_image, cv2.COLOR_BGR2RGB))

            #
            # plt.figure('123')
            # plt.imshow(poisson_fusion_image)
            # plt.show()
            tamper_poisson.append(poisson_fusion_image)
            return tamper_image, tamper_poisson, tamper_gt

        except Exception as e:
            traceback.print_exc()




def image_save_method(tamper_image, img=None, img1=None, tamper_poisson=None, ground_truth=None, save_path=None,
                      cat=None):
    """

    :param src:
    :param tamper_image:
    :param img:
    :param img1:
    :param tamper_poisson:
    :param ground_truth:
    :param save_path:
    :param cat:
    :return:
    """
    try:
        if save_path == None:
            print('请输入保存root路径')
        if os.path.exists(save_path) == False:
            print('请手动创建数据集root目录')
        tamper_path = 'tamper_result'
        tamper_poisson_path = 'tamper_poisson_result'
        ground_truth_path = 'ground_truth_result'
        if not os.path.exists(os.path.join(save_path,tamper_path)):
            os.makedirs(os.path.join(save_path,tamper_path))
        if not os.path.exists(os.path.join(save_path,tamper_poisson_path)):
            os.makedirs(os.path.join(save_path, tamper_poisson_path))
        if not os.path.exists(os.path.join(save_path,ground_truth_path)):
            os.makedirs(os.path.join(save_path, ground_truth_path))


        # os.makedirs(os.path.join(save_path,src))

        image_format = ['.jpg', '.png', '.bmp']
        tptype = ['Default', 'poisson', 'Gt']
        save_name_part3 = image_format
        save_name_part2 =  '_' + str(img['id']) + '_' + str(img1['id']) + '_' + cat
        save_name_part1 = {'tamper_result': os.path.join(save_path, tamper_path),
                           'tamper_poisson_result': os.path.join(save_path, tamper_poisson_path),
                           'ground_truth_result': os.path.join(save_path, ground_truth_path)}
        save_name = {'tamper_result': os.path.join(save_name_part1['tamper_result'], tptype[0] +save_name_part2+save_name_part3[np.random.randint(0,2)]),
                     'tamper_poisson_result': os.path.join(save_name_part1['tamper_poisson_result'], tptype[1] +save_name_part2+ save_name_part3[np.random.randint(0,2)]),
                     'ground_truth_result': os.path.join(save_name_part1['ground_truth_result'], tptype[2]+save_name_part2+save_name_part3[2])}

        if not os.path.isfile(save_name['tamper_result']):
            print(save_name['tamper_result'])
            if tamper_image is not None:
                if type(tamper_image) == type(Image.Image()):
                    tamper_image.save(save_name['tamper_result'])
                elif type(tamper_image) == 'numpy.ndarray':
                    (Image.fromarray(tamper_image)).save(save_name['tamper_result'])
                else:
                    tamper_image = cv2.cvtColor(np.asarray(tamper_image),cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_name['tamper_result'], tamper_image)
            else:
                traceback.print_exc()
                sys.exit()

        if not os.path.isfile(save_name['tamper_poisson_result']):
            print(save_name['tamper_poisson_result'])
            if tamper_poisson is not None:
                tamper_poisson.save(save_name['tamper_poisson_result'])
                # cv2.imwrite(save_name['tamper_poisson_result'], tamper_poisson)
            else:
                traceback.print_exc()
                sys.exit()

        if not os.path.isfile(save_name['ground_truth_result']):
            print(save_name['ground_truth_result'])
            if ground_truth is not None:
                ground_truth = np.array(ground_truth,dtype='uint8')
                ground_truth = Image.fromarray(ground_truth)
                ground_truth.save(save_name['ground_truth_result'])
            else:
                traceback.print_exc()
                sys.exit()


    except Exception as e:
        traceback.print_exc()
        sys.exit(0)
    return True


def main(cat_range=[1, 80], num_per_cat=100, area_constraint=[1000, 9999], optimize_constraint=False,
         save_root_path=None, dataset_root=None):
    cycle_flag = 0
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    dataset_root = 'D:\\实验室\\图像篡改检测\\数据集\\COCO\\'
    save_root_path = 'D:\\实验室\\1127splicing'
    if dataset_root == None:
        print('输入的数据集为空')
        sys.exit()
    else:
        dataDir = dataset_root


    dataType = 'train2017'
    annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    count_cat = -1
    count_num = 0

    for cat in cats[cat_range[0]:cat_range[1]]:
        count_cat+=1
        for num in range(num_per_cat):
            count_num+=1
            try:
                start_time = time.time()


                catIds = coco.getCatIds(catNms=[cat['name']])
                imgIds = coco.getImgIds(catIds=catIds)
                img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns = coco.loadAnns(annIds)
                img1 = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
                bbx = anns[0]['bbox']

                # 判断随机出来的两幅图像符不符合要求
                if judge_required_image(anns[0]['area'], (img['height'], img['width']), (img1['height'], img1['width']),
                                        5000, size_threshold=0.2):
                    cycle_flag += 1
                    print('循环的次数为:', cycle_flag)
                    if cycle_flag >= 50:
                        print('50 times 循环')
                    elif cycle_flag >= 30:
                        print('30 times 循环')
                    elif cycle_flag > 10:
                        print('10 times 循环')
                    elif cycle_flag > 3:
                        print('循环的次数为：%d', cycle_flag)
                    else:
                        pass
                    pass
                else:
                    continue

                I = io.imread(os.path.join(dataDir, dataType, '{:012d}.jpg'.format(img['id'])))
                b1 = io.imread(os.path.join(dataDir, dataType, '{:012d}.jpg'.format(img1['id'])))
                mask = np.array(coco.annToMask(anns[0]))


                # tamper_raw_image, tamper_poisson_image, ground_truth = splicing_tamper_one_image(I, b1, mask)


                tamper_raw_image, tamper_poisson_image, ground_truth = paste_object_to_background(I,b1,mask,bbx,1,False)
                # tamper_raw_image, tamper_poisson_image, ground_truth = random_area_to_background(b1,mask,1)
                # tamper_raw_image, tamper_poisson_image, ground_truth = random_area_to_fix_background(b1, mask, 1)
                # plt.figure('tamper_raw_image')
                # plt.imshow(tamper_raw_image[0])
                # plt.show()
                #
                # plt.figure('tamper_poisson_image')
                # plt.imshow(tamper_poisson_image[0])
                # plt.show()
                #
                # plt.figure('gt')
                # plt.imshow(ground_truth[0])
                # plt.show()
                if tamper_raw_image == False:
                    print('123123123123')
                    continue
                cycle_flag = 0
                # src, tamper_image, img = None, img1 = None, tamper_poisson = None, ground_truth = None, save_path = None, foreground_name = None, background_name = None, cat = None)
                image_save_method(tamper_image=tamper_raw_image[0], tamper_poisson=tamper_poisson_image[0],
                                  ground_truth=ground_truth[0], img=img, img1=img1, save_path=save_root_path,
                                  cat=cat['name'])
                end_time = time.time()
                co_time = end_time - start_time
                print('count_cat:%d' % count_cat, '+++++++', 'count_num:%d' % count_num,'======','time:%.2f'%co_time)
            except Exception as e:
                print(e)
    print('finished')

#
# class TamperDataGen:
#     def __init__(self):
#         pass
#

if __name__ == '__main__':
    main()

