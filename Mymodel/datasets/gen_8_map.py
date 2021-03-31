"""
用来生成8张图
created by haoran
time :2020-7-31
usage:
import gen_8_map.gen_8_2_map as gen_8
relation_8_map = gen_8(mask)
relation_8_map 是个list 每一个元素 2通道的numpy数组，顺时针方向对应8张关系图，
1通道：边缘关系 = 1 否则 0
2通道：同类边缘 =1 否则 0
"""
from PIL import Image
import numpy as np
import cv2 as cv
import traceback
import matplotlib.pyplot as plt
import os
import sys


def gen_8_2_map(mask, mask_area = 50, mask_edge = 255, not_mask_edge = 100, not_mask_area = 0):
    """
    输入mask，先按照固定参数标好，篡改区域、篡改区域边缘，非篡改区域边缘，非篡改区域
    :param mask:
    :return: 从左上角的点开始按照顺时针方向的8张二通道的图
    """
    relation_8_map = []
    edge_loc_=[1,1]
    # 找到内侧和外侧边缘
    mask_pad = np.pad(mask,(1,1),mode='constant')
    mask_pad = np.where(mask_pad == 50,0,mask_pad)
    edge_loc = np.where(mask_pad == mask_edge)
    edge_loc_1 = np.where(mask_pad == not_mask_edge)
    edge_loc_[0] = np.append(edge_loc[0],edge_loc_1[0])
    edge_loc_[1] = np.append(edge_loc[1], edge_loc_1[1])

    del  edge_loc_1
    del edge_loc
    edge_loc = edge_loc_
    mask_shape = mask_pad.shape
    # 生成8张结果图
    for i in range(8):
        temp = np.ones((mask_shape[0],mask_shape[1],2))
        relation_8_map.append(temp)

    for j in range(len(edge_loc[0])):
        row = edge_loc[0][j]
        col = edge_loc[1][j]
        if mask_pad[row - 1, col - 1] != 0:
            relation_8_map[0][row,col,0]=1
            if mask_pad[row, col] == mask_pad[row - 1, col - 1]:
                relation_8_map[0][row, col, 1] = 1
            else:
                relation_8_map[0][row, col, 1] = 0
        else:
            relation_8_map[0][row, col, 0] = 0
            relation_8_map[0][row, col, 1] = 0

        if mask_pad[row - 1, col] != 0:
            relation_8_map[1][row,col,0]=1
            if mask_pad[row, col] == mask_pad[row - 1, col]:
                relation_8_map[1][row, col, 1] = 1
            else:
                relation_8_map[1][row, col, 1] = 0
        else:
            relation_8_map[1][row, col, 0] = 0
            relation_8_map[1][row, col, 1] = 0

        if mask_pad[row - 1, col + 1] != 0:
            relation_8_map[2][row,col,0]=1
            if mask_pad[row, col] == mask_pad[row - 1, col + 1]:
                relation_8_map[2][row, col, 1] = 1
            else:
                relation_8_map[2][row, col, 1] = 0
        else:
            relation_8_map[2][row, col, 0] = 0
            relation_8_map[2][row, col, 1] = 0

        if mask_pad[row , col + 1] != 0:
            relation_8_map[3][row,col,0]=1
            if mask_pad[row, col] == mask_pad[row , col + 1]:
                relation_8_map[3][row, col, 1] = 1
            else:
                relation_8_map[3][row, col, 1] = 0
        else:
            relation_8_map[3][row, col, 0] = 0
            relation_8_map[3][row, col, 1] = 0

        if mask_pad[row + 1, col + 1] != 0:
            relation_8_map[4][row,col,0]=1
            if mask_pad[row, col] == mask_pad[row + 1, col + 1]:
                relation_8_map[4][row, col, 1] = 1
            else:
                relation_8_map[4][row, col, 1] = 0
        else:
            relation_8_map[4][row, col, 0] = 0
            relation_8_map[4][row, col, 1] = 0

        if mask_pad[row + 1, col] != 0:
            relation_8_map[5][row,col,0]=1
            if mask_pad[row, col] == mask_pad[row + 1, col]:
                relation_8_map[5][row, col, 1] = 1
            else:
                relation_8_map[5][row, col, 1] = 0
        else:
            relation_8_map[5][row, col, 0] = 0
            relation_8_map[5][row, col, 1] = 0

        if mask_pad[row + 1, col - 1] != 0:
            relation_8_map[6][row,col,0]=1
            if mask_pad[row, col] == mask_pad[row + 1, col - 1]:
                relation_8_map[6][row, col, 1] = 1
            else:
                relation_8_map[6][row, col, 1] = 0
        else:
            relation_8_map[6][row, col, 0] = 0
            relation_8_map[6][row, col, 1] = 0

        if mask_pad[row, col -1] != 0:
            relation_8_map[7][row,col,0]=1
            if mask_pad[row, col] == mask_pad[row, col - 1]:
                relation_8_map[7][row, col, 1] = 1
            else:
                relation_8_map[7][row, col, 1] = 0
        else:
            relation_8_map[7][row, col, 0] = 0
            relation_8_map[7][row, col, 1] = 0

    for i in range(8):
        relation_8_map[i] = relation_8_map[i][1:-1,1:-1,:]
        # plt.figure('123')
        # plt.imshow(relation_8_map[i][:,:,0])
        # plt.savefig('C:\\Users\\musk\\Desktop\\边缘8张图关系图\\%d.png'%i)
        # plt.show()
        # # temp = Image.fromarray(relation_8_map[i])
        # # temp = temp.convert('RGB')
        # # temp.save('C:\\Users\\musk\\Desktop\\边缘8张图关系图\\%d.png'%i)


    return relation_8_map

if __name__ == '__main__':
    img = plt.imread('1.jpg')
    gen_8_2_map(img)
    print()