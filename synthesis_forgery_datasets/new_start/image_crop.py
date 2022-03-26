"""
created by haoran
time:2020-8-4
输入一张图片(numpy array),随机crop到指定大小
"""
import numpy as np
import traceback
import sys
from PIL import Image

import matplotlib.pyplot as plt
def crop(img, target_shape=(320,320)):
    img_shape = img.shape
    height = img_shape[0]
    width = img_shape[1]
    if (height,width) == target_shape:
        return img
    else:
        random_height_range = height - target_shape[0]
        random_width_range = width - target_shape[1]

        if random_width_range <0 or random_height_range<0:
            print('臣妾暂时还做不到!!!')
            traceback.print_exc()
            return 'error'

        if random_height_range == 0:
            random_height =0
        else:
            random_height = np.random.randint(0,random_height_range)

        if random_width_range == 0:
            random_width =0
        else:
            random_width = np.random.randint(0, random_width_range)

        return img[random_height:random_height+target_shape[0],random_width:random_width+target_shape[1]]


if __name__ == '__main__':
    img = Image.open('2.png')
    plt.figure('src')
    plt.imshow(img)
    plt.show()
    img = np.array(img)
    img = crop(img)
    plt.figure('crop')
    plt.imshow(img)
    plt.show()
