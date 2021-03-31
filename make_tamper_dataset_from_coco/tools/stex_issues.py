"""
created by haoran
time : 2021/1/8
description:
1. 找到了一个新的纹理数据集stex
2. 但是这个数据集是pnm格式，首先要将其转化为常用格式
"""
import os
from PIL import Image
from rich.progress import track
import rich

class Stex:
    def __init__(self):
        self.image_dir = r'D:\实验室\stex-1024'
        self.convert_pnm2bmp()
        pass
    def convert_pnm2bmp(self):
        """
        突然发现不用转化，直接用就完事儿
        :return:
        """
        image_list = os.listdir(self.image_dir)
        for idx, item in enumerate(track(image_list)):
            img = Image.open(os.path.join(self.image_dir, item))
            print(img.size)

if __name__ == '__main__':
    Stex()