"""
Created by haoran
time: 10/23
description:
1. convert channel
"""
import numpy as np
from PIL import Image
import os
import traceback
import time
class ChannelConvert():
    def __init__(self):
        self.save_file = time.asctime()
        pass
    def batch_3dim_to_1dim(self,root_dir):
        """
        对于GT， 将三个通道的图转化为1个通道的，并保存为bmp格式
        :return:
        """
        img_list = os.listdir(root_dir)
        img_list = [os.path.join(root_dir,item) for item in img_list]
        # 创建保存文件夹
        if '/' in root_dir:
            save_file = root_dir.split('/')[-1]
        else:
            save_file = root_dir.split('\\')[-1]
        new_save_file = save_file + '_' + self.save_file
        if os.path.exists(root_dir.replace(save_file,new_save_file)):
            print('文件已存在')
        else:
            print('文件夹不存在创建中')
            os.mkdir(root_dir.replace(save_file,new_save_file))

        for index,item in enumerate(img_list):
            try:
                img = Image.open(item)
                img = np.array(img)
                img = img[:,:,0]
                img = Image.fromarray(img)
                new_save_file = save_file + '_' + self.save_file
                img.save(item.replace(save_file,new_save_file).replace('.png','.bmp').replace('jpg','.bmp'))
            except:
                traceback.print_exc()
            print('\r', 'The process is : %d / %d'%(index,len(img_list)),end='')
