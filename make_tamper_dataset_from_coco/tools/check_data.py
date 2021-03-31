"""
created by haoran
time : 2021-01-9
description:
1.确保输入的数据不会报错
"""
import os
import sys
from PIL import Image
import numpy as np
from rich.progress import track
import rich
class CheckData:
    def __init__(self):
        self.data_dir = r'D:\Image_Tamper_Project\Lab_project_code\TempWorkShop\no_periodic_texture_dataset0109\train_src'
        self.enter_gate()
    def enter_gate(self):
        error_list = []
        src_list = os.listdir(self.data_dir)
        for idx,item in enumerate(track(src_list)):

            try:
                src = Image.open(os.path.join(self.data_dir,item))
                flag_size = self.check_size(src)
                flag_dim = self.check_dim(src)
                if flag_size and flag_dim:
                    pass
                else:
                    error_list.append(item)
            except Exception as e:
                error_list.append(item)


    def check_readable(self):

        pass
    def check_dim(self, src, dim_required = 3):
        if len(src.split()) != dim_required:
            return False
        else:
            return True
    def check_size(self,src):

        if src.size !=(320,320):
            return False
        else:
            return True


if __name__ == '__main__':
    CheckData()