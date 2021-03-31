"""
created by haoran 12/24
"""
import shutil
import os
import sys
import traceback
class CombineTextureData:
    def __init__(self):
        self.src_dir = r'H:\texture_filler\texture_11-28'
        self.target_dir = r'H:\texture_filler\texture_for_tamper_task'
        self.__path_issue()
        self.single_dir_choose_image(self.src_dir, self.target_dir)
    def __path_issue(self):

        if not os.path.exists(self.src_dir):
            print('输入的src_dir 不存在，请重新选择')
            exit(0)
        else:
            pass



        if not os.path.exists(self.target_dir):
            print('target path 不存在,开始创建')
            os.mkdir(self.target_dir)
        else:
            print('target path 已存在，请重新选择')


    def single_dir_choose_image(self, in_dir,out_dir):
        file_list = os.listdir(in_dir)
        is_image = []
        for idx,item in enumerate(file_list):
            if '.jpg' in item or '.png' in item or '.bmp' in item:
                is_image.append(item)
            else:
                pass

        print('the number of image is :  ',len(is_image))

        # 开始搬运
        for idx,item in enumerate(is_image):
            print(idx,'/',len(is_image))
            src = os.path.join(in_dir, item)
            dst = os.path.join(out_dir, item)
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(e)
                exit(0)

    def double_dir_choose_image(self,in_dir, out_dir):
        chirld_dir = []
        for idx, item in enumerate(in_dir):
            if os.path.isdir(item):
                chirld_dir.append(item)
            else:
                pass
        print('The path of ')
if __name__ == '__main__':
    CombineTextureData()