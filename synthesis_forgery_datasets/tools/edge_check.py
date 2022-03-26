"""
created by haoran
time:8-24
检查边缘,mask图
"""
import numpy as np
from PIL import Image
import os,sys
import traceback
import matplotlib.pyplot as plt

class EdgeCheck():
    def __init__(self,img_path,mask_path,edge_path=None,save_path=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.edge_path = edge_path
        self.save_path = save_path

        pass
    def required_condition(self,area_filter = False):

        pass
    def gen_src_mask(self):
        src = Image.open(self.img_path)
        mask = Image.open(self.mask_path)

        print('src and mask size',src.size,mask.size)
        mask3 = Image.new(mode='RGB', size=(src.size))
        mask = np.array(mask)
        mask = mask[:,:,0]
        mask3 = np.array(mask3)
        mask3[:, :, 0] = mask
        mask3[:, :, 1] = mask
        mask3[:, :, 2] = mask
        mask3 = Image.fromarray(mask3)
        mask_src = Image.blend(src, mask3, .7)
        # plt.figure('mask_src')
        # plt.imshow(mask_src)
        # plt.show()
        mask_src = np.array(mask_src,dtype='uint8')
        mask_src = Image.fromarray(mask_src).convert('RGB')
        mask_src.save(self.save_path)
        pass
    def gen_src_edge(self):
        pass
    def mask_to_double_edge(self):
        pass

if __name__ == '__main__':
    img_path = 'H:\\COD10K_resize\\src\\camourflage_00011.png'
    mask_path = 'H:\\COD10K_resize\\GT\\camourflage_00011.bmp'
    img_root_path = 'H:\\COD10K_resize\\src'
    mask_root_path = 'H:\\COD10K_resize\\GT'
    save_root_path = 'H:\\mask_check_blend'
    if os.path.exists(save_root_path):
        print('文件夹已经存在')
    else:
        os.mkdir(save_root_path)
        print('创建成功')

    img_list = os.listdir(img_root_path)
    mask_list = []
    edge_list = []
    for img in img_list:
        mask_list.append(img.replace('png','bmp'))

    for index,i in enumerate(img_list):
        img_path = os.path.join(img_root_path,i)
        mask_path = os.path.join(mask_root_path,i.replace('png','bmp'))
        save_path = os.path.join(save_root_path,'blend'+i)
        edgecheck = EdgeCheck(img_path,mask_path,save_path=save_path).gen_src_mask()
        print('%d/%d'%(index,len(img_list)))