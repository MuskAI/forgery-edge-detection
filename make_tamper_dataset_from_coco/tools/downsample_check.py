"""
created by haoran
time:8-24
检查各种下采样方法
"""
import numpy as np
from PIL import Image
import os,sys
import traceback
import matplotlib.pyplot as plt
class DownSample():
    def __init__(self):
        self.image_path = ''
        self.save_path = ''

        pass
    def numpy_direct_resize(self):
        pass

    def Image_BiCubic_resize(self,target_size=(320,320), Binary_01 = False):
        img = Image.open(self.image_path)

        # 首先将GT转化为无类别图
        if Binary_01 == True:
            dou_em = np.array(img)
            dou_em = np.where(dou_em == 50, 0, dou_em)
            dou_em = np.where(dou_em == 100, 1, dou_em)
            dou_em = np.where(dou_em == 255, 1, dou_em)
            dou_em = np.array(dou_em,dtype='float32')

            plt.figure(1)
            plt.imshow(dou_em)
            plt.show()

            img = Image.fromarray(dou_em)

        if self.required_condition(img.size,target_size) ==False:
            return False

        img_downsample = img.resize(target_size, Image.BICUBIC)
        if Binary_01 == False:

            plt.figure(1)
            plt.imshow(img_downsample)
            plt.show()

            img_downsample.save(self.save_path)
        else:
            img_downsample = np.array(img_downsample,dtype='float32')
            img_downsample = np.where(img_downsample>0,255,0)
            img_downsample = np.array(img_downsample,dtype='uint8')
            img_downsample = Image.fromarray(img_downsample).convert('RGB')
            img_downsample.save(self.save_path)
        print('双三次插值，保存在:',self.save_path)
        return True
    def tamper_gt_downsample(self, image_path, save_path,down_sample_type,target_size):
        self.image_path = image_path
        self.save_path = save_path
        if down_sample_type == 'BICUBIC':
            self.Image_BiCubic_resize(Binary_01=True,target_size=target_size)
    def required_condition(self,input_size,target_size):
        """
        判断尺寸大小合不合适
        :param input_size:
        :param target_size:
        :return:
        """
        if input_size[0]>=target_size[0] and input_size[1] >= target_size[1]:
            return True
        else:
            return False


if __name__ == '__main__':
    img_path = r'C:\Users\musk\Desktop\fix_bk\ground_truth_result\Gt_13659_56288_tv.bmp'
    save_path = r'C:\Users\musk\Desktop\bicubic\Gt_13659_56288_tv_160.bmp'
    DownSample().tamper_gt_downsample(img_path,save_path,'BICUBIC',target_size=(160,160))




    # # image_path = 'H:\\TrainDataset\\GT\\camourflage_00328.png'
    # # save_path = './11.bmp'
    # img_root_path = 'H:\\TrainDataset\\GT'
    # save_root_path = 'H:\\COD10K_resize\\GT'
    # if os.path.exists(img_root_path):
    #     print(img_root_path,' 已经存在')
    # else:
    #     print('数据集文件夹不存在，请检查')
    #     sys.exit()
    #
    # if os.path.exists(save_root_path):
    #     print(save_root_path, ' 已经存在')
    # else:
    #     os.mkdir(save_root_path)
    #
    # size_error = []
    # for index,img in enumerate(os.listdir(img_root_path)):
    #     img_path = os.path.join(img_root_path,img)
    #
    #     save_path = os.path.join(save_root_path,img.replace('png','bmp').replace('jpg','bmp'))
    #     down = DownSample(img_path, save_path).Image_BiCubic_resize(Binary_01=True)
    #     if down == False:
    #         size_error.append(img)
    #         print(img,'大小不符合要求')
    #     print('the process:%d/%d'%(index,len(os.listdir(img_root_path))))
    # print('%d张'%len(size_error),'不符合要求')
    # print(size_error)








