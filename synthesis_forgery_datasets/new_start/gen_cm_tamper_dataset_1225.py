"""
created by haoran
time : 12/25
description:
我在生成纹理错位数据的时候，发现生成方式与之前生成的cm数据很类似
所以，这个文件首先将之前cm的生成方法封装成一个类
继承这个类去完成错位纹理数据的生成
"""
import numpy as np
import cv2
from PIL import Image
import sys
from get_double_edge import mask_to_outeedge
import poisson_image_editing
import skimage.morphology as dilation
import traceback
import image_crop,os
import matplotlib.pyplot as plt
import random

from rich.progress import track
class CopyMove:
    """
    这个类是解决cm问题的类
    """

    def __init__(self):
        self.rotation_flag = True
        pass

    def random_area_to_fix_background(self, background, mask, tamper_num=1, bk_shape=(320, 320)):
        """
           输入一张背景图和一张mask图，该mask图的size不应需要与背景图一致

           :param background:numpy
           :param mask: mask任意一个前景图的mask,numpy,这里的mask 是 01 mask
           :param tamper_num :篡改的数量，默认为1
           :return: 篡改好了的图像和一张GT 是list
           """
        for gen_time in range(5):
            background = image_crop.crop(background, target_shape=(320, 320))
            origin = background.copy()
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
            if paste_area[0] < 3 or paste_area[1] < 3:
                print('提醒：允许的粘贴区域太小')
            if paste_area[0] < 1 or paste_area[1] < 1:
                print('无允许粘贴的区域')
                if gen_time <3:
                    continue
                else:
                    return False, False
            # 随机在background上贴上该mask的区域，并且保证与原区域有一定的像素偏移,然后生成新的mask图

            tamper_image = []
            tamper_mask = []
            tamper_gt = []
            for times in range(tamper_num):
                bk_mask = np.zeros((background_shape[0], background_shape[1]), dtype='uint8')
                bk_area = np.zeros((background_shape[0], background_shape[1], 3), dtype='uint8')
                bk_mask[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1]] = cut_mask
                bk_area[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1], :] = cut_area

                if self.rotation_flag:
                    try:
                        cut_area, cut_mask,rotaion_flag = self.__random_rotation(cut_area, cut_mask)
                    except Exception as e:
                        print(e)

                background[:, :, 0] = background[:, :, 0] * np.where(bk_mask == 1, 0, 1)
                background[:, :, 1] = background[:, :, 1] * np.where(bk_mask == 1, 0, 1)
                background[:, :, 2] = background[:, :, 2] * np.where(bk_mask == 1, 0, 1)
                background = background + bk_area

                tamper_image.append(background)
                tamper_mask.append(bk_mask)

            periodic_flag = False
            for _t_img in tamper_image:
                if self.__no_periodic_check(origin, _t_img):
                    periodic_flag = True
                    break
                else:
                    continue

            if periodic_flag:
                pass
            else:
                continue

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
                    return tamper_image, tamper_gt, (row1,col1,row2,col2,rotaion_flag)

                except Exception as e:
                    traceback.print_exc()

    def __random_rotation(self, src, gt):
        """
        为了让src于 gt不出错， 只实现90°的旋转
        :param src:输入是numpy
        :param gt:输入是numpy
        :return:
        """
        rotation_src = src
        rotation_gt = gt
        rotation_flag = np.random.randint(0, 3)
        # plt.imshow(gt)
        # plt.show()
        if rotation_flag == 0:

            pass
        elif rotation_flag == 1:
            rotation_src = np.rot90(rotation_src)
            rotation_gt = np.rot90(rotation_gt)
        elif rotation_flag == 2:
            rotation_src = np.rot90(rotation_src)
            rotation_gt = np.rot90(rotation_gt)

            rotation_src = np.rot90(rotation_src)
            rotation_gt = np.rot90(rotation_gt)
        elif rotation_flag == 3:
            rotation_src = np.rot90(rotation_src)
            rotation_gt = np.rot90(rotation_gt)

            rotation_src = np.rot90(rotation_src)
            rotation_gt = np.rot90(rotation_gt)

            rotation_src = np.rot90(rotation_src)
            rotation_gt = np.rot90(rotation_gt)
        else:
            pass
        # plt.imshow(rotation_gt)
        # plt.show()
        return rotation_src, rotation_gt, rotation_flag

    def __no_periodic_check(self, origin, after_tamper):
        try:
            difference = abs(origin - after_tamper)
            difference = np.sum(difference)
            print(difference)
            if difference < 10:
                return False
            else:
                return True
        except Exception as e:
            pass

class NoPeriodicTextureData(CopyMove):
    def __init__(self):
        self.src_dir = r'D:\实验室\stex-1024'
        self.gt_dir = r'H:\8_20_dataset_after_divide\test_gt_train_percent_0.80@8_20'
        self.save_dir = r'D:\Image_Tamper_Project\Lab_project_code\TempWorkShop\no_periodic_texture_dataset0109'
        self.save_src_dir = r'D:\Image_Tamper_Project\Lab_project_code\TempWorkShop\no_periodic_texture_dataset0109\train_src'
        self.save_gt_dir = r'D:\Image_Tamper_Project\Lab_project_code\TempWorkShop\no_periodic_texture_dataset0109\train_gt'
        self.rotation_flag = True
        self.__path_issues()
        super(NoPeriodicTextureData).__init__()

    def __my_gt_to_01_mask(self,mask):
        return np.where((mask==50) | (mask == 255),1,0)

    def __path_issues(self):
        save_dir = self.save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, 'train_src'))
            os.mkdir(os.path.join(save_dir, 'train_gt'))
        else:
            print('the save path is exists')
            exit(0)

    def gen_data(self,background=None, mask=None):
        src_list = os.listdir(self.src_dir)
        mask_list = os.listdir(self.gt_dir)
        for idx, item in enumerate(track(src_list)):
            print(idx,'/',len(src_list))
            src_path = os.path.join(self.src_dir,item)
            random_gt_list = []
            for _i in range(10):
                random_gt = random.sample(mask_list, 1)[0]
                if random_gt in random_gt_list:
                    continue
                random_gt_list.append(random_gt)
                gt_path = os.path.join(self.gt_dir,str(random_gt))
                try:
                    src = Image.open(src_path)
                    src = np.array(src, dtype='uint8')

                    mask = Image.open(gt_path)
                    if len(mask.split()) == 3:
                        mask = mask.split()[0]
                    mask = np.array(mask, dtype='uint8')
                    mask = self.__my_gt_to_01_mask(mask)
                except:
                    traceback.print_exc('when reading image %s error'%item)
                    continue

                try:
                    src, gt, row_and_col = self.random_area_to_fix_background(src, mask)
                except:
                    continue
                src = src[0]
                gt = gt[0]
                src = Image.fromarray(src)
                gt = Image.fromarray(gt).convert('RGB')
                src_name = item.split('.')[0] + '('+random_gt.split('.')[0] + ')(%d_%d_%d_%d_%d)' % row_and_col + '.png'
                src_save_path = os.path.join(self.save_src_dir,src_name)
                gt_save_path = os.path.join(self.save_gt_dir,src_name)

                try:
                    src.save(src_save_path.replace('.bmp', '.png'))
                    src.save(src_save_path.replace('.bmp', '.jpg').replace('.png', '.jpg'))
                    gt.save(gt_save_path.replace('.png', '.bmp').replace('.jpg', '.bmp'))
                except:
                    continue
if __name__ == '__main__':

    NoPeriodicTextureData().gen_data()