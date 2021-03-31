"""
created by haoran
time:8-26
合并CM SP COD10K训练数据
"""
import os
import sys
import numpy as np
import shutil
import random
class CombineDataset():
    def __init__(self):
        pass
    def using_tamper_name_gen_gt_name(self,data_name_list,data_type):
        """
        从tamper image名字到GT名字的一个映射
        :param data_name_list:
        :param data_type:
        :return:
        """
        gt_name_list = []
        if data_type =='cm':
            for name in data_name_list:
                gt_name_list.append(name.replace('Default','CM_Gt').replace('poisson','CM_Gt').replace('png','bmp').replace('jpg','bmp'))
            pass
        elif data_type == 'sp':
            for name in data_name_list:
                gt_name_list.append(name.replace('Default','Gt').replace('poisson','Gt').replace('png','bmp').replace('jpg','bmp'))
            pass
        elif data_type == 'cod10k':
            for index,name in enumerate(data_name_list):
                print(name.replace('tamper','Gt').replace('png','bmp').replace('jpg','bmp'))
                gt_name_list.append(name.replace('tamper','Gt').replace('png','bmp').replace('jpg','bmp'))
            pass
        return gt_name_list

    def random_gen_specific_num_data(self,data_pair,num, data_type):
        src_list = os.listdir(data_pair[0])
        gt_list = os.listdir(data_pair[1])
        if num > len(src_list):
            print('超出最大范围,委而求其全')
            num = len(src_list)
        else:
            pass
        train_data_list = random.sample(src_list, num)
        train_gt_list = self.using_tamper_name_gen_gt_name(train_data_list, data_type=data_type)
        for i in range(train_data_list):

            train_data_list[i] = os.path.join(data_pair[0],train_data_list[i])
            train_gt_list[i] = os.path.join(data_pair[1],train_gt_list[i])

            # 判断gt中的路径是否存在
            if os.path.exists(train_gt_list[i]) == False:
                print('gt:',train_gt_list[i],'不存在')
                sys.exit()

        return train_data_list, train_gt_list
    def move_data(self,src_path_list, target_path_list):
        for index, item in enumerate(src_path_list):
            print('move process:', '%d/%d'%(index,len(src_path_list)))
            shutil.copy(src_path_list,target_path_list)

    def combine_cm_sp_cod10k(self,cm_dir,sp_dir,cod10k_dir,save_dir,base_num='average'):
        """
        :param cm_dir:
        :param sp_dir:
        :param cod10k_dir:
        :param save_dir:
        :param base_num:
        :return:
        """
        # 路径检查
        if os.path.exists(cm_dir):
            print(cm_dir,'路径存在')
            cm_list = os.listdir(cm_dir)
            if len(cm_list) == 0:
                print(cm_dir,'内无数据')
                sys.exit()
            else:
                print('copy-move数据量:',len(cm_list),'张')

        if os.path.exists(sp_dir):
            print(sp_dir,'路径存在')
            sp_list = os.listdir(sp_dir)
            if len(sp_dir) == 0:
                print(sp_dir,'内无数据')
                sys.exit()
            else:
                print('splicing数据量:', len(sp_list),'张')

        if os.path.exists(cod10k_dir):
            print(cod10k_dir,'路径存在')
            cod10k_list = os.listdir(cod10k_dir)
            if len(cod10k_list) == 0:
                print(cod10k_dir,'内无数据')
                sys.exit()
            else:
                print('splicing数据量:', len(cod10k_list),'张')

        if os.path.exists(save_dir):
            print('保存路径',save_dir,'已经存在，请重新确认')
            sys.exit()
        else:
            print('保存路径不存在，准备创建')
            os.mkdir(save_dir)
            train_save_dir = os.path.join(save_dir,'train_cm_sp_cod10k')
            train_gt_save_dir = os.path.join(save_dir, 'gt_cm_sp_cod10k')
            os.mkdir(train_save_dir)
            os.mkdir(train_gt_save_dir)
            print('保存路径创建成功')

        train_cm_list, train_cm_gt_list = self.random_gen_specific_num_data(data_pair=cm_dir, num=5000, data_type='average')
        train_sp_list, train_sp_gt_list = self.random_gen_specific_num_data(data_pair=sp_dir, num=5000,data_type='average')
        train_cod10k_list, train_cod10k_gt_list = self.random_gen_specific_num_data(data_pair=cod10k_dir, num=5000, data_type='average')

        data_move_list = [train_cm_list,train_cm_gt_list,train_sp_list,train_sp_gt_list,train_cod10k_list,train_cod10k_gt_list]
        for data_list in data_move_list:
            if 'gt' in data_list:
                self.move_data(data_list,os.path.join(train_gt_save_dir,data_list.split('\\')[-1]))
            else:
                self.move_data(data_list, os.path.join(train_save_dir,data_list.split('\\')[-1]))

    def cod10k_rename(self,tamper_path,src_path,target_dir='H:\\COD10K_resize\\test\save3\\rename'):

        for index,name in enumerate(os.listdir(tamper_path)):
            gt_name = name.replace('tamper','tamper_gt').replace('png','bmp')
            os.renames(os.path.join(tamper_path_,name), os.path.join(target_dir+'\\tamper','COD10K_tamper_%d.png'%index))
            os.renames(os.path.join(src_path_, gt_name), os.path.join(target_dir +'\\gt', 'COD10K_Gt_%d.bmp' % index))




if __name__ == '__main__':
    cm_dir = ('', '')
    sp_dir = ('', '')
    cod10k_dir = ('','')
    # rename 测试
    # tamper_path_ = 'H:\\COD10K_resize\\test\\save3\\tamper_result_320_cod10k'
    # src_path_ = 'H:\\COD10K_resize\\test\\save3\\ground_truth_result_320_cod10k'
    # CombineDataset().cod10k_rename(tamper_path_,src_path_)
