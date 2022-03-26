"""
created by haoran
time:8-17
"""
import os
import sys
import shutil
if __name__ == '__main__':
    large_dataset_path = r'H:\8_20_dataset_after_divide'
    target_dataset_path = r'H:\少量调试数据2'
    src = 'train_dataset_train_percent_0.80@8_20'
    # poisson = 'tamper_poisson_result'
    gt = 'train_gt_train_percent_0.80@8_20'

    if os.path.exists(target_dataset_path) == False:
        print('No target path')
        sys.exit()
    if os.path.exists(large_dataset_path) ==False:
        print('No large dataset path')
        sys.exit()
    num = eval(input('输入你想获得的数量:'))
    src_path = os.path.join(large_dataset_path,src)
    # poisson_path = os.path.join(large_dataset_path,poisson)
    gt_path = os.path.join(large_dataset_path,gt)

    src_list = os.listdir(src_path)
    # poisson_list = os.listdir(poisson)
    # gt_list = os.listdir(gt)
    for index,src_name in enumerate(src_list):
        if index == num:
            break
        poisson_name = src_name.replace('Default','poisson')
        gt_name = src_name.replace('Default','Gt')
        gt_name = gt_name.replace('.jpg','.bmp')
        gt_name = gt_name.replace('.png', '.bmp')
        src_path_file = os.path.join(src_path, src_name)
        # poisson_path_file = os.path.join(poisson_path, poisson_name)
        gt_path_file = os.path.join(gt_path, gt_name)

        # 开始搬运
        shutil.copy(src_path_file, os.path.join(os.path.join(target_dataset_path,src),src_name))
        # shutil.copy(poisson_path_file, os.path.join(os.path.join(target_dataset_path,poisson), poisson_name))
        shutil.copy(gt_path_file, os.path.join(os.path.join(target_dataset_path,gt), gt_name))
        print('%d/%d'%(index,num))