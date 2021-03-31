"""
created by hoaran
time : 2020-8-9
将数据集按照比例划分为训练集和测试集
"""
import random
import os
import sys
import datetime
import shutil
import traceback
DATASET_SRC_PATH = r'G:\3月最新数据\periodic_texture\train_src'
DATASET_GT_PATH = r'G:\3月最新数据\periodic_texture\train_gt'
DATASET_TARGET_PATH = r'G:\3月最新数据\periodic_texture\divide'


class DataDivide:
    def __init__(self):
        # 相关数据集的路径
        src_path_list = [
            # '/media/liu/File/8_26_Sp_dataset_after_divide/train_dataset_train_percent_0.80@8_26',
            '/media/liu/File/10月数据准备/10月12日实验数据/negative/src',
            '/media/liu/File/Sp_320_dataset/tamper_result_320',
            # '/media/liu/File/10月数据准备/10月12日实验数据/casia/src'
            '/media/liu/File/11月数据准备/CASIA2.0_DATA_FOR_TRAIN/src',
            '/media/liu/File/11月数据准备/CASIA_TEMPLATE_TRAIN/src'
        ]
        gt_path_list = []
        self.src_path_list = src_path_list

        # self.sp_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/splicing/ground_truth_result_320'
        self.sp_gt_path = '/media/liu/File/Sp_320_dataset/ground_truth_result_320'
        self.cm_gt_path = '/media/liu/File/8_26_Sp_dataset_after_divide/test_dataset_train_percent_0.80@8_26'
        self.negative_gt_path = '/media/liu/File/10月数据准备/10月12日实验数据/negative/gt'
        self.casia_gt_path = '/media/liu/File/11月数据准备/CASIA2.0_DATA_FOR_TRAIN/gt'
        self.template_gt_path = '/media/liu/File/11月数据准备/CASIA_TEMPLATE_TRAIN/gt'




        self.divide_mode_flag = 'num'
        self.train_num_you_want = 100
        self.train_percent = 0.7
        self.__path_issues()
        self.divide()

    def __switch_case(self, path):
        """
        针对不同类型的数据集做处理
        :return: 返回一个路径，这个路径是path 所对应的gt路径，并且需要检查该路径是否存在
        """
        # 0 判断路径的合法性
        if os.path.exists(path):
            pass
        else:
            print('The path :', path, 'does not exist')
            return ''
        # 1 分析属于何种类型
        # there are
        # 1.  sp generate data
        # 2. cm generate data
        # 3. negative data
        # 4. CASIA data

        sp_type = ['Sp']
        cm_type = ['Default','poisson']
        negative_type = ['negative']
        CASIA_type = ['Tp']
        debug_type = ['debug']
        template_type = ['TEMPLATE']
        type= []
        # name = path.split('/')[-1]
        name = path.split('\\')[-1]
        for sp_flag in sp_type:
            if sp_flag in name[:2]:
               type.append('sp')
               break

        for cm_flag in cm_type:
            if cm_flag in name[:7]:
                type.append('cm')
                break

        for negative_flag in negative_type:
            if negative_flag in name:
                type.append('negative')
                break

        for CASIA_flag in CASIA_type:
            if CASIA_flag in name[:2] and 'TEMPLATE' not in path:
                type.append('casia')
                break

        for template_flag in template_type:
            if template_flag in path:
                type.append('template')
                break

        # 判断正确性

        if len(type) != 1:
            print('The type len is ', len(type))
            return ''

        if type[0] == 'sp':
            gt_path = name.replace('Default','Gt').replace('.jpg','.bmp').replace('.png', '.bmp').replace('poisson','Gt')
            gt_path = os.path.join(self.sp_gt_path, gt_path)
            pass
        elif type[0] == 'cm':
            gt_path = name.replace('Default', 'Gt').replace('.jpg','.bmp').replace('.png', '.bmp').replace('poisson','Gt')
            gt_path = os.path.join(self.cm_gt_path, gt_path)
            pass
        elif type[0] == 'negative':
            gt_path = 'negative_gt.bmp'
            gt_path = os.path.join(self.negative_gt_path, gt_path)
            pass
        elif type[0] == 'casia':
            gt_path = name.split('.')[0] + '_gt' + '.png'
            gt_path = os.path.join(self.casia_gt_path, gt_path)
            pass
        elif type[0] == 'template':
            gt_path = name.split('.')[0] + '.bmp'
            gt_path = os.path.join(self.template_gt_path, gt_path)
            pass
        else:
            traceback.print_exc()
            print('Error')
            sys.exit()
        # 判断gt是否存在
        if os.path.exists(gt_path):
            pass
        else:
            return ''

        return gt_path
    def __gt_name_match(self, src_name):
        """
        使用src根据名字去寻找对应的GT
        :return:
        """
        if True:
            gt_name = src_name.replace('.png', '.bmp').replace('.jpg', '.bmp')
        elif 'COD10K' in src_name:
            gt_name = src_name.replace('.png', '.bmp').replace('.jpg','.bmp').replace('tamper','Gt')
        elif 'Sp_Default' in src_name:
            gt_name = src_name.replace('Default', 'Gt').replace('.jpg', '.bmp').replace('.png', '.bmp').replace('poisson','Gt')
        else:
            gt_name = src_name.replace('.png', '.bmp').replace('.jpg','.bmp')
        return gt_name
    def __path_issues(self):
        if not os.path.exists(DATASET_SRC_PATH):
            print(DATASET_SRC_PATH)
            print('DATASET_SRC_PATH错误，请确认输入的数据路径')
            sys.exit(1)
        if not os.path.exists(DATASET_TARGET_PATH):
            print('目标路径不存在，准备创建...')
            os.mkdir(DATASET_TARGET_PATH)
            os.mkdir(os.path.join(DATASET_TARGET_PATH, 'train_src'))
            os.mkdir(os.path.join(DATASET_TARGET_PATH, 'test_src'))
            os.mkdir(os.path.join(DATASET_TARGET_PATH, 'train_gt'))
            os.mkdir(os.path.join(DATASET_TARGET_PATH, 'test_gt'))

        else:
            print('目标路径存在，检查子文件夹')
            if os.path.exists(os.path.join(DATASET_TARGET_PATH, 'train_src')):
                print('数据集已经存在,请修改路径或者检查')
                sys.exit(1)


    def divide(self):
        train_percent = self.train_percent
        data_list = os.listdir(DATASET_SRC_PATH)
        if self.divide_mode_flag =='percent':
            print('总共数据有：%d张' % len(data_list))
            print('训练:测试 = %d:%d' % (train_percent * 10, 10 - train_percent))
            data_list = random.sample(data_list, len(data_list))
            train_set = random.sample(data_list, int(len(data_list) * train_percent))
            test_set = list(set(data_list).difference(set(train_set)))
            print(train_set)
            print(test_set)
        elif self.divide_mode_flag == 'num':
            print('总共数据有：%d张' % len(data_list))
            while(True):
                self.train_num_you_want = eval((input('请输入你想要的训练数据数量，其余的将会作为测试数据')))
                if self.train_num_you_want <= len(data_list):
                    break
                else:
                    print('总共数据有：%d张' % len(data_list))
                    print('你想要的训练数据量有：%d张' % self.train_num_you_want)
                    print('请重新输入')

            data_list = random.sample(data_list, len(data_list))
            train_set = random.sample(data_list, self.train_num_you_want)
            test_set = list(set(data_list).difference(set(train_set)))
            print('训练数据:%d,测试数据:%d' % (len(train_set), len(test_set)))
            print(train_set)
            print(test_set)
        else:
            print('总共数据有：%d张' % len(data_list))
            print('训练:测试 = %d:%d' % (train_percent * 10, 10 - train_percent))
            data_list = random.sample(data_list, len(data_list))
            train_set = random.sample(data_list, int(len(data_list) * train_percent))
            test_set = list(set(data_list).difference(set(train_set)))
            print(train_set)
            print(test_set)


        for index, train in enumerate(train_set):

            # 开始搬运train_set
            shutil.copy(os.path.join(DATASET_SRC_PATH, train),
                            os.path.join(DATASET_TARGET_PATH, 'train_src'+'/'+train))

            print('train_dataset:', index, '/', len(train_set))
        for index, test in enumerate(test_set):

            # 开始搬运test_set
            shutil.copy(os.path.join(DATASET_SRC_PATH, test),
                        os.path.join(DATASET_TARGET_PATH, 'test_src'+'/'+test))
            print('test_dataset:', index, '/', len(test_set))

        # 开始搬运GT

        for index, train in enumerate(train_set):

            gt_name = self.__gt_name_match(train)
            shutil.copy(os.path.join(DATASET_GT_PATH, gt_name),
                        os.path.join(DATASET_TARGET_PATH, 'train_gt'+'/'+gt_name))
            print('train_GT:', index, '/', len(train_set))
        for index, test in enumerate(test_set):
            gt_name = self.__gt_name_match(test)
            shutil.copy(os.path.join(DATASET_GT_PATH, gt_name),
                        os.path.join(DATASET_TARGET_PATH, 'test_gt'+'/'+gt_name))
            print('test_GT:', index, '/', len(test_set))

if __name__ == '__main__':
    DataDivide()