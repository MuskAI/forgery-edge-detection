"""
@created by haoran
time :0330
希望统一所有数据集,返回字典的形式,
只需要获取src的路径，不需要gt的路径
要创建保存结果的路径
"""
import os

import cv2
import shutil
class Datasets:
    def __init__(self, model_name=None, using_data=None, datasets_path=None):
        """
        @param model_name: 将作为保存路径的根目录名
        @param using_data: 用什么数据集，是一个dict
        @param datasets_path: 每个数据集所在的路径

        """
        assert model_name is not None
        if using_data is None:
            using_data = {'columbia': True,
                          'coverage': True,
                          'casia': False,
                          'ps-battle': False,
                          'in-the-wild': False,
                          }
        else:
            assert isinstance(using_data,dict)
            using_data = using_data
        if datasets_path is None:
            datasets_path = {
                'root': None,
                'columbia': None,
                'coverage': None,
                'casia': None,
                'my-protocol': None
            }
        else:
            assert isinstance(using_data, dict)
            datasets_path = datasets_path

        self.using_data = using_data
        self.datasets_path = datasets_path
        self.datasets_dict = {}
        self.default_setting ={'model_name':model_name,
                               'save_root':'/home/liu/haoran/test_results'}
        self.read()
        self.create_save_dirs()

    def read(self):
        # 准备好数据集
        for dataset_item in self.using_data:
            # 如果要使用这个数据集
            if self.using_data[dataset_item]:
                path = self.datasets_path[dataset_item] if self.datasets_path[dataset_item] is not None \
                    else os.path.join(self.datasets_path['root'], dataset_item, 'src')
                assert os.path.exists(path), '路径 {} 不存在'.format(path)
                self.datasets_dict[dataset_item] = {'path': path}
                # 获取每张图片的名称
                image_name_list = []

                for idx, item in enumerate(os.listdir(path)):
                    if item.split('.')[-1] in ('jpg','png','JPG','PNG','tif','bmp'):
                        image_name_list.append(item)
                        # TODO 检查是否是图片
                        if self.check_image(os.path.join(path,item)):
                            pass
                        else:
                            pass

                    else:
                        print('文件{}不是图片'.format(item))

                self.datasets_dict[dataset_item]['names'] = image_name_list
                self.datasets_dict[dataset_item]['nums'] = len(image_name_list)

        return self.datasets_dict

    def create_save_dirs(self, save_root=None):
        """
        创建结果的保存目录

        @param save_root:
        @return:
        """
        if save_root is not None:
            pass
        else:
            save_root = os.path.join(self.default_setting['save_root'],self.default_setting['model_name'])

        if not os.path.exists(save_root) or os.path.getsize(save_root)<10000:
            print('开始创建根目录 {}'.format(save_root))
            if not os.path.exists(save_root):
                os.mkdir(save_root)
            else:
                print('目录 {} 存在，但为空，进行删除'.format(save_root))
                if input('确定删除? 【y/n】:') == 'y':
                    shutil.rmtree(save_root)
                else:
                    raise '路径 {} 已经存在，请重新选择model_name'.format(save_root)
                os.mkdir(save_root)

            for i in self.datasets_dict:
                print('开始创建子目录 {}'.format(os.path.join(save_root,i)))
                os.mkdir(os.path.join(save_root,i))
                print('开始创建子目录 {}'.format(os.path.join(save_root, i, 'pred')))
                os.mkdir(os.path.join(save_root, i, 'pred'))
                self.datasets_dict[i]['save_path'] = os.path.join(save_root, i, 'pred')

        else:

            raise '路径 {} 已经存在，请重新选择model_name'.format(save_root)

        print('保存路径创建完成！')

    @staticmethod
    def check_image(path):
        try:
            img = cv2.imread(path)
            return True

        except:
            return False

    def get_datasets_dict(self):
        return self.datasets_dict
