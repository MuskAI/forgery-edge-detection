"""
@author :haoran
time:0326
"""
from PIL import Image,ImageFilter
import numpy as np
import traceback
import random
from torchvision import transforms
import matplotlib.pylab as plt

class AddGlobalBlur(object):
    """
    增加全局模糊t
    """

    def __init__(self, kernel_size=1):
        """
        :param kernel_size: kernel_size
        """
        self.kernel_size = kernel_size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        img_ = np.array(img).copy()
        img_ = Image.fromarray(img_)
        img_ = img_.filter(ImageFilter.GaussianBlur(radius=self.kernel_size))
        img_ = np.array(img_)
        return Image.fromarray(img_.astype('uint8')).convert('RGB')

class PressureTest:
    def __init__(self):
        """
        进来的是Image 输出的 tensor
        """
        pass
    def jepg(self):
        pass
    def blur(self, src_path, kernel_size=1):
        try:
            src = Image.open(src_path)
        except Exception as e:
            print(e)
            return 'read img error'

        src = transforms.Compose([
            AddGlobalBlur(kernel_size=kernel_size),
            # transforms.ToTensor(),
            # transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
        ])(src)
        plt.imshow(src)
        plt.show()

if __name__ == '__main__':
   pred_path = './test/39t.bmp'
   gt_path = './test/39t_gt.bmp'
   src_path = './test/Tp_D_CND_S_N_txt00028_txt00006_10848.jpg'
   tm = PressureTest()
   tm.blur(src_path=src_path,kernel_size=10)
