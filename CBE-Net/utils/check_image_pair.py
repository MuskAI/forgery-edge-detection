"""
created by haoran
time:2021-1-3
description:
1. using this file to check input tamper image and gt
2. using plt.show to visualization
"""
import matplotlib.pyplot as plt
import traceback
import numpy as np
def check_4dim_img_pair(img, gt):
    try:
        img = img.numpy().transpose(0,2,3,1)
        gt = gt.numpy().transpose(0,2,3,1)
        img_size = img.shape
        num_img = img_size[0]
        for i in range(num_img):
            _img = img[i]
            _gt = gt[i].squeeze(2)
            print(_img.shape)
            print(_gt.shape)
            plt.subplot(num_img,2,2*i+1)
            plt.xlabel('Max=%.2f Min=%.2f' % (max(_img.reshape(320 * 320 * 3)), min(_img.reshape(320 * 320 * 3))))
            plt.imshow(_img)
            plt.subplot(num_img,2,2*i+2)
            plt.xlabel('Max=%.2f Min=%.2f' % (max(_gt.reshape(320 * 320)), min(_gt.reshape(320 * 320))))
            plt.imshow(_gt)

        plt.show()

    except Exception as e:
        traceback.print_exc(e)


