#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：forgery-edge-detection 
@File    ：demo.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/10 08:29 
'''
import os
import io
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.mvssnet import get_mvss
from models.resfcn import ResFCN
from common.tools import inference_single
from common.utils import calculate_pixel_f1
from apex import amp

fake_path = './data/demo/fake.jpg'
fake_gt_path = './data/demo/fake_gt.png'
real_path = './data/demo/real.jpg'
model_type = 'mvssnet'
mvssnet_path = './ckpt/mvssnet_casia.pt'
resfcn_path = './ckpt/resfcn_casia.pt'
resize = 512
th = 0.5

if model_type == 'mvssnet':
    model = get_mvss(backbone='resnet50',
                     pretrained_base=True,
                     nclass=1,
                     sobel=True,
                     constrain=True,
                     n_input=3,
                     )
    checkpoint = torch.load(mvssnet_path, map_location='cpu')
elif model_type == 'fcn':
    model = ResFCN()
    checkpoint = torch.load(resfcn_path, map_location='cpu')

model.load_state_dict(checkpoint, strict=True)
model.cuda()
model.eval()


fake = cv2.imread(fake_path)
fake_size = fake.shape
fake_ = cv2.resize(fake, (resize, resize))
Image.fromarray(fake[:,:,::-1])

real = cv2.imread(real_path)
real_size = real.shape
real_ = cv2.resize(real, (resize, resize))
Image.fromarray(real[:,:,::-1])


with torch.no_grad():
    fake_seg, _ = inference_single(img=fake_, model=model, th=0)
    fake_seg = cv2.resize(fake_seg, (fake_size[1], fake_size[0]))


with torch.no_grad():
    real_seg, _ = inference_single(img=real_, model=model, th=0)
    real_seg = cv2.resize(real_seg.astype(np.uint8), (real_size[1], real_size[0]))


Image.fromarray((fake_seg).astype(np.uint8))
Image.fromarray((real_seg).astype(np.uint8))

if os.path.exists(fake_gt_path):
    fake_gt = cv2.imread(fake_gt_path, 0) / 255.0
    f1, p, r = calculate_pixel_f1(fake_seg.flatten(),fake_gt.flatten())
    print("fake_img, pixel_f1: %.4f, p: %.4f, r: %.4f"%(f1, p, r))
else:
    print("groundtruth: %s not exists."%fake_gt_path)
Image.fromarray((fake_gt * 255).astype(np.uint8))