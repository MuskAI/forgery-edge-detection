#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：forgery-edge-detection 
@File    ：cam.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/2 8:14 PM 
'''
# Define your model
import os

import torch
from torchvision.models import resnet101
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
model = resnet101(num_classes=2).eval()
checkpoint = torch.load('/home/liu/haoran/forgery-edge-detection/CBE-Net/train/save_model/test2/epoch89_loss0.0680070139162415.pth')
model.load_state_dict(checkpoint['state_dict'])

# Set your CAM extractor
from torchcam.methods import SmoothGradCAMpp
cam_extractor = SmoothGradCAMpp(model)


# Get your input
img_root = '/home/liu/haoran/3月最新数据/casia_au_and_casia_template_after_divide/train_src'
img_list = os.listdir(img_root)
for name in img_list:
    path = os.path.join(img_root,name)
    img = read_image(path)
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (320, 320)) / 255., (0.47, 0.43, 0.39), (0.27, 0.26, 0.27))

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    # import matplotlib.pyplot as plt
    # # Visualize the raw CAM
    # plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()


    import matplotlib.pyplot as plt
    from torchcam.utils import overlay_mask

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.subplot(121)
    plt.imshow(to_pil_image(img))
    plt.subplot(122)
    plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()