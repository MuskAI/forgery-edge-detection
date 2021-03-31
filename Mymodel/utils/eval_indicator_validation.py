"""
created by haoran
time:8-21
对部分代码的有效性进行验证
"""
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score

# 全局变量
PREDICTION_IMG_PATH = r'C:\Users\musk\Desktop\mid_result_epoch_0\mid_output\mid_output_epoch0_batch_index0@0.png'
LABEL_IMG_PATH = r'C:\Users\musk\Desktop\mid_result_epoch_0\mid_label\mid_label_epoch0_batch_index0@0.png'


def my_precision_score(prediction, label):
    y = np.reshape(prediction,prediction.size)
    l = np.reshape(label,label.size)
    y = np.where(y > 0.5, 1, 0).astype('int')
    return precision_score(y, l, average='macro')


def my_acc_score(prediction, label):
    y = np.reshape(prediction,prediction.size)
    l = np.reshape(label,label.size)
    y = np.where(y > 0.5, 1, 0).astype('int')
    return accuracy_score(y, l)


def my_f1_score(prediction, label):
    y = np.reshape(prediction,prediction.size)
    l = np.reshape(label,label.size)
    y = np.where(y > 0.5, 1, 0).astype('int')
    print('sum',sum(y))

    return f1_score(y, l, average='macro')

def img_read():
    prediction_img = Image.open(PREDICTION_IMG_PATH)
    label_img = Image.open(LABEL_IMG_PATH)
    prediction_img = np.array(prediction_img)
    label_img = np.array(label_img)
    print(prediction_img.shape)
    print(label_img.shape)
    prediction_img = prediction_img[:,:,0] /255
    label_img = label_img[:,:,0]/255
    f1 = my_f1_score(prediction_img,label=label_img)
    acc = my_acc_score(prediction_img,label_img)
    precision = my_precision_score(prediction_img,label_img)
    print('f1:',f1)
    print('acc',acc)
    print('precision',precision)

if __name__ == '__main__':
    img_read()