import os, sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()
class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MidResultSave:
    def __init__(self):
        pass
    def train_mid_result(self):
        # if abs(batch_index - dataParser.steps_per_epoch) == 50:
        #     _output = outputs[0].cpu()
        #     _output = _output.detach().numpy()
        #     for i in range(2):
        #         t = _output[i, :, :]
        #         t = np.squeeze(t, 0)
        #         t = t*255
        #         t = np.array(t,dtype='uint8')
        #         t = Image.fromarray(t)
        #
        #         t.save('the_midoutput_%d_%d'%(epoch,batch_index))
        pass

def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1,4,1,1]):
            torch.nn.init.constant_(m.weight, 0.25)
        if m.bias is not None:
            m.bias.data.zero_()

def save_mid_result(mid_output, label, epoch, batch_index, mid_save_root='./mid_result_820',train_phase=True,save_8map=False):
    """
    输入的是网络直接输出的结果，一个list
    label也是一个list
    分为train阶段的中间结果
    和val阶段的中间结果
    :param mid_output:
    :return:
    """
    if mid_save_root == '':
        print('mid_save_root为空')
        sys.exit()

    # 训练阶段、验证阶段分开处理
    if train_phase:
        mid_save_root = mid_save_root+'_train'
    else:
        mid_save_root = mid_save_root + '_val'


    if os.path.exists(mid_save_root) == False:
        print('中间结果根目录不存在，正在创建')
        os.mkdir(mid_save_root)
    else:
        pass

    dir_name = 'mid_result_epoch_%d' % (epoch)
    dir_path = os.path.join(mid_save_root, dir_name)
    mid_output_dir = os.path.join(dir_path, 'mid_output')
    mid_label_dir = os.path.join(dir_path, 'mid_label')
    mid_8_map_dir = os.path.join(dir_path, 'mid_8_map')
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)
        os.mkdir(mid_output_dir)
        os.mkdir(mid_label_dir)
        os.mkdir(mid_8_map_dir)
        print(dir_path)
        print(mid_label_dir)
        print(mid_output_dir)
        print(mid_8_map_dir)
        print('创建目录成功！！！！！')
    else:
        print(dir_path, '已经存在')

    # 每一个list 里面有一个pred 和 8张图 每一个又是N个batch_size
    # 先遍历list区分pred 和 8张图
    # 然后对每一个进行batch遍历保存每一张图
    # 命名应该在内层得到
    for index in range(len(mid_output)):
        if index == 0:

            show_outputs = np.array(mid_output[index].cpu().detach()) * 255
            show_outputs = np.array(show_outputs, dtype='uint8')
            show_labels = np.array(label[index].cpu().detach()) * 255
            show_labels = np.array(show_labels, dtype='uint8')
            for i in range(show_outputs.shape[0]):
                file_name_output = 'mid_output_epoch%d_batch_index%d@%d.png' % (epoch, batch_index, i)
                file_output_dir = os.path.join(mid_output_dir, file_name_output)
                file_name_label = 'mid_label_epoch%d_batch_index%d@%d.png' % (epoch, batch_index, i)
                file_label_dir = os.path.join(mid_label_dir, file_name_label)

                show_outputs_t = Image.fromarray(show_outputs[i, 0, :, :]).convert('RGB')
                show_outputs_t.save(file_output_dir)

                show_labels_t = Image.fromarray(show_labels[i, 0, :, :]).convert('RGB')
                show_labels_t.save(file_label_dir)
        else:
            if save_8map:

                for i in range(show_outputs.shape[0]):
                    show_8map = np.array(mid_output[index].cpu().detach()) * 255
                    show_8map = np.array(show_8map, dtype='uint8')
                    file_name_8map = 'mid_8map_epoch%d_batch_index%d@No%d-%d.png' % (epoch, batch_index,index,i)
                    file_8map_dir = os.path.join(mid_8_map_dir, file_name_8map)
                    show_8map = show_8map.transpose((0,2,3,1))
                    show_8map_t = Image.fromarray(show_8map[i, :, :, :]).convert('RGB')
                    show_8map_t.save(file_8map_dir)
            else:
                # print('不保存')
                break

def to_none_class_map(dou_em):
    # 转化为无类别的GT 100 255 为边缘
    dou_em = np.where(dou_em == 50, 0, dou_em)
    dou_em = np.where(dou_em == 100, 1, dou_em)
    dou_em = np.where(dou_em == 255, 1, dou_em)
    dou_em = np.array(dou_em[:, :])
    # dou_em = np.expand_dims(dou_em, 2)
    return dou_em



import urllib, urllib.request, sys
import ssl

def send_msn(epoch,f1):
    host = 'https://intlsms.market.alicloudapi.com'
    path = '/comms/sms/sendmsgall'
    method = 'POST'
    appcode = ''
    querys = ''
    bodys = {}
    url = host + path

    bodys['callbackUrl'] = '''http://test.dev.esandcloud.com'''
    bodys['channel'] = '''0'''
    bodys['mobile'] = '''+8613329825566'''
    bodys['templateID'] = '''20201108001936'''
    epoch = str(epoch)
    f1 = str(f1)
    bodys['templateParamSet'] = [epoch,f1]
    post_data = urllib.parse.urlencode(bodys).encode("UTF8")
    request = urllib.request.Request(url, post_data)
    request.add_header('Authorization', 'APPCODE ' + appcode)
    # 根据API的要求，定义相对应的Content-Type
    request.add_header('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8')
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    response = urllib.request.urlopen(request, context=ctx)
    content = response.read()
    if (content):
        print(content)

