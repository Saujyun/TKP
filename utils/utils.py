from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp

import torch
import re
import matplotlib.pyplot as plt

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
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

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def draw_curve():
    y_mAP_i2i = []
    y_mAP_i2v = []
    y_mAP_v2i =[]
    y_mAP_v2v = []
    y_mAP = [y_mAP_i2i,y_mAP_v2v,y_mAP_v2i,y_mAP_i2v]
    # y_mAP = []
    x_epoch = list(range(10,160,10))
    print(x_epoch)
    y_rank1_i2i = []
    y_rank1_i2v = []
    y_rank1_v2i = []
    y_rank1_v2v = []
    y_rank1 = [y_rank1_i2i,y_rank1_v2v,y_rank1_v2i,y_rank1_i2v]
    # y_rank1 = []
    fp = open('/home/lhy/-/模型论文代码/TKP/log-mars/log_train.txt',"r")
    line = fp.readline()
    index = 0
    while line:
        if(index==4):
            index = 0
        if(line.__contains__("mAP")):
            print("--" + line)
            y1 = re.compile('(?<=top1:)\s*\d*\.\d*').findall(line)
            y2 = re.compile('(?<=mAP:)\s*\d*\.\d*').findall(line)
            y_rank1[index].append(y1)
            y_mAP[index].append(y2)
            index = index+1
            print(y1,y2)
        line = fp.readline()
    print(y_rank1, y_mAP)
    name = ['i2i','v2v','v2i','i2v']
    for i in range(4):
        ax1 = plt.figure().add_subplot(111)
        ax1.plot(x_epoch, y_rank1[i], color="r", linestyle="-", marker="^", label='y_rank1')
        ax1.plot(x_epoch, y_mAP[i], color="b", linestyle="-", marker="s", label='y_mAP')
        plt.ylabel("y_rank-1/y_mAP")
        plt.xlabel("Epoch")
        plt.legend(loc=4)
        y_max = max(y_rank1[i])
        y_max_mAP = max(y_mAP[i])
        x_max_mAP = 10 * (y_rank1[i].index(y_max)+1)
        x_max = 10 * (y_rank1[i].index(y_max)+1)
        plt.annotate("[" + str(x_max) + "," + str(y_max[0]) + "]", xy=(x_max, y_max[0]))
        ax1.plot(x_max, y_max, 'ks')
        plt.annotate("[" + str(x_max_mAP) + "," + str(y_max_mAP[0]) + "]", xy=(x_max_mAP, y_max_mAP[0]))
        ax1.plot(x_max_mAP, y_max_mAP, 'ks')
        plt.title(name[i])
        plt.savefig(name[i])
    plt.show()
    fp.close()

if __name__ == '__main__':
    draw_curve()









