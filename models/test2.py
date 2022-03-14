from audioop import bias
from configparser import Interpolation
from hashlib import sha1
from pydoc import plain
from time import time
from turtle import shape

#from zmq import device
import os

from numpy import dtype, flip
import torch
import time
import numpy as np
import cv2
import yaml
from numpy import size
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.utils import save_image
# from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

RUN_MODEL = False


def init_tensor():
    b = np.arange(72).reshape(3,6,4).astype("int")
    a = torch.Tensor(b)
    c = torch.unsqueeze(a,0)
    #a = torch.arange(27).view(3,3,3)
    d = c[0,0,:,:]/255
    transformed_tensor = d.cpu().detach().numpy()
    print("normalized tensor is", transformed_tensor)
    return d

def init_tensor_2():
    inp_img = read_image("/home/sush/Notes/PyTorch_for_DL/images/ex_image_1.png")
    print("example rgb size",inp_img)

def init_tensor_3():
    pad_mask = torch.FloatTensor(torch.zeros(3, 1328, 1328))
    return pad_mask

def norm_test():
    img = cv2.imread("/home/sush/depth_0.jpg", 1)
    print("orig image shape is", img.shape)
    start = time.time()
    orig_norm = img/255
    print("time taken for norm on large image is {}".format(round((time.time() - start) * 1000, 3)))

    reshaped_image = cv2.resize(img, (640,640))
    start_2 = time.time()
    reshaped_norm = reshaped_image/255
    print("time taken for norm on small image is {}".format(round((time.time() - start_2) * 1000, 3)))

if __name__ == '__main__':
    #input_tensor = init_tensor()
    norm_test()