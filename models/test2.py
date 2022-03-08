from audioop import bias
from configparser import Interpolation
from hashlib import sha1
from pydoc import plain
from turtle import shape

#from zmq import device
import os

from numpy import dtype, flip
import torch
import numpy as np
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
    print("initial bgr image  \n",a.size())
    return a, c

def init_tensor_2():
    inp_img = read_image("/home/sush/Notes/PyTorch_for_DL/images/ex_image_1.png")
    print("example rgb size",inp_img)

def init_tensor_3():
    pad_mask = torch.FloatTensor(torch.zeros(3, 1328, 1328))
    return pad_mask

if __name__ == '__main__':
    input_tensor = init_tensor()
    print("input tensor is ", input_tensor)
    print("input tensor type is ", input_tensor.dtype)