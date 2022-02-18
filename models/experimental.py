# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""

from pickletools import int4
from turtle import forward
from unicodedata import name
import numpy as np
import math

from pandas import Int16Dtype, Int32Dtype
from obj_det_lib.tf_obj_det_api.official.vision.image_classification.classifier_trainer import initialize
import torch
import torch.nn as nn
import torchvision.transforms as T

from models.common import Conv
from utils.downloads import attempt_download


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()
        self.stride = [1,1]

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

class Custom_Layer(nn.Module):
    # Custom layer for preprocessing
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        # flip channels to go from bgr to rgb, the input tensor automatically gets dimension added
        # input tensor becomes [1,3,1200,1328] and we will flip the 2nd dimension
        #initialize tensors
        # flip channels to go from bgr to rgb, the input tensor automatically gets dimension added
        # input tensor becomes [1,3,1200,1328] and we will flip the 2nd dimension
        #initialize tensors
        #h,w = input_tensor.size(1), input_tensor.size(2)
        #pad_const = int((w-h)/2)
        input_tensor = input_tensor.type(torch.HalfTensor)
        print("input tensor type: ", input_tensor.dtype)
        pad_mask = torch.zeros(3, 1328, 1328, dtype=torch.float16)
        
        flipped_image = torch.flip(input_tensor,[1])
        pad_mask[:,64:1264,:] = flipped_image
    
        interpolation = T.InterpolationMode.NEAREST
        #transformer = torch.nn.Sequential(T.Pad((0,pad_const)),T.Resize((640,640),interpolation=interpolation))
        transformer = torch.nn.Sequential(T.Resize((640,640),interpolation=interpolation))
        transformed_tensor = transformer(pad_mask)
        return transformed_tensor

class Custom_Model(nn.Module):
    def __init__(self, pretrained_model, nc, names, stride):
        super(Custom_Model, self).__init__()
        self.nc = nc
        self.names = names
        self.preproc_layers = Custom_Layer()
        self.pretrained = pretrained_model
        self.stride = stride
    
    def forward(self, input):
        mod = self.preproc_layers(input)
        mod = self.pretrained(mod)
        return mod


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse


    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        print("returned plain model")
        print("model.stride is: ", model.stride)
        return model[-1], model.stride   # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model # return ensemble

def custom_load(weights, device):
    existing_model, stride = attempt_load(weights, map_location=device, inplace=True, fuse=True)
    nc_existing, names_existing = existing_model.nc, existing_model.names
    extended_model = Custom_Model(pretrained_model=existing_model, nc=nc_existing, names=names_existing, stride=stride)
    return extended_model
