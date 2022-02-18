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

def placeholder():
    dummy_matrix = np.arange(1228800).reshape(3,640,640)
    dummy_matrix = np.expand_dims(dummy_matrix, axis=0)
    # Convert the image to row-major order, also known as "C order":
    dummy_matrix = np.ascontiguousarray(dummy_matrix)
    dummy_inp = torch.Tensor(dummy_matrix)
    print(dummy_inp.size())
    #dummy_model = model(dummy_inp)
    #make_dot(dummy_model, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    '''
    # flip channels to go from bgr to rgb, the input tensor automatically gets dimension added
    # input tensor becomes [1,3,1200,1328] and we will flip the 2nd dimension
    #initialize tensors
    h,w = input_tensor.size(1), input_tensor.size(2)
    pad_const = int((w-h)/2)

    weights_1 = torch.Tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
    print("weight size is :", weights_1.size())
    weights_2 = torch.Tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
    padded_inp = torch.cat((weights_1, weights_2), 1)
    padded_inp_2 = torch.cat((padded_inp, weights_2), 1)

    # Define conv which is used only for padding
    conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=(pad_const,0), bias=False)
    #flipped_image = torch.flip(input_tensor,[0]).unsqueeze(0)
    flipped_image = input_tensor.unsqueeze(0)
    print("flipped image size is", flipped_image.size())
    # convolution requires a normalized image
    flipped_image_2 = flipped_image/255
    # make sure conv runs without grad (as the weights should not be learnt)
    with torch.no_grad():
        conv.weight = nn.Parameter(padded_inp_2)
        padded_img = conv(flipped_image_2)
        print("after conv size", padded_img.size())
    interpolation = T.InterpolationMode.NEAREST
    #transformer = torch.nn.Sequential(T.Pad((0,pad_const)),T.Resize((640,640),interpolation=interpolation))
    transformer = torch.nn.Sequential(T.Resize((640,640),interpolation=interpolation))
    transformed_tensor = transformer(padded_img)
    #transformed_tensor = torch.unsqueeze(transformed_tensor,0)
    return transformed_tensor[0,:,:,:]*255
    '''

def transformer(input_tensor):
    # flip channels to go from bgr to rgb, the input tensor automatically gets dimension added
    # input tensor becomes [1,3,1200,1328] and we will flip the 2nd dimension
    #initialize tensors
    #h,w = input_tensor.size(1), input_tensor.size(2)
    #pad_const = int((w-h)/2)
    input_tensor = input_tensor.type(torch.HalfTensor)
    print("input tensor type: ", input_tensor.dtype)
    pad_mask = torch.zeros(3, 1328, 1328, dtype=torch.float16)
    #pad_mask[:,0:64,:] = 1
    #pad_mask[:,1264:1328,:] = 1
    print("pad mask size is :", pad_mask.size)
    
    #flipped_image = torch.flip(input_tensor,[0])
    flipped_image = input_tensor
    print("flipped image size is", flipped_image.size())

    pad_mask[:,64:1264,:] = flipped_image
    print("padded image size is", pad_mask.size())
    
    interpolation = T.InterpolationMode.NEAREST
    #transformer = torch.nn.Sequential(T.Pad((0,pad_const)),T.Resize((640,640),interpolation=interpolation))
    transformer = torch.nn.Sequential(T.Resize((640,640),interpolation=interpolation))
    transformed_tensor = transformer(pad_mask)
    transformed_tensor = transformed_tensor.type(torch.IntTensor)
    print("final image_tensor", transformed_tensor)
    return transformed_tensor

def load_yaml(cfg='yolov5m.yaml'):
    with open(cfg, errors='ignore') as f:
                yaml_1 = yaml.safe_load(f)  # model dict

    # Define model
    print(yaml_1)
    yaml_1['ch'] = yaml_1.get('ch')  # input channels\
    ch = yaml_1['ch']
    return ch

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.features =nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=0)
        '''
        with torch.no_grad():
            weights = torch.tensor(...)
            self.features.weight = nn.Parameter(weights)
        '''

    def forward(self, x):
        x = self.features(x)
        return x

class Custom_Layer(nn.Module):
    # Custom layer for preprocessing
    def __init__(self):
        super().__init__()

    def forward(self,input):
        # flip channels to go from bgr to rgb
        x = torch.flip(input,[2])
        return x
    
    # to add layer to model do:
    # in init: self.preproc_layers = Custom_Layer()
    # in forward: mod = self.preproc_layers(input)

if __name__ == '__main__':
    #plain_tensor, exp_tensor = init_tensor()
    #ch = load_yaml()
    #print(ch)
    
    inp_img = read_image("/home/sush/depth_0.jpg")

    print("input image size",inp_img.size())
    transformed_tensor = transformer(inp_img)
    #transformed_tensor_2 = transformed_tensor.int()
    #print("transformed image size",transformed_tensor_2.size())
    #print("transformed image is", transformed_tensor_2)
    #transformed_tensor = transformed_tensor.cpu().detach().numpy()
    #im = Image.fromarray(transformed_tensor)
    #im.save("/home/sush/TS/depth_img_transformed.png")
    print("random element of output image", transformed_tensor[1,200,200])
    plt.imshow(transformed_tensor.permute(1, 2, 0))
    plt.show()
    #save_image(transformed_tensor,'/home/sush/depth_img_transformed.png')
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = NeuralNetwork().to(device)
    print(model)
    if RUN_MODEL == True:
        logits = model(inp_tensor)
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)
        print(f"Predicted class: {y_pred}")
    '''
