from hashlib import sha1
from pydoc import plain
from turtle import shape

#from zmq import device
import os
import torch
import numpy as np
from numpy import size
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.io import read_image
# from torchvision import datasets, transforms

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

def transformer(inp_tensor):
    transformed_img = torch.flip(inp_tensor,[2])
    transformed_img = torch.nn.Sequential(T.Resize((640,640),antialias=True))
    transformed_tensor = transformed_img(inp_tensor)
    transformed_tensor = torch.unsqueeze(transformed_tensor,0)
    return transformed_tensor

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
    inp_img = read_image("/home/sush/depth_0.jpg")
    print(inp_img.size())
    transformed_tensor = transformer(inp_img)
    print(transformed_tensor.size())
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
