from hashlib import sha1
from turtle import shape

from numpy import size
import torch
import numpy as np

def main():
    b = np.arange(48).reshape(4,4,3).astype("int")
    a = torch.IntTensor(b)
    #a = torch.arange(27).view(3,3,3)
    print("initial bgr image \n",a)
    '''
    b_channel = a[:,:,0].detach().clone()
    r_channel = a[:,:,2].detach().clone()
    print("b channel is \n", b_channel)
    print("r channel is \n", r_channel)
    a[:,:,0] = r_channel
    a[:,:,2] = b_channel
    '''
    c = torch.flip(a,[2])
    print("flipped image is \n",c)

def placeholder():
    dummy_matrix = np.arange(1228800).reshape(3,640,640)
    dummy_matrix = np.expand_dims(dummy_matrix, axis=0)
    # Convert the image to row-major order, also known as "C order":
    dummy_matrix = np.ascontiguousarray(dummy_matrix)
    dummy_inp = torch.Tensor(dummy_matrix)
    print(dummy_inp.size())
    dummy_model = model(dummy_inp)
    make_dot(dummy_model, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

def test():
    b = np.arange(1228800).reshape(640,640,3)
    a = torch.IntTensor(b)
    a = torch.unsqueeze(a,0)
    print(a.size())

if __name__ == '__main__':
    test()