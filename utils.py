'''
Created on Aug 27, 2023

@author: iapalm
'''

import torch
import numpy as np

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def pad_board(t, dim=25):
    assert dim >= t.shape[1]
    assert dim >= t.shape[2]
    padded = torch.zeros((t.shape[0], dim, dim)).to(device)
    i_offset_factor = int((dim - t.shape[1]) / 2)
    j_offset_factor = int((dim - t.shape[2]) / 2)
    padded[:, i_offset_factor:i_offset_factor + t.shape[1], j_offset_factor:j_offset_factor + t.shape[2]] += t
    return padded

def state_to_tensor(state):
    return pad_board(torch.from_numpy(state).to(device)).unsqueeze(0).float()

def get_device():
    return device