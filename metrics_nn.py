'''
Created on Aug 20, 2023

@author: iapalm
'''

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class MetricsNN(nn.Module):
    def __init__(self):
        super(MetricsNN, self).__init__()
        
        self.fc1 = nn.Linear(15, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
    
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.sigmoid(out)
        
        return out