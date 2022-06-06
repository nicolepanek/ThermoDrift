import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Declare all the layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(64, 3),
            nn.Softmax(dim=1))

    def forward(self, x):
        
        x = self.classifier(x)
        
    
        return x
