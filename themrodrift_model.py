import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=1,
                                                out_channels=5,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=5,
                                                out_channels=10,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1), 
                                      nn.MaxPool2d(2, 2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(10),
                                      nn.Conv2d(in_channels=10,
                                                out_channels=20,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(20),
                                      nn.Conv2d(in_channels=20,
                                                out_channels=40,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                      nn.MaxPool2d(2, 2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(40))
        
        # Declare all the layers for classification
        self.classifier = nn.Sequential(
            #nn.Linear(7 * 7 * 40, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 3))
        
    def forward(self, x):
        print(f'x forward: {x.size()}')      
        # Apply the feature extractor in the input
        x = self.features(x)
       
        print(f'x from features {x.size()') 
        out_size = torch.tensor(x.size())
        # Squeeze the three spatial dimensions in one
        #x = x.view(-1, 7 * 7 * 40)
        x = x.view(-1,torch.prod(out_size[1:]))
        print(f'after squeeze: {x.size()}')
        # Classify the images
        lin = nn.Linear(torch.prod(out_size[1:]), 200)
        x = lin(x)
        print(f'x lin: {x.size()}')
        x = self.classifier(x)
        print(f'after classifier: {x.size()}')
        return x
