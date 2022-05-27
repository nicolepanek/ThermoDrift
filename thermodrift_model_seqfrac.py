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

    # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=1,
                                                out_channels=5,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=5,
                                                out_channels=20,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                      nn.MaxPool2d(2, 2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(20),
                                      nn.Conv2d(in_channels=20,
                                                out_channels=50,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(50),
                                      nn.Conv2d(in_channels=50,
                                                out_channels=100,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                      nn.MaxPool2d(2, 2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(100))

        self.fraction = nn.Sequential(
            nn.Linear(20, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 100))

        # Declare all the layers for classification
        self.classifier = nn.Sequential(
            # nn.Linear(7 * 7 * 40, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 3),
            nn.Softmax(dim=1))

    def forward(self, x_seq, x_frac):
        x_seq = self.features(x_seq.float())
        out_size = torch.tensor(x_seq.size())
        x_frac = self.fraction(x_frac.float())
    # Squeeze the three spatial dimensions in one
        x_seq = x_seq.view(-1, torch.prod(out_size[1:]))
    # Classify the images

        print('x_seq: ', x_seq.size())
        print('x_frac: ', x_frac.size())
        x_combo = torch.cat((x_seq, x_frac), dim=1)
        combo_len = torch.tensor(x_combo.size())
        print('combo_len', combo_len.size())
        lin = nn.Linear(torch.prod(combo_len[1:]), 200)
        x = lin(x_combo)
        x = self.classifier(x)
        return x
