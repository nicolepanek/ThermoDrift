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
			out_channels=10,
			kernel_size=3,
			stride=1,
			padding=1), 
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=10,
			out_channels=100,
			kernel_size=3,
			stride=1,
			padding=1), 
			nn.MaxPool2d(2, 2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(100),
			nn.Conv2d(in_channels=100,
			out_channels=300,
			kernel_size=3,
			stride=1,
			padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(300),
			nn.Conv2d(in_channels=300,
			out_channels=1000,
			kernel_size=3,
			stride=1,
			padding=1),
			nn.MaxPool2d(2, 2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(1000))

        # Declare all the layers for classification
		self.classifier = nn.Sequential(
            #nn.Linear(7 * 7 * 40, 200),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(200, 500),
			nn.ReLU(inplace=True),
			nn.Linear(500, 3))


	def forward(self, x):
		print('x forward: ',x.size())
        # Apply the feature extractor in the input
		print('x type: ',x.type())
		x = self.features(x.float())
		print('x from features: ',x.size())
		out_size = torch.tensor(x.size())
        # Squeeze the three spatial dimensions in one
        #x = x.view(-1, 7 * 7 * 40)
		x = x.view(-1,torch.prod(out_size[1:]))
		print('after squeeze: ',x.size())
        # Classify the images
		lin = nn.Linear(torch.prod(out_size[1:]), 200)
		x = lin(x)
		print('x lin: ',x.size())
		x = self.classifier(x)
		print('after classifier: ',x.size())
		return x
