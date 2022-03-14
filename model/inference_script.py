import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import thermodrift_model

#DATA LOADING FUNCTIONS



def forward_pass(data):
    '''
    Input data in shape [N,L,25]
    will process data through the model and then predict
    '''
    #Load model from saved outputs
    model_save_path = 'INSERT MODEL PATH HERE WHEN READY'
    model = thermodrift_model.Net()
    if os.path .isfile(model_save_path):
        model.load_state_dict(torch.load(PATH))
    
    outputs = model(data.unsqueeze(1))
    predicted = torch.max(outputs.data, 1)[1]
    raw_out = outputs.data
    return predicted, raw_out


def main(path):
	#Dataloading functions:
	








	#forward pass
    predicted, raw_out = forward_pass(data)

    return predicted, raw_out
		
