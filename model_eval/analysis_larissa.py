import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import os, sys

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
sys.path.append("/usr/lusers/aultl/ThermoDrift/model_eval")
from inference_script import forward_pass_analysis


def load_data():
    # Load data
    X = torch.load('/gscratch/stf/jgershon/tensor_x.pt')
    Y = torch.load('/gscratch/stf/jgershon/tensor_y.pt')
    return X, Y


def split_data(X, Y):
    if 'X_train.pt' not in os.listdir('/gscratch/stf/jgershon/'):
        # Convert y back from one hot encoding
        Y = torch.argmax(Y, dim=1)
        print('new Y: ', Y[:10])

        print('X load: ', X.size())
        print('Y load: ', Y.size())

        # Split data tensors into dev and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.20, random_state=42)
        print('X_train: ', X_train.size())
        print('X_test: ', X_test.size())
        print('y_train: ', y_train.size())
        print('y_test: ', y_test.size())
        torch.save(X_train, '/gscratch/stf/jgershon/X_train.pt')
        torch.save(X_test, '/gscratch/stf/jgershon/X_test.pt')
        torch.save(y_train, '/gscratch/stf/jgershon/y_train.pt')
        torch.save(y_test, '/gscratch/stf/jgershon/y_test.pt')
    else:
        X_train = torch.load('/gscratch/stf/jgershon/X_train.pt')
        X_test = torch.load('/gscratch/stf/jgershon/X_test.pt')
        y_train = torch.load('/gscratch/stf/jgershon/y_train.pt')
        y_test = torch.load('/gscratch/stf/jgershon/y_test.pt')

    return X_train, X_test, y_train, y_test


# Loading and processing the data:
X, Y = load_data()
X_train, X_test, y_train, y_test = split_data(X, Y)

print("Success! All data has been loaded.")
model_out = forward_pass_analysis(X_train, y_train)
with open('/usr/lusers/aultl/ThermoDrift/model_eval/20220429_analysis_train.txt', "w") as f:
    f.writelines(model_out)
print("Train data processed by model.")

model_out = forward_pass_analysis(X_test, y_test)
with open('/usr/lusers/aultl/ThermoDrift/model_eval/20220429_analysis_test.txt', "w") as f:
    f.writelines(model_out)
print("Test data processed by model.")


