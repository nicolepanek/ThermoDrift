import os
import torch.nn as nn
import torch.optim as optim
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inference_script import forward_pass_analysis
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import csv
import ipdb



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

model_out = forward_pass_analysis(X_train, y_train, aa_comp=True)
with open('/usr/lusers/aultl/ThermoDrift/model_eval/20220527_analysis_train.csv', "w") as f:
    writer = csv.writer(f)
    header = ["predicted", "raw_probabilities", "true_label"]
    writer.writerow(header)
    f.writelines(model_out)

model_out = forward_pass_analysis(X_test, y_test)
with open('/usr/lusers/aultl/ThermoDrift/model_eval/20220527_analysis_test.csv', "w") as f:
    writer = csv.writer(f)
    header = ["predicted", "raw_probabilities", "true_label"]
    writer.writerow(header)
    f.writelines(model_out)
