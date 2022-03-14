import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def load_data():
	#Load data
	X = torch.load('/gscratch/stf/jgershon/tensor_x.pt')
	Y = torch.load('/gscratch/stf/jgershon/tensor_y.pt')
	return X,Y

def split_data(X,Y, save_data=False):
    assert X.size()[0] == Y.size()[0]
    #Convert y back from one hot encoding
    Y = torch.argmax(Y,dim=1)
    print('new Y: ',Y[:10])
    print('X load: ',X.size())
    print('Y load: ',Y.size())
    # Split data tensors into dev and test sets
    X_train, X_test, y_train, y_test = train_test_split( \
        X, Y, test_size = 0.20, random_state=42)
    print('X_train: ', X_train.size())
    print('X_test: ',X_test.size())
    print('y_train: ', y_train.size())
    print('y_test: ',y_test.size())
    if save_data:
        torch.save(X_train,'/gscratch/stf/jgershon/X_train.pt')
        torch.save(X_test,'/gscratch/stf/jgershon/X_test.pt')
        torch.save(y_train,'/gscratch/stf/jgershon/y_train.pt')
        torch.save(y_test,'/gscratch/stf/jgershon/y_test.pt')
    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)

    return trainset, testset

def make_data_loader(trainset, testset, batchsize=100):
    assert isinstance(batchsize, int),'Batch size should be 100'

    # Prepare train and test loaders
    train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size = batchsize,
                                          shuffle = True,
                                           num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size = batchsize,
                                         shuffle = True,
                                          num_workers=2) 
    return train_loader, test_loader

