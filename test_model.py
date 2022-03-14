import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
#import torch.nn as nn
#from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# DATAPREP
# Test data tensors. This will be replaced by Adam's data


#X = torch.rand(2000, 500, 20)
#y = torch.rand(2000, 1)
#batchsize = 100

def split_data(X, y):
  
  # Check that X and y have same number of examples
  assert X.size()[0] == y.size()[0], \
  "Mismatch in X and y number of examples. Check tensor size."

  # Split data tensors into dev and test sets
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size = 0.20, random_state=42)

  # Do we need to normalize the one hot encoded tensors? Prob not.
  # Generate train and test datasets
  trainset = TensorDataset(X_train, y_train)
  testset = TensorDataset(X_test, y_test)

  return trainset, testset

def make_dataloader(trainset, testset, batchsize):
  # Check that batchsize input is an int
  assert isinstance(batchsize, int), \
  "Variable batchsize input is not an integer"

  # Prepare train and test loaders
  train_loader = torch.utils.data.DataLoader(trainset,
                                             batch_size = batchsize,
                                             shuffle = True,
                                             num_workers=0)
  test_loader = torch.utils.data.DataLoader(testset,
                                            batch_size = batchsize,
                                            shuffle = False,
                                            num_workers=0) 
  return train_loader, test_loader

