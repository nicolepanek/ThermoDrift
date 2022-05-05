import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import os

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import thermodrift_model_seqfrac


def add_aa_composition_to_tensor(X_tensor):
    # first get the shape of the entire tensor
    tensor_shape = X_tensor.shape

    # sum_aa_counts has summed all the 1.0 for every type of aa in a 500 aa sequence
    # this array should have size [n_sequences(i.e. 60k), 25(types of aas)]
    sum_aa_counts = torch.sum(X_tensor, dim=1)
    print('sum_aa_counts: ', sum_aa_counts.shape)
    #assert sum_aa_counts.shape[0] == tensor_shape[0]

    # sum the number of amino acid positions that contain data in them to get the length of every sequence
    # this array should have size [n_sequences]
    #lengths_old = torch.sum((sum_aa_counts > 0).long(),dim=1)
    lengths = torch.sum(sum_aa_counts, dim=1)
    print('lengths: ', lengths.shape)

    # unsqueeze the lengths array to make an array of size [n_sequences,1]
    # Then expand this array to have duplicate sequence lengths for all 25 amino acid types
    # this repeat array should have size [n_sequences, 25(types of aas)]
    repeat = lengths.unsqueeze(dim=1).expand(-1, 25)
    print('repeat: ', repeat.shape)

    # Divide the summed counts of each aa type in a given sequence by the length of that sequence to get percent aa
    # Then unsqueeze to add a dimension at dim=1
    # this divide_tensors array should have size [n_sequences, 1, 25(types of aas)]
    divide_tensors = sum_aa_counts/repeat
    divide_tensors = divide_tensors.unsqueeze(dim=1)
    print('divide_tensors: ', divide_tensors.shape)

    # Now concatenate the amino acid fractions of each sequence to the end of the original tensor
    # this new combine_tensors array should have size [n_sequences, 501, 25]
    combine_tensors = torch.cat((X_tensor, divide_tensors), dim=1)
    print('combine_tensors: ', combine_tensors.shape)

    return combine_tensors


def load_data():
    # Load data
    X = torch.load('/gscratch/stf/jgershon/tensor_x.pt')
    Y = torch.load('/gscratch/stf/jgershon/tensor_y.pt')
    X = add_aa_composition_to_tensor(X)

    return X, Y


def split_data(X, Y):
    if 'X_train_aacomp.pt' not in os.listdir('/gscratch/stf/jgershon/'):
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
        torch.save(X_train, '/gscratch/stf/jgershon/X_train_aa_comp.pt')
        torch.save(X_test, '/gscratch/stf/jgershon/X_test_aa_comp.pt')
        torch.save(y_train, '/gscratch/stf/jgershon/y_train_aa_comp.pt')
        torch.save(y_test, '/gscratch/stf/jgershon/y_test_aa_comp.pt')
    else:
        X_train = torch.load('/gscratch/stf/jgershon/X_train_aa_comp.pt')
        X_test = torch.load('/gscratch/stf/jgershon/X_test_aa_comp.pt')
        y_train = torch.load('/gscratch/stf/jgershon/y_train_aa_comp.pt')
        y_test = torch.load('/gscratch/stf/jgershon/y_test_aa_comp.pt')

    return X_train, X_test, y_train, y_test


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-indir', type=str, required=False, default=None)
    parser.add_argument('-outdir', type=str, required=True, default=None)
    args = parser.parse_args()
    return args


args = get_args()
indir = args.indir
outdir = args.outdir


# Loading and processing the data:
X, Y = load_data()
X_train, X_test, y_train, y_test = split_data(X, Y)


# Do we need to normalize the one hot encoded tensors? Prob not.
# Generate train and test datasets
trainset = TensorDataset(X_train, y_train)
testset = TensorDataset(X_test, y_test)

# Prepare train and test loaders
train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=2)


# Instantiate the network
model = thermodrift_model_seqfrac.Net()
# Load model from previous state if indir arg is specified
if indir is not None:
    if len(indir) > 0:
        model.load_state_dict(torch.load(indir))
        model.eval()
        print('Model loaded from: ', indir)

# Instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()
# Instantiate the Adam optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=3e-4,
                       weight_decay=0.001)


# Moving tensors over to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device chosen: ', device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
model = model.to(device)


# batch_size, epoch and iteration
batch_size = 100
features_train = X.size()[0]
n_iters = 100000
num_epochs = int(n_iters/(features_train/batch_size))

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: ', num_parameters)


# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

output_dict = {}

# Number of iterations between validation cycles
n_run_valid = 500

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        train, labels = data

        # Clear gradients
        optimizer.zero_grad()
        print('train: ', train.size())
        # Forward propagation
        x_seq = train[:, :-1, :]
        x_frac = train[:, -1, :20]

        print('x_seq premodel: ', x_seq.size())
        print('x_frac premodel: ', x_frac.size())
        outputs = model(x_seq.unsqueeze(1), x_frac)

        # Calculate relu and cross entropy loss
        loss = criterion(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update weights
        optimizer.step()

        count += 1

        print('Train - example: '+str(i)+' loss: '+str(float(loss.data)))

        if count % n_run_valid == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            valid_loss = 0
            # Iterate through test dataset
            for j, data in enumerate(test_loader, 0):
                test, labels = data
                # Forward propagation
                x_seq = test[:, :-1, :]
                x_frac = test[:, -1, :20]

                outputs = model(x_seq.unsqueeze(1), x_frac)

                loss_valid = criterion(outputs, labels)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]

                # Total number of labels
                total += len(labels)

                correct += (predicted == labels).sum()
                valid_loss += float(loss_valid.data)
                #print('valid_loss: ', valid_loss)
            accuracy = 100 * correct / float(total)
            print('Valid - iter: '+str(count/n_run_valid) +
                  ' loss: '+str(float(valid_loss/(j+1))))

        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Train Loss: {}  Test Accuracy: {} %'.format(
                count, loss.data, accuracy))
            path = os.path.join(outdir, 'save_model/',
                                'model_'+str(count)+'.pt')
            torch.save(model.state_dict(), path)
            print('Model '+str(count)+' was saved.')
