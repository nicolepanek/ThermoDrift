import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import thermodrift_model


#Load data
X = torch.load('/gscratch/stf/jgershon/tensor_x.pt')
Y = torch.load('/gscratch/stf/jgershon/tensor_y.pt')


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
torch.save(X_train,'/gscratch/stf/jgershon/X_train.pt')
torch.save(X_test,'/gscratch/stf/jgershon/X_test.pt')
torch.save(y_train,'/gscratch/stf/jgershon/y_train.pt')
torch.save(y_test,'/gscratch/stf/jgershon/y_test.pt')

# Check that the X_train has the same number of examples as y_train
assert X_train.size()[0] == y_train.size()[0], \
"Mismatch in X_train and y_train number of examples. Check tensor size."

# Check that X_test has the same number of examples as y_test
assert X_test.size()[0] == y_test.size()[0], \
"Mismatch in X_test and y_test number of examples. Check tensor size."

# Do we need to normalize the one hot encoded tensors? Prob not.
# Generate train and test datasets
trainset = TensorDataset(X_train, y_train)
testset = TensorDataset(X_test, y_test)

# Prepare train and test loaders
train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size = 100,
                                          shuffle = True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size = 100,
                                         shuffle = False,
                                          num_workers=2) 


# Instantiate the network
model = thermodrift_model.Net()
# Instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()
# Instantiate the Adam optimizer
optimizer = optim.Adam(model.parameters(),
                       lr = 3e-4,
                       weight_decay= 0.001)

#Moving tensors over to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
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
print('Number of parameters: ',num_parameters)


# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        train, labels = data
        
        #Clear gradients
        optimizer.zero_grad()
        
        #Forward propagation
        outputs = model(train.unsqueeze(1))
	        
        #Calculate relu and cross entropy loss
        loss = criterion(outputs, labels)

        #Calculating gradients
        loss.backward()
        
        #Update weights
        optimizer.step()
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for j, data in enumerate(test_loader, 0):
                test, labels = data
                
                # Forward propagation
                outputs = model(test.unsqueeze(1))
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss value and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 600 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
            loss = torch.tensor(loss_list)
            torch.save(loss, '/gscratch/stf/jgershon/loss.pt')
            accuracy = torch.tensor(accuracy_list)
            torch.save(accuracy, '/gscratch/stf/jgershon/accuracy.pt')



