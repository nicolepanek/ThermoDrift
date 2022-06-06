import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import os
import random
import pickle
import esm
import csv

from torch.utils.data import TensorDataset, DataLoader

import emb_classifier


torch.cuda.empty_cache()


def get_emb_esm1b(seq, LM_model, average=True):
    B, L = seq.shape
    L = L - 2 # remove start and end token
    #LM_model.eval()
    with torch.no_grad():
        output = LM_model(seq, repr_layers=[33], return_contacts=True) # get the output from the language model
        embedding = output['representations'][33][:,1:-1,:] # embedding size (1, L, 1280)
        attention_map = output['attentions'][:,:,:,1:-1,1:-1] # attention map size (1, 33, 20, L, L)
        attention_map = attention_map.reshape(B, 33*20, L, L).permute(0,2,3,1) # (1, L, L, 660)
        
        
        # if you wanna average the embeddings along the sequence dimension -- i think this could be really cool too
        if (average): 
            embedding = embedding.mean(1)
            
        return embedding,attention_map

def load_data(indir):
    with open(os.path.join(indir,'train_tuple_data.pkl'), 'rb') as f:
        train = pickle.load(f)
         
    with open(os.path.join(indir,'valid_tuple_data.pkl'), 'rb') as f:
        valid = pickle.load(f)
    
    return train, valid


#get arguments
indir = '/home/ec2-user/ThermoDrift/data/'

# Loading and processing the data:
train, valid = load_data(indir)
print('Data loaded')


#Preprocess data into tensors
LM_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter() 
print('ESM1b loaded')

#Convert data into format that esm1b will like
y_train, _, x_train = batch_converter(train)
y_val, _, x_val = batch_converter(valid)
y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)


# Instantiate the network
classifier = emb_classifier.Net()

indir = '/home/ec2-user/ThermoDrift/experiments/trainv6/save_model/model_11.pt' 
# Load model from previous state if indir arg is specified
classifier.load_state_dict(torch.load(indir))
print(f'loaded model from {indir}')        

# Instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()
# Instantiate the Adam optimizer
optimizer = optim.Adam(classifier.parameters(),lr=3e-4)
print('Classifier, optimizer, and criterion compiled')


# Moving tensors over to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

#x_train = x_train.to(device)
#y_train = y_train.to(device)
#x_val = x_val.to(device)
#y_val = y_val.to(device)
classifier = classifier.to(device)
LM_model = LM_model.to(device)
LM_model.eval()
print(f'Moved tensors to {device}')

trainset = TensorDataset(x_train, y_train)
valset = TensorDataset(x_val, y_val)

train_loader = DataLoader(trainset, shuffle=True, batch_size=1)
valid_loader = DataLoader(valset, shuffle=True, batch_size=1)
print('Dataloaders built')

num_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
print('Number of parameters classifier: ', num_parameters)

num_parameters = sum(p.numel() for p in LM_model.parameters() if p.requires_grad)
print('Number of parameters esm1b: ', num_parameters)

grad_accum = 256


# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
num_epochs = 1

output_dict = {}
print('Now beginning training')

torch.cuda.empty_cache()
model_out = []

for epoch in range(num_epochs):
    correct = 0
    total = 0
    valid_loss = 0
    for j, val_data in enumerate(valid_loader):
        with torch.no_grad():
                    
            val_seq, val_labels = val_data
            val_seq = val_seq.to(device)
            val_labels = val_labels.to(device) 
            val_x, _ = get_emb_esm1b(val_seq, LM_model=LM_model, average=True)

            outputs = classifier(val_x)
            
 
            #loss_valid = criterion(outputs, val_labels)

            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            
            raw_out = outputs.data
            pred = str([x for x in predicted])
            raw = str([x.tolist() for x in raw_out])
            #y = torch.nn.functional.one_hot(val_labels, num_classes=3)
            true_label = str([x for x in val_labels])
            s = ','.join([pred,raw,true_label,'\n'])
            model_out.append(s)
            
            print(f'Now on example {j+1}', end='\r')
            

 
with open('/home/ec2-user/ThermoDrift/20220606_analysis_test_esm1b_classifier.csv', "w") as f:
    writer = csv.writer(f)
    header = ["predicted", "raw_probabilities", "true_label"]
    writer.writerow(header)
    f.writelines(model_out) 
