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

from torch.utils.data import TensorDataset, DataLoader

import emb_classifier


torch.cuda.empty_cache()

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-indir', type=str, required=False, default=None)
    parser.add_argument('-outdir', type=str, required=False, default=None)
    parser.add_argument('-data_dir', type=str, required=False, default='data/')
    args = parser.parse_args()
    return args

def suffle_n_batch(data, batch_size):
    batched = []
    random.shuffle(data)
    for i in range(len(data)//batch_size+1):
        batched.append(data[i*batch_size:i*batch_size+batch_size])
    
    if len(batched[-1])==0:
        return batched[:-1]
    else:
        return batched

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

def load_data(args):
    with open(os.path.join(args.data_dir,'train_tuple_data.pkl'), 'rb') as f:
        train = pickle.load(f)
         
    with open(os.path.join(args.data_dir,'valid_tuple_data.pkl'), 'rb') as f:
        valid = pickle.load(f)
    
    return train, valid


#get arguments
args = get_args()
indir = args.indir
outdir = args.outdir
print('Args got')
print(f'indir {indir}')
print(f'outdir {outdir}')

# Loading and processing the data:
train, valid = load_data(args)
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
# Load model from previous state if indir arg is specified
if indir is not None:
    if os.path.exists(indir):
        classifier.load_state_dict(torch.load(indir))
        print(f'loaded model from {indir}')        

# Instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()
# Instantiate the Adam optimizer
optimizer = optim.Adam(classifier.parameters(),lr=5e-5)
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
num_epochs = 100

output_dict = {}
print('Now beginning training')

torch.cuda.empty_cache()

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        seq, labels = data
        seq = seq.to(device)
        labels = labels.to(device)
        x, _ = get_emb_esm1b(seq, LM_model=LM_model, average=True)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = classifier(x)
        
        # Calculate relu and cross entropy loss
        loss = criterion(outputs, labels)/grad_accum
        print(f'outputs {outputs.tolist()}  lables {labels.tolist()}')
        # Calculating gradients
        loss.backward()
        
        if (i+1) % grad_accum == 0:
            total_norm = torch.nn.utils.clip_grad_norm_(classifier.parameters(),1.0)
            if not (total_norm == total_norm):
                print('Gradient are NaN')
                optimizer.zero_grad()
                continue
            optimizer.step()
            print('Train - epoch: '+str(epoch)+' batch: '+str(int((i+1)/grad_accum))+' loss: '+str(float(loss.data)*grad_accum))
        count += 1
 
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
            
            loss_valid = criterion(outputs, val_labels)

            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]

            # Total number of labels
            total += len(val_labels)

            correct += (predicted == val_labels).sum()
            valid_loss += float(loss_valid.data)
            # print('valid_loss: ', valid_loss)


    accuracy = 100 * correct / float(total)
    print('Valid - epcoh: '+str(epoch) +
              ' loss: '+str(float(valid_loss/(j+1)))+' accuracy: '+str(float(accuracy)))
    
    
    path = os.path.join(outdir,'save_model/model_'+str(epoch)+'.pt')
    torch.save(classifier.state_dict(), path)
    print('Model '+str(epoch)+' was saved.')
