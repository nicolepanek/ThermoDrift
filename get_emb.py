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


# Moving tensors over to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)
LM_model = LM_model.to(device)
LM_model.eval()
print(f'Moved tensors to {device}')

trainset = TensorDataset(x_train, y_train)
valset = TensorDataset(x_val, y_val)

train_loader = DataLoader(trainset, batch_size=1)
valid_loader = DataLoader(valset, batch_size=1)
print('Dataloaders built')



grad_accum = 256



print('Now beginning training')


for i, data in enumerate(train_loader):
    seq, labels = data
    seq = seq.to(device)
    labels = labels.to(device)
    x, _ = get_emb_esm1b(seq, LM_model=LM_model, average=True)
    if i == 0:
        comp_emb = torch.clone(x)
        y = torch.clone(labels)
    else:
        comp_emb = torch.cat((comp_emb,x),dim=0)
        y = torch.cat((y,labels),dim=0)
    print(f'Now embiding seq {i+1}', end='\r')


torch.save(comp_emb,os.path.join(args.data_dir,'train_embs.pt'))
torch.save(y,os.path.join(args.data_dir,'train_embs_y.pt'))


with open(os.path.join(args.data_dir,'train_embs.pkl'), 'wb') as f:
    print(f'Now saving to {os.path.join(args.data_dir,"train_embs.pkl")}')
    pickle.dump(comp_emb.cpu().detach().numpy(),f)
   
print('SAVED FILE') 
