#!/usr/bin/env python
# coding: utf-8

# In[42]:


#import modules
import pandas as pd
import numpy as np
import Bio
from Bio import Seq
from Bio import SeqIO
import torch
import matplotlib.pyplot as plt
import sys
from torch.utils.data import TensorDataset, DataLoader


# In[44]:


#define the fasta_to_classified_df function; which inputs fasta seqs and classifies them in a df
def fasta_to_classified_df(fasta_path,protein_class='',sample=False):
    seq_dict = {}  #define empty dict to store sequence ids and sequences
    with open(fasta_path) as fasta_file:  # Will close handle cleanly
        identifiers = []   #define empty id list 
        sequence = []   #define empty seq list 
        for seq_record in SeqIO.parse(fasta_path, 'fasta'):  # (generator)
            identifiers.append(str(seq_record.id))    #append ids to id list
            sequence.append(str(seq_record.seq))    #append seqs to seq list
            seq_dict[str(seq_record.id)] = str(seq_record.seq)    #define an ID, seq dictionary
    seq_list = list(seq_dict.items())  #enumerate the dictionary
    df_seqs = pd.DataFrame(seq_list)    #create a df from enumerated dictionary
    df_seqs.columns = ['protein','sequence']    #define column names
    df_seqs['class'] = protein_class    #define the class of each imported csv
    if sample == True:     
        df_seqs = df_seqs.sample(frac=0.20)    #if sample == True, then sample 1/5 of the data(i.e. for Thermo proteins)
    print(len(df_seqs.index))
    return df_seqs


# In[46]:


# define the combine_dfs function; which concatenates the three dataframes
def combine_dfs(list_of_dfs):
    df_combine = pd.concat(list_of_dfs).reset_index(drop=True)
    return df_combine


# In[48]:


#define the filter_seqs function
def filter_seqs(df_seqs):
    good_list = []
    bad_list = []
    sequence_list = df_seqs['sequence'].tolist()
    for seq in sequence_list:
        if seq.startswith('M'):
            if len(seq) > 75:
                good_list.append(seq)

        else:
            bad_list.append(seq)
    boolean_series = df_seqs.sequence.isin(good_list)
    df_filter = df_seqs[boolean_series]
    return df_filter


# In[29]:


# define the seq1hot function
def seq1hot(seq_list):
    amino_acids = "ARNDCQEGHILKMFPSTWYVUX_?-"  #  the order of the one hot encoded amino acids and other symbols
    aa2num= {x:i for i,x in enumerate(amino_acids)} # create a dictionary that maps amino acid to integer
    X_data = torch.tensor([])    #define an empty tensor to store one hot encoded proteins seqs
    for i,seq in enumerate(seq_list): 
        if len(seq) > 500:    #crop sequences longer than 500 aas
            seq = seq[:500]     
        protein1hot = np.eye(len(amino_acids))[np.array([aa2num.get(res) for res in seq])]    #one hot encode protein seq
        tensor = torch.tensor(protein1hot)    #create a tensor of one hot encoded proteins sequences 
        tensor = torch.nn.functional.pad(tensor, (0,0,0,500-len(seq)))   #for sequences less than 500 aas pad the end with zeros
        if X_data.size()[0] == 0:    #for the first iteration create an empty tensor
            X_data = tensor[None]
            print('Just made new tensor X_data')
        else:
            X_data = torch.cat((X_data,tensor[None]), axis=0)    #for each iteration concatenate the new sequence tensor to existing tensor
            if i % 1000 == 0:    #update user on the status of 1hotencoding, which is quite computationally expensive 
                print(f'Looped through {int(i)} sequences...')
    print(X_data.shape)
    print(type(X_data))
    return X_data


# In[51]:


#define the class1hot function
def class1hot(class_list):
    classes = ['Thermophillic','Mesophillic','Psychrophillic']   #the order of the one hot encoded classes
    class2num= {x:i for i,x in enumerate(classes)}    #  create a dictionary that maps class to integer
    y_data = torch.tensor([])   #define empty tensor to store class data
    print('Just made new tensor y_data')
    class_temp = [class2num[s] for s in class_list]   #loop through each class in the clast list and map string to dict
    y_data = torch.nn.functional.one_hot(torch.tensor(class_temp),3)  #one hot encode the classes as defined by the dict
    print(type(y_data))
    print(y_data.shape)
    return y_data


# In[53]:


#define the save_tensor function
def save_tensor(tensor,file_path_name):
    torch.save(tensor, file_path_name) #download tensor 

