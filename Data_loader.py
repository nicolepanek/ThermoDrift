#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#define data file paths for running on hyak
thermo_path = '/usr/lusers/achazing/thermo_proteins/ThermoDrift/data/uniprot-thermophilus.fasta'
psychro_path = '/usr/lusers/achazing/thermo_proteins/ThermoDrift/data/uniprot-psychrophilus.fasta'
meso_path = '/usr/lusers/achazing/thermo_proteins/ThermoDrift/data/uniprot-mesophilus.fasta'


# In[3]:


#define data file paths for running locally/testing
thermo_path_local = '../../../data_sets/thermal_proteins/uniprot-thermophilus.fasta'
psychro_path_local = '../../../data_sets/thermal_proteins/uniprot-psychrophilus.fasta'
meso_path_local = '../../../data_sets/thermal_proteins/uniprot-mesophilus.fasta'


# In[4]:


#import Thermophllic proteins datasets
thermo_dict = {}
with open(thermo_path) as thermo_fasta_file:  # Will close handle cleanly
    identifiers = []
    sequence = []
    for seq_record in SeqIO.parse(thermo_fasta_file, 'fasta'):  # (generator)
        identifiers.append(str(seq_record.id))
        sequence.append(str(seq_record.seq))
        thermo_dict[str(seq_record.id)] = str(seq_record.seq)

thermo_list = list(thermo_dict.items())
df_thermo = pd.DataFrame(thermo_list)
df_thermo.columns = ['protein','sequence']
df_thermo['class'] = 'Thermophillic'
print(len(df_thermo.index))
#sample 1/5 of the thermo data
df_thermo = df_thermo.sample(frac=0.20)
print(len(df_thermo.index))
print('Thermo data has been processed')


# In[5]:


#import Psychrophillic proteins datasets
psychro_dict = {}
with open(psychro_path) as psychro_fasta_file:  # Will close handle cleanly
    psychro_identifiers = []
    psychro_sequence = []
    for seq_record in SeqIO.parse(psychro_fasta_file, 'fasta'):  # (generator)
        psychro_identifiers.append(str(seq_record.id))
        psychro_sequence.append(str(seq_record.seq))
        psychro_dict[str(seq_record.id)] = str(seq_record.seq)

psychro_list = list(psychro_dict.items())
df_psychro = pd.DataFrame(psychro_list)
df_psychro.columns = ['protein','sequence']
df_psychro['class'] = 'Psychrophillic'
print(len(df_psychro.index))
print('Psychro data has been processed')


# In[6]:


#import mesophillic proteins datasets
meso_dict = {}
with open(meso_path) as meso_fasta_file:  # Will close handle cleanly
    meso_identifiers = []
    meso_sequence = []
    for seq_record in SeqIO.parse(meso_fasta_file, 'fasta'):  # (generator)
        meso_identifiers.append(str(seq_record.id))
        meso_sequence.append(str(seq_record.seq))
        meso_dict[str(seq_record.id)] = str(seq_record.seq)

meso_list = list(meso_dict.items())
df_meso = pd.DataFrame(meso_list)
df_meso.columns = ['protein','sequence']
df_meso['class'] = 'Mesophillic'
print(len(df_meso.index))
print('Meso data has been processed')


# In[7]:


# concatenate the three dataframes
df_combine = pd.concat([df_thermo,df_meso,df_psychro]).reset_index(drop=True)
df_combine.head()


# In[8]:


#filter out proteins that don't start with methionine and are greater than 75 aas
good_list = []
bad_list = []
sequence_list = df_combine['sequence'].tolist()
for seq in sequence_list:
    if seq.startswith('M'):
        if len(seq) > 75:
            good_list.append(seq)
        
    else:
        bad_list.append(seq)
boolean_series = df_combine.sequence.isin(good_list)
df_filter = df_combine[boolean_series]
print(len(df_combine))
print(len(df_filter))


# In[62]:


#define lsts of filtered sequences and classes
seq_list = df_filter['sequence'].tolist()
class_list = df_filter['class'].tolist()


# In[10]:


#one hot encode protein sequence
amino_acids = "ARNDCQEGHILKMFPSTWYVUX_?-"  #  the order of the one hot encoded amino acids 
aa2num= {x:i for i,x in enumerate(amino_acids)} # create a dictionary that maps amino acid to integer

X_data = torch.tensor([])
for i,seq in enumerate(seq_list): 
    if len(seq) > 500:
        seq = seq[:500]     
    seq1hot = np.eye(len(amino_acids))[np.array([aa2num.get(res) for res in seq])] 
    #print(seq1hot)
    tensor = torch.tensor(seq1hot)
    tensor = torch.nn.functional.pad(tensor, (0,0,0,500-len(seq)))
    #print(tensor)
    if X_data.size()[0] == 0:
        X_data = tensor[None]
        print('Just made new tensor X_data')
    else:
        X_data = torch.cat((X_data,tensor[None]), axis=0)
        if i % 1000 == 0:
            print(f'Looped through {int(i)} sequences...')
        


# In[46]:


#one hot encode classes
classes = ['Thermophillic','Mesophillic','Psychrophillic']
#  the order of the one hot encoded classes 
class2num= {x:i for i,x in enumerate(classes)} 
# create a dictionary that maps class to integer
print(class2num)

y_data = torch.tensor([])

class_temp = [class2num[s] for s in class_list]
print(f'class_temp: {class_temp[:10]}')
print(f'class_temp: {len(class_temp)}')
y_data = torch.nn.functional.one_hot(torch.tensor(class_temp),3)
print(y_data.size())
print(y_data[0])


# In[15]:


#load the arrays into tensors and save them for training and testing the CNN model
  # transform to torch tensor


torch.save(y_data, '/gscratch/stf/achazing/tensor_y.pt') #download tensor to share for training
torch.save(X_data, '/gscratch/stf/achazing/tensor_x.pt') #download tensor to share for training



# In[ ]:





# In[61]:


# #test the data

# def one_hot_encode_protein(protein_seq):
#     amino_acids = "ARNDCQEGHILKMFPSTWYVUX_?-" #  the order of the one hot encoded sequences 
#     aa2num= {x:i for i,x in enumerate(amino_acids)} # create a dictionary that maps amino acid to integer
#     seq1hot = np.eye(len(amino_acids))[np.array([aa2num.get(res) for res in protein_seq])]
#     return seq1hot

# #define a function to take in a protein class and output a one hot encoded version
# def one_hot_encode_class(Class):
#     classes = ['Thermophillic','Mesophillic','Psychrophillic'] #  the order of the one hot encoded classes 
#     class2num= {x:i for i,x in enumerate(classes)} # create a dictionary that maps class to integer
#     class1hot = np.eye(len(classes))[np.array([class2num.get(Class)])]
#     return class1hot


# In[60]:


# test1 = seq_list[999]
# test2 = class_list[999]
# print(test2)


# In[59]:


# hot1_test1 = one_hot_encode_protein(test1)
# hot1_test2 = one_hot_encode_class(test2)
# class2num
# print(hot1_test2)


# In[57]:


# X_4real = torch.load('tensor_x.pt')[999]
# y_4real = torch.load('tensor_y.pt')[999][None]
# print(y_4real.size())
# print(f'y_4real: {y_4real}')
# #test_x = [aa2num[s] for s in seq_list[-1]]
# onehot_x = torch.tensor(one_hot_encode_protein(test1))
# onehot_y_temp = torch.tensor(one_hot_encode_class(test2))
# onehot_y = torch.nn.functional.one_hot(torch.tensor([class2num[test2]]),3)
# print(onehot_y_temp.shape)
# print(onehot_y.shape)
# #onehot_x.size()
# print(y_4real)
# print(onehot_y_temp)
# print(onehot_y)

# torch.equal(y_4real.long(),onehot_y.long())


# In[58]:


# print(f'seq_list: {len(seq_list)}')
# print(f'class_list: {len(seq_list)}')
# print(f'x_data: {X_data.size()}')


# In[ ]:




