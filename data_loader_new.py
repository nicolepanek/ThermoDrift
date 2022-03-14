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
import functions


# In[2]:


#define data file paths for running locally/testing
thermo_path_local = '../../../data_sets/thermal_proteins/uniprot-thermophilus.fasta'
psychro_path_local = '../../../data_sets/thermal_proteins/uniprot-psychrophilus.fasta'
meso_path_local = '../../../data_sets/thermal_proteins/uniprot-mesophilus.fasta'


# In[3]:


#run the fasta_to_classified_df function
df_thermo = functions.fasta_to_classified_df(thermo_path_local,protein_class='Thermophillic',sample=True)
df_meso = functions.fasta_to_classified_df(meso_path_local,protein_class='Mesophillic')
df_psychro = functions.fasta_to_classified_df(psychro_path_local,protein_class='Psychrophillic')


# In[4]:


#run the combine_dfs function
list_dfs = [df_thermo,df_meso,df_psychro]
df_combine = functions.combine_dfs(list_dfs)


# In[6]:


#run the filter_seqs function
df_filter = functions.filter_seqs(df_combine)


# In[7]:


#define lists of filtered sequences and classes
seq_list = df_filter['sequence'].tolist()
class_list = df_filter['class'].tolist()


# In[9]:


#run the seq1hot function
X_data = functions.seq1hot(seq_list)


# In[10]:


#run the class1hot function
y_data = functions.class1hot(class_list)


# In[11]:


#run the save_tensor function
functions.save_tensor(X_data,'test_x.pt')
funstions.save_tensor(y_data,'test_y.pt')


# In[ ]:




