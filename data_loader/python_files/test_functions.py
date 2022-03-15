#!/usr/bin/env python
# coding: utf-8

# In[60]:


#import modules
import pandas as pd
import numpy as np
import Bio
from Bio import Seq
from Bio import SeqIO
import torch
from torch.utils.data import TensorDataset, DataLoader
import unittest
import functions


# In[64]:


# Define a class in which the tests will run
class TestDataLoader(unittest.TestCase):
    
    #define data file paths for test fasta files which contain 10 examples each
    thermo_path_test = '../data/thermo_test.fasta'
    psychro_path_test = '../data/psychro_test.fasta'
    meso_path_test = '../data/meso_test.fasta'
    
    #define a test for the fasta_to_classified_df function
    def test_fasta_to_classified_df(self):
        #run the fasta_to_classified_df function on each test fasta file
        self.df_thermo = functions.fasta_to_classified_df(self.thermo_path_test,'Thermophillic')
        self.df_psychro = functions.fasta_to_classified_df(self.psychro_path_test,'Psychrophillic')
        self.df_meso = functions.fasta_to_classified_df(self.meso_path_test,'Mesophillic')
        #count the number of sequences in the thermo fasta file manually
        self.file = open(self.thermo_path_test,'r')
        self.count = 0
        for line in self.file:
            if line.startswith('>'):
                self.count = self.count+1
        #assert that the len of seqs in the fasta file (count) is equal to the length of the df (rows)
        self.assertEqual(self.df_thermo.shape[0], self.count)

     #construct dataframes for each test fasta file
    df_thermo = functions.fasta_to_classified_df(thermo_path_test,'Thermophillic')
    df_psychro = functions.fasta_to_classified_df(psychro_path_test,'Psychrophillic')
    df_meso = functions.fasta_to_classified_df(meso_path_test,'Mesophillic')
        
    def test_combine_dfs(self):
        #construct combined dataframe
        self.df_list = [self.df_thermo, self.df_psychro, self.df_meso]
        #assert that the length of the combined df is equal to the sum of the lengths of each df
        self.assertEqual(len(functions.combine_dfs(self.df_list)), sum([len(l) for l in self.df_list]))

    #construct combined dataframe
    df_combine = functions.combine_dfs([df_thermo,df_psychro,df_meso])
    
    def test_filter_seqs(self):
        #filter the combined dataframe
        self.df_filter = functions.filter_seqs(self.df_combine)
        #assert that the length of filtered sequences are greater than 75 aas
        for l in self.df_filter['sequence'].tolist(): 
            self.assertTrue(len(l)>75)
        #assert that the filtered sequences start with M
        for l in self.df_filter['sequence'].tolist(): 
            self.assertTrue(l.startswith('M'))
    
    #filter the combined dataframe
    df_filter = functions.filter_seqs(df_combine)
    
    def test_seq1hot(self):
        self.X_test = functions.seq1hot(self.df_filter['sequence'].tolist())
        #assert that the 1hot encoded sequences have the same length (number of seqs) as df_filter
        self.assertTrue(self.X_test.shape[0] == self.df_filter.shape[0])
        #assert that the 1hot encoded sequences have been cropped and padded to be 500 aas long
        self.assertEqual(self.X_test.shape[1],500)
        #assert that the 1hot encoded sequences have 25 different aas and characters
        self.assertEqual(self.X_test.shape[2],25)  
               
    #1hotencode the seqs and classes
    X_data = functions.seq1hot(df_filter['sequence'].tolist())
    y_data = functions.class1hot(df_filter['class'].tolist())  
    
    def test_class1hot(self):
        self.y_test = functions.class1hot(self.df_filter['class'].tolist())
        #assert that the 1hot encoded classes have the same length (number of classes) as df_filter
        self.assertTrue(self.y_test.shape[0] == self.df_filter.shape[0])
        #assert that the 1hotencoded classes have three possible classes
        self.assertEqual(self.y_test.shape[1], 3)


# In[65]:


suite = unittest.TestLoader().loadTestsFromTestCase(TestDataLoader)
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)


# In[ ]:





# In[ ]:




