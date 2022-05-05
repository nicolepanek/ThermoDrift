import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import Bio
from Bio import Seq
from Bio import SeqIO
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

#import ipdb
sys.path.append('/usr/lusers/aultl/ThermoDrift/')
import thermodrift_model

#DATA LOADING FUNCTIONS

#the fasta_to_classified_df function; which inputs fasta seqs and classifies them in a df
def fasta_to_classified_df(fasta_path):
    seq_dict = {}  #define empty dict to store sequence ids and sequences
    identifiers = []   #define empty id list 
    sequence = []   #define empty seq list 
    for seq_record in SeqIO.parse(fasta_path, 'fasta'):  # (generator)
        identifiers.append(str(seq_record.id))    #append ids to id list
        sequence.append(str(seq_record.seq))    #append seqs to seq list
        seq_dict[str(seq_record.id)] = str(seq_record.seq)    #define an ID, seq dictionary
    seq_list = list(seq_dict.items())  #enumerate the dictionary
    df_seqs = pd.DataFrame(seq_list)    #create a df from enumerated dictionary
    df_seqs.columns = ['protein','sequence']    #define column names
    return df_seqs

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
    return X_data


##########################################
## temporary model pass to demo the GUI ##
##########################################

# def forward_pass(data):
#     '''
#     Input data in shape [N,L,25]
#     will process data through the model and then predict
#     '''
#     #Load model from saved outputs
#     model_save_path = '/usr/lusers/aultl/ThermoDrift/thermodrift_model.py'
#     model = thermodrift_model.Net()
#     if os.path.isfile(model_save_path):
#         model.load_state_dict(torch.load(PATH))
    
#     outputs = model(data.unsqueeze(1))
#     predicted = torch.max(outputs.data, 1)[1]
#     raw_out = outputs.data
#     return predicted, raw_out

def forward_pass_analysis(x, y):
    '''
    Input data in shape [N,L,25]
    will process data through the model and then predict
    '''
    #Load model from saved outputs
    model_out = []
    model_save_path = '/gscratch/stf/jgershon/experiments/medium_widthv8/save_model/model_1500.pt'
    model = thermodrift_model.Net()
    
    model.load_state_dict(torch.load(model_save_path))
    print('x shape ', x.shape, 'y shape ', y.shape)
    
    ### named tuple for organized data set output ###
    for i in range(x.shape[0]): 
        if i%100==0:
            print('Now running example ',i)
        outputs = model(x[i].unsqueeze(1))
        predicted = torch.max(outputs.data, 1)[1]
        raw_out = outputs.data

        pred = str([x.item() for x in predicted])
        raw = str([x.item() for x in raw_out])
        true_label = str([x.item() for x in y[i]])
        s = 'pred: ' + pred + ' raw ' + raw + ' true_label ' + true_label + "\n"
        model_out.append(s)

    return model_out


def main(path):
	#Dataloading functions:

    # define classification dictionary
    class_dict = {0: 'thermophile', 1: 'mesophile', 2: 'psychrophile'}
    
	#run the fasta_to_classified_df function
    df_user = fasta_to_classified_df(path)
    #run the filter_seqs function
    df_filter = filter_seqs(df_user)
    #extract the sequences from the df
    seq_list = df_filter['sequence'].tolist()
    #run the seq1hot function
    X_data = seq1hot(seq_list)

    #forward pass
    predicted, raw_out = forward_pass(X_data)

    predictions = []
    for i in range(predicted.size()[0]):
        pred = class_dict[int(predicted[i])]
        predictions.append(pred)

    class_0_prob = []
    class_1_prob = []
    class_2_prob = []
    for i in range(raw_out.size()[0]):
        class_0_prob.append(float(raw_out[i,0]))
        class_1_prob.append(float(raw_out[i,1]))
        class_2_prob.append(float(raw_out[i,2]))

    df_user['prediction'] = predictions
    df_user['thermophile probability'] = class_0_prob
    df_user['mesophile probability'] = class_1_prob
    df_user['psychrophile probability'] = class_2_prob
        
    return df_user
		

