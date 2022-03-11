#Temporary "model" function to test that our GUI is functional
import pandas as pd
import numpy as np
import Bio
from Bio.Seq import Seq
from Bio import SeqIO


def temp_model(fasta_file):

    '''
    This function was written to test the functionality of our dash GUI
    It takes a .fasta file, imports the sequence into a Pd dataframe and
    returns the dataframe

    inputs:
    - fasta_file_path: str; path to .fasta file with protein sequence
    '''

    #read in FASTA file with Thermophllic proteins datasets

    thermo_dict = {}
    
    identifiers = []
    sequence = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        identifiers.append(str(seq_record.id))
        sequence.append(str(seq_record.seq))
        thermo_dict[str(seq_record.id)] = str(seq_record.seq)

    thermo_list = list(thermo_dict.items())
    df_thermo = pd.DataFrame(thermo_list)
    df_thermo.columns = ['protein','sequence']

    return df_thermo

#--------------------------------------------------------------------------------


def temp_model_2(fasta_file_path):

    '''
    This function was written to test the functionality of our dash GUI
    It takes the file path to a .fasta file, imports the sequence into a Pd dataframe and
    returns the dataframe

    inputs:
    - fasta_file_path: str; path to .fasta file with protein sequence
    '''

    #read in FASTA file with Thermophllic proteins datasets

    thermo_dict = {}
    with open(fasta_file_path) as thermo_fasta_file:  # Will close handle cleanly
        identifiers = []
        sequence = []
        for seq_record in SeqIO.parse(thermo_fasta_file, 'fasta'):  # (generator)
            identifiers.append(str(seq_record.id))
            sequence.append(str(seq_record.seq))
            thermo_dict[str(seq_record.id)] = str(seq_record.seq)

    thermo_list = list(thermo_dict.items())
    df_thermo = pd.DataFrame(thermo_list)
    df_thermo.columns = ['protein','sequence']

    #convert to csv and save in "Examples" folder
    #If this were the real model, it would return a .csv file 
	#with the classification probabilities
	
    #df_thermo.to_csv('temp_out.csv', index=True)
    return df_thermo



#-----------------------------------------------------------------------------------
#Example
#-----------------------------------------------------------------------------------

#df = temp_model_2('Examples/2AYQ.fasta')
#print(df)


