#Temporary "model" to test that our GUI is functional
import pandas as pd
import numpy as np
import Bio
from Bio.Seq import Seq
from Bio import SeqIO

#read in FASTO file with Thermophllic proteins datasets

thermo_dict = {}
with open('Examples/2AYQ.fasta') as thermo_fasta_file:  # Will close handle cleanly
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
	
df_thermo.to_csv('Examples/temp_out.csv', index=True)
print(df_thermo)
