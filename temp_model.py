#Temporary "model" to test that our GUI is functional
import pandas as pd
import numpy as np
import Bio
from Bio.Seq import Seq
from Bio import SeqIO

#read in FASTO file

#import Thermophllic proteins datasets
thermo_dict = {}
with open('uniprot-thermophilus.fasta') as thermo_fasta_file:  # Will close handle cleanly
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