import os
import torch.nn as nn
import torch.optim as optim
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inference_script import forward_pass_analysis
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import csv
from pathlib import Path
import ipdb


def load_data(path):
    """
    Loads tensor data

    :param path: str - path to tensor data 
    :return loaded: loaded tensor data
    """
    loaded = torch.load(path)
    return loaded


# set working directory containing tensors
tensor_dir = Path("/gscratch/stf/jgershon")

# load tensors generated with new aa composition feature
tensors = {}
for tensor in tensor_dir.glob("*aa_comp.pt"):
    print(tensor.name)
    name = tensor.name.split("_aa_comp.pt")[0]
    tensors[name] = load_data(tensor)

model_out = forward_pass_analysis(
    tensors['X_train'], tensors['y_train'], aa_comp=True)
with open('/usr/lusers/aultl/ThermoDrift/model_eval/20220605_analysis_train_aa_comp.csv', "w") as f:
    writer = csv.writer(f)
    header = ["predicted", "raw_probabilities", "true_label"]
    writer.writerow(header)
    f.writelines(model_out)

model_out = forward_pass_analysis(
    tensors['X_test'], tensors['y_test'], aa_comp=True)
with open('/usr/lusers/aultl/ThermoDrift/model_eval/20220605_analysis_test_aa_comp.csv', "w") as f:
    writer = csv.writer(f)
    header = ["predicted", "raw_probabilities", "true_label"]
    writer.writerow(header)
    f.writelines(model_out)
