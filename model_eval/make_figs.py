#!/usr/bin/env python 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import ipdb

# working directory
p = Path("/usr/lusers/aultl/ThermoDrift/model_eval")

df_test = pd.read_csv(p/"datasets"/"processed_analysis_test_data.csv") 
df_train = pd.read_csv(p/"datasets"/"processed_analysis_train_data.csv")

# graph distribution of classification probabilities
fig, ax = plt.subplots(2,2, figsize=(10,20))
for i, df in enumerate([df_test, df_train]):
    # label df type
    if i==0:
        df_type="test"
    else:
        df_type="train"

    # generate kde distribution plots
    for col in ["thermo_prob", "meso_prob", "psychro_prob"]:
        df[col].plot(kind='density', label=col, ax=ax[0, i])
        plt.xlabel("probability of model classification")
        plt.legend()
        ax[0,i].set_title(f"{df_type} distribution of classification probabilities")

    # generate heatmap plots
    df = df.drop(columns="predicted").groupby("true_class").agg("mean")
    sns.heatmap(df, annot=True, linewidths=.5, cmap="YlGnBu", ax=ax[1, i])
    ax[1, i].set_title(f"{df_type} classfication probabilities")

plt.tight_layout()
plt.savefig(p/"figs"/"analysis_plots.png", dpi=600)



