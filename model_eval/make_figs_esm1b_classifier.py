#!/usr/bin/env python 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import ipdb

# working directory
p = Path("/usr/lusers/aultl/ThermoDrift/model_eval")

df = pd.read_csv(p/"datasets"/"processed_analysis_test_esm1b_classifier.csv") 
df["true_class"] = pd.CategoricalIndex(df["true_class"], categories=["Psychrophile", "Mesophile", "Thermophile"], ordered=True)


# graph distribution of classification probabilities
# fig, ax = plt.subplots(1,2, figsize=(8,6))
# # generate kde distribution plots
# for col in ["thermo_prob", "meso_prob", "psychro_prob"]:
#     df[col].plot(kind='density', label=col, ax=ax[0], legend=True)

# ax[0].set_xlabel("probability of model classification", fontsize=13)
# plt.legend()

# generate heatmap plots

df = df.drop(columns="predicted").groupby("true_class").agg("mean")
df.index = (pd.CategoricalIndex(['Psychrophile', 'Mesophile', 'Thermophile'], 
                                categories=['Psychrophile', 'Mesophile', 'Thermophile'], 
                                ordered=True, dtype='category', name='True Class'))

df.columns = ["Thermophile", "Mesophile", "Psychrophile"]


g = sns.heatmap(df, annot=True, linewidths=.5, cmap="YlGnBu")
g.set_title("Esm1b Embedding Classifier Classification Probabilities", fontsize=14)
txt="Probability"
plt.figtext(0.5, 0.0001, txt, wrap=True, horizontalalignment='center', fontsize=11)

plt.tight_layout()
plt.savefig(p/"figs"/"220606_fig2_esm1b_classifier_heatmaponly.png", dpi=600)



