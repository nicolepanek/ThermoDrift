#!/usr/bin/env python 

import pandas as pd
import csv
from pathlib import Path

# working directory
p = Path("/usr/lusers/aultl/ThermoDrift/model_eval")

# load tensor classification dict
tensor_class_dict = (pd.read_csv(p/"datasets"/"tensor_classifications.csv", 
                                header=None, index_col=0, squeeze=True)
                     .to_dict())

# load model test and train set analysis
# df_test = pd.read_csv(p/"datasets"/"20220605_analysis_test.csv", skiprows=1, 
#                       usecols=[0,1,2,3,4])
df_train = pd.read_csv(p/"datasets"/"20220605_analysis_train.csv", skiprows=1, 
                       usecols=[0,1,2,3,4])


for df in [df_train]:#, df_train]:
    df.columns=["predicted", "thermo_prob", "meso_prob", "psychro_prob", "true_class"]

    # reformat probability strings 
    ##### go back to analysis code to clean this up
    df.loc[:, "thermo_prob"] = [x.split("[")[2] for x in df["thermo_prob"]]
    df.loc[:, "psychro_prob"] = [x.split("]")[0] for x in df["psychro_prob"]]
    df.loc[:, "predicted"] = [x.split("[")[1].split("]")[0] for x in df["predicted"]]
    df.loc[:, "true_class"] = [x.split("[")[1].split("]")[0] for x in df["true_class"]]

    # translate tensor labels to classifications
    df.loc[:, "predicted"] = [tensor_class_dict[x] for x in df.predicted]
    df.loc[:, "true_class"] = [tensor_class_dict[x] for x in df.true_class]

    # probabilities data type = float 
    df[["thermo_prob", "meso_prob", "psychro_prob"]] = df[["thermo_prob", "meso_prob", "psychro_prob"]].astype(float)

# save dataframe
#df_test.to_csv(p/"datasets"/"processed_analysis_train_data.csv", index=False)
df_train.to_csv(p/"datasets"/"processed_analysis_train_data.csv", index=False)




