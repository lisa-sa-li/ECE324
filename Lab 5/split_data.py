import numpy as np
import pandas as pd


data = pd.read_csv('data/data.tsv', sep='\t')

# Split based on the label
data_1 = data[data['label'] == 1]
data_0 = data[data['label'] == 0]

# Find index to split at
split_1 = int(np.floor(0.64*len(data_1)))
split_2 = int(np.floor(0.16*len(data_1))) + split_1

# Split and shuffle the data
data_train = pd.concat([data_1[:split_1], data_0[:split_1]], ignore_index=True).sample(frac=1).reset_index(drop=True)
data_valid = pd.concat([data_1[split_1:split_2], data_0[split_1:split_2]], ignore_index=True).sample(frac=1).reset_index(drop=True)
data_test = pd.concat([data_1[split_2:], data_0[split_2:]], ignore_index=True).sample(frac=1).reset_index(drop=True)
data_overfit = pd.concat([data_1[:50], data_0[:50]], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Save the data
data_train.to_csv('data/train.tsv',sep='\t', index=False)
data_valid.to_csv('data/validation.tsv',sep='\t', index=False)
data_test.to_csv('data/test.tsv',sep='\t', index=False)
data_overfit.to_csv('data/overfit.tsv',sep='\t', index=False)

# print("num of each label: ", data_train["label"].value_counts(), data_valid["label"].value_counts(), data_test["label"].value_counts() )
