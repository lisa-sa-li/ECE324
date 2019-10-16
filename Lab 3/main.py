import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# load data set
data = pd.read_csv('data/adult.csv')

# 3.2 data visualization
# print("shape: ", data.shape)
# print("column names: ", data.columns)
# print("first 5 rows: ", data.head())

# print("the number of occurrences of each value of the column: ", data["income"].value_counts() )
# print(">50K: ", data["income"].value_counts()[0] )
# print("<=50K: ", data["income"].value_counts()[1] )

# data cleaning

# missing entries there are for each feature
# print(data.isin(["?"]).sum())

col_names = data.columns
num_rows = data.shape[0]

for feature in col_names:
    # 3.3 throw out all rows (samples) with 1 or more "?"
    data = data[data[feature] != "?"]

# print("clean data: ", data)
# print("new shape: ", data.shape)
# print("rows deleted = ", num_rows - data.shape[0])


# 3.4 balance dataset
new_data_size = min(data[data["income"] == ">50K"].shape[0], data[data["income"] == "<=50K"].shape[0])
data1 = data[data["income"] == ">50K"].sample(n=new_data_size, random_state=seed)
data2 = data[data["income"] == "<=50K"].sample(n=new_data_size, random_state=seed)
data = pd.concat([data1, data2], ignore_index=True)
# print("dropped data", data)
# print("dropped data shape", data.shape)


# 3.5 data visualization
# print("describe data:")
# verbose_print(data.describe())

categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                     'relationship', 'gender', 'native-country', 'income']
continuous_feats = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# print("value :")
# for feature in categorical_feats:
#     print(data[feature].value_counts())

# visualize the first 3 features using pie and bar graphs

# for feature in categorical_feats:
#     pie_chart(data, feature)
#     binary_bar_chart(data, feature)


# 3.6 normalize continuous features
data_cts = data[continuous_feats].copy()
data_cat = data[categorical_feats].copy()

# print("data_cat", data_cat)


for feature in continuous_feats:
    data_cts.loc[:,feature] = (data_cts.loc[:,feature] - data_cts.loc[:,feature].mean()) /  data_cts.loc[:,feature].std()

# numpy representation of the data
data.to_numpy()
# print("normalized numpy data ", data)

#3.6 ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()
oneh_encoder = OneHotEncoder(categories="auto")

for feature in categorical_feats:
    data_cat[feature] = label_encoder.fit_transform(data_cat[feature])
# print("data_cat", data_cat)

# remove income column and store in new variable
income_feature = data_cat["income"]
income_feature = income_feature.to_numpy()

data_cat = data_cat.drop(columns="income")
# print("no income data", data)
categorical_feats2 = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                     'relationship', 'gender', 'native-country']
one_hot = []
for feature in categorical_feats2:
    one_hot.append(oneh_encoder.fit_transform(data_cat[feature].values.reshape(-1, 1)).toarray())

one_hot = np.concatenate(one_hot, axis=1)
# print("one hot", one_hot)
# print("one hot shape", one_hot.shape)


data = np.concatenate((one_hot, data_cts), axis=1)
# print("final data shape", data.shape)


# 3.7 Split the data
test_size = 0.2
x_train, x_val, y_train, y_val = train_test_split(data, income_feature, test_size=test_size, random_state=seed)
# print("x_train", x_train)
# print("x_val", x_val)
# print("y_train", y_train)
# print("y_val", y_val)

# 4.2
def load_data(batch_size):
    train_data = AdultDataset(x_train, y_train)
    val_data = AdultDataset(x_val, y_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=len(x_val), shuffle=True)

    return train_loader, val_loader


def load_model(lr):
    model = MultiLayerPerceptron(len(x_train[0]))
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer


def evaluate(model, valid_loader):
    total_corr = 0
    for i, vbatch in enumerate(valid_loader):
        feats, label = vbatch 
        prediction = model(feats)
        corr = (prediction > 0.5).squeeze().long() == label
        total_corr += int(corr.sum())
    return float(total_corr)/len(valid_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--eval_every', type=int, default=200)

    args = parser.parse_args()

    train_loader, val_loader = load_data(args.batch_size)
    model, loss_function, optimizer = load_model(args.lr)

    nRec = []
    timeRec = []
    trainAccRec = []
    validAccRec = [] 
    train_accuracy = 0

    t = 0
    start = time()

    for epoch in range(args.epochs):
        accum_loss = 0
        tot_corr = 0

        for i_batch, sample_batched in enumerate(train_loader):
            feats, label = sample_batched

            optimizer.zero_grad()
            predict = model(feats)
            loss = loss_function(input=predict.squeeze(), target=label.float())

            accum_loss += loss
            
            loss.backward()
            optimizer.step()

            corr = (predict > 0.5).squeeze().long() == label
            tot_corr += int(corr.sum())

            for k in range(len(predict)):
                if round(predict[k].item()) == label[k]:
                    train_accuracy += 1

            if (t+1) % args.eval_every == 0:
                train_acc = float(train_accuracy) / (args.batch_size * args.eval_every)
                trainAccRec.append(train_acc)
                valid_acc = evaluate(model, val_loader)
                validAccRec.append(valid_acc)
                nRec.append(t+1)
                timeRec.append(time() - start)

                print("Epoch: {}, Step {} | Loss: {}| Valid acc: {}".format(epoch+1, t+1, accum_loss / args.eval_every, valid_acc))
                accum_loss = 0
                train_accuracy = 0
            t += 1

    end = time()

    print("Train acc: {}".format(float(tot_corr)/len(train_loader.dataset)))
    print("Max valid acc: ", max(validAccRec))
    print("Time: ", end - start)
    
    plt.subplot(1, 2, 1)
    plt.plot(nRec, savgol_filter(trainAccRec, 13, 3), label='Train')
    plt.plot(nRec, savgol_filter(validAccRec, 13, 3), label='Validation')
    plt.title('Training and Validation Accuracy vs. # of gradient steps')
    plt.xlabel('number of gradient steps')
    plt.ylabel('accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(timeRec, savgol_filter(trainAccRec, 13, 3), label='Train')
    plt.plot(timeRec, savgol_filter(validAccRec, 13, 3), label='Validation')
    plt.title('Training and Validation Accuracy vs. time')
    plt.xlabel('time (s)')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
