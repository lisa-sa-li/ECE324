import torch
import torch.optim as optim
from time import time

import torchtext
from torchtext import data
import spacy
import readline
import matplotlib.pyplot as plt

import argparse
import os

from models import *


model_name = ['baseline', 'cnn', 'rnn']

def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(text)]

def main():

    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)
    train_data= data.TabularDataset('data/train.tsv', format='tsv', skip_header=True, fields=[('text', TEXT), ('label', LABELS)])
    TEXT.build_vocab(train_data)
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    baseline_model = torch.load("model_baseline.pt")
    rnn_model = torch.load("model_rnn.pt")
    cnn_model = torch.load("model_cnn.pt")

    models = [baseline_model, rnn_model, cnn_model]

    while True:
        print("\n---------------------")
        print("Enter a sentence:")
        sentence = input()

        if sentence == "exit":
            print("~~~Byeeeeee~~~")
            break
    
        tokens = tokenizer(sentence)
        token_ints = [vocab.stoi[tok] for tok in tokens]
        token_tensor = torch.LongTensor(token_ints).view(-1,1) # Shape is [sentence_len, 1]
        lengths = torch.Tensor([len(token_ints)])


        for i, model in enumerate(models):
            predict = model(token_tensor, lengths)

            if i == 0: # baseline
                predict = torch.sigmoid(predict)
            
            predict = predict.detach().numpy()

            result = 'subjective' if predict > 0.5 else 'objective'
            print('Model %s: %s (%.3f)' % (model_name[i], result, predict))


if __name__ == '__main__':
    main()