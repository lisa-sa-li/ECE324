import torch
import torch.optim as optim
from time import time

import torchtext
from torchtext import data
import spacy
import matplotlib.pyplot as plt

import argparse
import os

from models import *

def load_model(args, vocab):
    if args.model == 'baseline':
        model = Baseline(args.emb_dim, vocab) 
    elif args.model == 'cnn':
        model = CNN(args.emb_dim, vocab, args.num_filt, (2, 4))
    elif args.model == 'rnn':
        model = RNN(args.emb_dim, vocab, args.rnn_hidden_dim)
    
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
 
    return model, loss_func, optimizer

def evaluate(model, iter_data, loss_func):
    acc = 0.0
    loss = 0.0
    total_count = 0.0

    for i, batch in enumerate(iter_data):
        batch_input, batch_input_len = batch.text
        labels = batch.label.float()

        predict = model(batch_input, batch_input_len)
        predict = predict.float()

        acc += int(((predict > 0.5).squeeze().float() == labels).sum())
        loss += loss_func(predict.squeeze(),labels).item()
        total_count += len(batch_input_len)

    return acc/total_count, loss/float(i+1)

def main(args):
    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
            path='data/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
      sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    TEXT.build_vocab(train_data, val_data, test_data)
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:",TEXT.vocab.vectors.shape)

    model, loss_func, optimizer = load_model(args, vocab)
     
    running_loss = []
    running_accuracy = []
    running_valid_loss = []
    running_valid_accuracy = []
    nRec = []

    # overfit_loss = []
    # overfit_accuracy = []

    start = time()
    # Training loop
    for epoch in range(args.epochs):
        train_acc = 0.0
        train_loss = 0.0
        total_count = 0.0

        # for m, batch in enumerate(overfit_iter):
        #     batch_input, batch_input_len = batch.text
        #     labels = batch.label.float()
 
        #     optimizer.zero_grad()
        #     predict = model(batch_input, batch_input_len)
        #     predict = predict.float()
 
        #     loss = loss_func(predict.squeeze(), labels)
        #     loss.backward()
        #     optimizer.step()

        #     train_acc += int(((predict > 0.5).squeeze().float() == labels).sum())
        #     total_count += args.batch_size
        #     train_loss += loss.item()

        # overfit_accuracy.append(train_acc/total_count)
        # overfit_loss.append(train_loss/float(m+1))


        for i, batch in enumerate(train_iter):
            batch_input, batch_input_len = batch.text
            labels = batch.label.float()
 
            optimizer.zero_grad()
            predict = model(batch_input, batch_input_len)
            predict = predict.float()
 
            loss = loss_func(predict.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_acc += int(((predict > 0.5).squeeze().float() == labels).sum())
            total_count += args.batch_size
            train_loss += loss.item()
            
        running_accuracy.append(train_acc/total_count)
        running_loss.append(train_loss/float(i+1)) 

        # Validation 
        valid_acc, valid_loss = evaluate(model, val_iter, loss_func)
        running_valid_accuracy.append(valid_acc)
        running_valid_loss.append(valid_loss) 
        nRec.append(epoch)
        
        if epoch % args.eval_every == args.eval_every-1:
            print("Epoch: %d | Training accuracy: %f | Training loss: %f | Val accuracy: %f | Val loss: %f"
              % (epoch, running_accuracy[-1], running_loss[-1], running_valid_accuracy[-1], running_valid_loss[-1]))
            # print("Overfit accuracy: %f | Overfit loss: %f"
            #     % (overfit_accuracy[-1], overfit_loss[-1]))
        
        if valid_acc == max(running_valid_accuracy):
            torch.save(model, 'model_%s.pt' % args.model)
            test_accuracy, test_loss = evaluate(model, test_iter, loss_func)
    
    end = time()
    print("====FINAL VALUES====")
    print("Test accuracy: %f | Test loss: %f" % (test_accuracy, test_loss))  
    print("Training acc: %f | Valid acc: %f | Time: %f" % (max(running_accuracy), max(running_valid_accuracy), end - start))
    print("Training loss: %f | Valid loss: %f " % (min(running_loss), min(running_valid_loss)))
    # print("overfit acc: %f | overfit loss: %f" % (max(overfit_accuracy), min(overfit_loss)))

    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    ax.plot(nRec, running_loss, label='Training')
    ax.plot(nRec,running_valid_loss, label='Validation')
    plt.title('Training Loss vs. epoch')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax.legend()

    bx = plt.subplot(1, 2, 2)
    bx.plot(nRec, running_accuracy, label='Training')
    bx.plot(nRec,running_valid_accuracy, label='Validation')
    plt.title('Training Accuracy vs. epoch')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    bx.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)