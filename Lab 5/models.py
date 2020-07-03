import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # x = [sentence length, batch size]
        embedded = self.embedding(x)

        average = embedded.mean(0) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)

        return output

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim)).float()
        self.conv2 = nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim)).float()
        self.fc1 = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(1)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x1, _ = torch.max(x1, 2)
        x2, _ = torch.max(x2, 2)

        x = torch.cat([x1, x2], 1).squeeze()
        x = torch.sigmoid(self.fc1(x))

        return x

class RNN(nn.Module):

    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        _, x = self.gru(x)
        x = torch.sigmoid(self.fc1(x))

        return x