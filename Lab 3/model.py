import torch.nn as nn
import torch.nn.functional as F
import torch

neurons = 64

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, neurons)
        self.fc2 = nn.Linear(neurons, 1)


    def forward(self, features):
        x = F.relu(self.fc1(features.float()))
        x = F.sigmoid(self.fc2(x))
        return x
