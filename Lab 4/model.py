import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5)

        self.fc1 = nn.Linear(8*11*11, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
       

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8*11*11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

class ImgDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class ConvNN_pt4(nn.Module):
    def __init__(self):
        super(ConvNN_pt4, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 30, 3)

        self.fc1 = nn.Linear(30*12*12, 32)
        self.fc2 = nn.Linear(32, 10)
       
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30*12*12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class ConvNN_pt5(nn.Module):
    def __init__(self):
        super(ConvNN_pt5, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, 3)
        self.conv1_BN = nn.BatchNorm2d(30)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 30, 3)

        self.fc1 = nn.Linear(30*12*12, 32)
        self.fc2 = nn.Linear(32, 10)
        self.fc_BN = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_BN(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30*12*12)
        x = F.relu(self.fc_BN(self.fc1(x)))
        x = self.fc2(x)
        return x

class ConvNN_pt6(nn.Module):
    def __init__(self):
        super(ConvNN_pt6, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.conv1_BN = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_BN = nn.BatchNorm2d(50)
        self.conv2 = nn.Conv2d(20, 50, 3)
        self.conv3 = nn.Conv2d(50, 100, 5)

        self.fc1 = nn.Linear(100*4*4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc_BN = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_BN(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_BN(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,100*4*4)
        x = F.relu(self.fc_BN(self.fc1(x)))
        x = self.fc2(x)
        return x

class ConvNN_pt6_small(nn.Module):
    def __init__(self):
        super(ConvNN_pt6_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, 3)
        self.conv1_BN = nn.BatchNorm2d(2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_BN = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(2, 5, 3)
        self.conv3 = nn.Conv2d(5, 10, 5)

        self.fc1 = nn.Linear(10*4*4, 3)
        self.fc2 = nn.Linear(3, 10)
        self.fc_BN = nn.BatchNorm1d(3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_BN(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_BN(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,10*4*4)
        x = F.relu(self.fc_BN(self.fc1(x)))
        x = self.fc2(x)
        return x