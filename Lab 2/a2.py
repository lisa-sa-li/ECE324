# Assignment 2 skeleton code
# This code shows you how to use the 'argparse' library to read in parameters

import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from dispkernel import dispKernel

# Command Line Arguments
parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
parser.add_argument('trainingfile', help='name stub for training data and label output in csv format',default="train")
parser.add_argument('validationfile', help='name stub for validation data and label output in csv format',default="valid")
parser.add_argument('numtrain', help='number of training samples',type= int,default=200)
parser.add_argument('numvalid', help='number of validation samples',type= int,default=20)
parser.add_argument('-seed', help='random seed', type= int,default=0.00001)
parser.add_argument('-learningrate', help='learning rate', type= float,default=15)
parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'], default='sigmoid')
parser.add_argument('-numepoch', help='number of epochs', type= int,default=200)

args = parser.parse_args()

traindataname = args.trainingfile + "data.csv"
trainlabelname = args.trainingfile + "label.csv"

print("training data file name: ", traindataname)
print("training label file name: ", trainlabelname)

validdataname = args.validationfile + "data.csv"
validlabelname = args.validationfile + "label.csv"

print("validation data file name: ", validdataname)
print("validation label file name: ", validlabelname)

print("number of training samples = ", args.numtrain)
print("number of validation samples = ", args.numvalid)

print("learning rate = ", args.learningrate)
print("seed = ", args.seed)
print("number of epoch = ", args.numepoch)

print("activation function is ",args.actfunction)


class NeuronClassifier:
    def __init__(self, bias, weights, activation):
        self.activation = activation
        self.bias = bias
        self.weights = weights

    def __call__(self, input):
        b = np.dot(input, self.weights) + self.bias

        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-b))
        elif self.activation == "relu":
            return np.where(b > 0, b, 0)
        elif self.activation == "linear":
            return b
        else:
            return False

# Load data in from CSV files
trainingdata = np.loadtxt(traindataname, delimiter=',')
trainingdatalabel = np.loadtxt(trainlabelname, delimiter=',')
validationdata = np.loadtxt(validdataname, delimiter=',')
validationdatalabel = np.loadtxt(validlabelname, delimiter=',')

# Set random values to bias and weights
random.seed(args.seed)
bias = random.random()
weight = []
for i in range(len(trainingdata[0, :])):
    weight.append(random.random())
neuron = NeuronClassifier(bias, weight, args.actfunction)
loss_array = np.zeros([args.numepoch, 2])
accuracy_array = np.zeros([args.numepoch, 2])

for i in range(args.numepoch):
    accuracy_train = 0
    accuracy_valid = 0
    loss_train = 0
    loss_valid = 0
    store = []
    for j in range(args.numtrain):
        train = neuron(trainingdata[j, :])
        store.append(train)
        loss_train += (train - trainingdatalabel[j]) ** 2
        if train.round() == trainingdatalabel[j]:
            accuracy_train += 1

        for m in range(len(trainingdata[0])):
            if args.actfunction == "linear":
                d_loss = 2*(train - trainingdatalabel[j]) * trainingdata[j][m]
                d_b = 2*(train - trainingdatalabel[j])
            elif args.actfunction == "sigmoid":
                d_loss = 2*(train - trainingdatalabel[j]) * trainingdata[j][m] * store[j] * (1-store[j])
                d_b = 2*(train - trainingdatalabel[j]) * store[j] * (1-store[j])
            elif args.actfunction == "relu":
                d_loss = 2*(train - trainingdatalabel[j]) * trainingdata[j][m]
                d_b = 2*(train - trainingdatalabel[j])
        
            neuron.weights[m] = neuron.weights[m] - d_loss * args.learningrate
        
        neuron.bias = neuron.bias - d_b * args.learningrate

    for k in range(args.numvalid):
        valid = neuron(validationdata[k, :])
        loss_valid += (valid - validationdatalabel[k]) ** 2
        if valid.round() == validationdatalabel[k]:
            accuracy_valid += 1

    loss_array[i, :] = [loss_train / args.numtrain, loss_valid / args.numvalid] 
    accuracy_array[i, :] = [accuracy_train / args.numtrain, accuracy_valid / args.numvalid]

print("accuracy_array", accuracy_array[args.numepoch-1])
print("loss_array", loss_array[args.numepoch-1])

kernel = np.array(neuron.weights)
kernel.reshape([3, 3])

plt.subplot(1, 3, 1)
plt.plot(loss_array)
plt.title("Loss over Epochs")
plt.legend(['Training', 'Validation'])
plt.subplot(1, 3, 2)
plt.plot(accuracy_array)
plt.title("Accuracy over Epochs")
plt.legend(['Training', 'Validation'])
plt.subplot(1, 3, 3)
dispKernel(kernel, 3, 3)