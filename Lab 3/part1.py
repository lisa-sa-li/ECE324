import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import math
import numpy as np

import torch  
import torch.nn as nn

# Command Line Arguments
trainingfile = "train"
validationfile = "valid"
inputSize = 9
seed = 1
learningrate = 0.001
actfunction = "linear"
numepoch = 50

print("Activation function is",actfunction)

torch.manual_seed(seed)

# Function to read in data and labels from files and combine into a single structure

def readDataLabels(datafile, labelfile):
  dataset = np.loadtxt(datafile, dtype=np.single, delimiter=',')
  labelset = np.loadtxt(labelfile, dtype=np.single, delimiter=',')

  return dataset, labelset

# Read Training Data

tdataname = trainingfile + "data.csv"
tlabelname = trainingfile + "label.csv"

Td, Tl = readDataLabels(tdataname,tlabelname)

numberOfTrainingSamples = len(Td)
print("Number of Training Samples=",numberOfTrainingSamples)
print("Shape of Td=",Td.shape)
print("Shape of Tl=",Tl.shape)

print("Training Data:", Td)
print("Training Labels:", Tl)

# Read Validation Data
vdataname = validationfile + "data.csv"
vlabelname = validationfile + "label.csv"

Vd, Vl = readDataLabels(vdataname,vlabelname)

numberOfValidationSamples = len(Vd)
print("Number of Validation Samples=",numberOfValidationSamples)

print("Validation Data:", Vd)
print("Validation Labels:", Vl)


class SNC(nn.Module):
    def __init__(self):
        super(SNC, self).__init__()
        self.fc1 = nn.Linear(9 , 1)

    def forward(self, I):
        x = self.fc1(I)
        return x

def accuracy(predictions,label):

    total_corr = 0

    index = 0
    for c in predictions.flatten():
        if (c.item() > 0.5):
            r = 1.0
        else:
            r = 0.0
        if (r == label[index].item()):
            total_corr += 1
        index +=1

    return (total_corr/len(label))

smallNN = SNC()

print("Parameter Names and Initial (random) values: ")
for name, param in smallNN.named_parameters():
    print("name:",name, "value:", param)


Tdata = torch.from_numpy(Td)
print("Shape/Size of Tdata = ", Tdata.size())
Tlabel = torch.from_numpy(Tl)
print("Shape/Size of Tlabel = ", Tlabel.size())

Vdata = torch.from_numpy(Vd)
print("Shape/Size of Vdata = ", Vdata.size())
Vlabel = torch.from_numpy(Vl)
print("Shape/Size of Vlabel = ", Vlabel.size())


# print ("Tlabel[0]=", Tlabel[0])
# print ("Tlabel[0].item=", Tlabel[0].item())
# print ("Vlabel[0]=", Vlabel[0])
# print ("Vlabel[0].item=", Vlabel[0].item())


predict = smallNN(Tdata)
print("Shape of Predict", predict.size())
print("Accuracy of Tdata: ", accuracy(predict,Tlabel))


print(smallNN.fc1.weight.grad)
loss_function = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(smallNN.parameters(),lr=learningrate)

#  Store values for plots at end
lossRec = []
vlossRec = []
nRec = []
trainAccRec = []
validAccRec = [] 


for i in range(numepoch):
    optimizer.zero_grad()
    predict = smallNN(Tdata)
    loss = loss_function(input=predict.squeeze(), target=Tlabel.float())
    loss.backward()  
    optimizer.step()  
    trainAcc = accuracy(predict,Tlabel)

    predict = smallNN(Vdata)
    vloss = loss_function(input=predict.squeeze(), target=Vlabel.float())
    validAcc = accuracy(predict, Vlabel)    

    print("loss: ", f'{loss:.4f}'," trainAcc: ", f'{trainAcc:.4f}', " validAcc: ", f'{validAcc:.4f}')
    
    lossRec.append(loss)
    vlossRec.append(vloss)
    nRec.append(i)
    trainAccRec.append(trainAcc)
    validAccRec.append(validAcc)


plt.subplot(1, 2, 1)
plt.plot(nRec,lossRec, label='Train')
plt.plot(nRec,vlossRec, label='Validation')
plt.title('Training and Validation Loss vs. epoch')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(1, 2, 2)
plt.plot(nRec,trainAccRec, label='Train')
plt.plot(nRec,validAccRec, label='Validation')
plt.title('Training and Validation Accuracy vs. epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


print("Model Weights")
for name, param in smallNN.named_parameters():
    print("name:",name, "value:", param)
    




