import argparse
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torch.utils.data as data
from torchsummary import summary

from model import ConvNN
from model import ConvNN_pt4
from model import ConvNN_pt5
from model import ConvNN_pt6
from model import ConvNN_pt6_small
from model import ImgDataset

from sklearn.metrics import confusion_matrix

# import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


# =============== Part 2 ===============

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
images_data = torchvision.datasets.ImageFolder("asl_images", transform=transform)
train_loader = DataLoader(images_data, batch_size=4, shuffle=True)

classes = ['A','B','C','D','E','F','G','H','I','K']

# Prints 4 images and their labels
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# =============== Part 3 ===============

neuron = ConvNN()

# Define hyperparameters in the assignment
batch_size = 30
lr = 0.5
numepoch = 200
loss_function = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(neuron.parameters(),lr=lr)
seed = 1
torch.manual_seed(seed)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
images_data = torchvision.datasets.ImageFolder("my_asl", transform=transform)

image_train = []
labels = []
for i in images_data:
    image_train.append(i[0].numpy())
    labels.append(i[1])

labels = pd.DataFrame(labels,columns=["letter"])

oneh_encoder = OneHotEncoder(categories="auto")
label_one_hot = oneh_encoder.fit_transform(labels["letter"].values.reshape(-1, 1)).toarray()

train_data = ImgDataset(image_train, label_one_hot)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#  Store values for plots at end
lossRec = []
nRec = []
trainAccRec = []

start = time()

for i in range(numepoch):
    running_loss = []
    running_accuracy = []
    for j, data in enumerate(train_loader):
        feats, labels = data
        optimizer.zero_grad()
        predict = neuron(feats)
        loss = loss_function(input=predict.squeeze(), target=labels.float())
        loss.backward()  
        optimizer.step()  

        _, predicted = torch.max(predict.data, 1)
        trainAcc = (labels.nonzero()[:,1] == predicted).sum().item() /batch_size

        print("loss: ", f'{loss:.4f}'," trainAcc: ", f'{trainAcc:.4f}')
        running_accuracy.append(trainAcc)
        running_loss.append(loss)

    trainAcc = sum(running_accuracy) / float(len(running_accuracy))
    loss = sum(running_loss) / float(len(running_loss))
        
    lossRec.append(loss)
    nRec.append(i)
    trainAccRec.append(trainAcc)

end = time()
summary(neuron, (3, 56, 56))
print("Time: ", end - start)

plt.subplot(1, 2, 1)
plt.plot(nRec,lossRec, label='Train')
plt.title('Training Loss vs. epoch')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(1, 2, 2)
plt.plot(nRec,trainAccRec, label='Train')
plt.title('Training Accuracy vs. epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# =============== Part 4 ===============

# Part 4.1 Data splitting 
transform_pt4 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
images_data_pt4 = torchvision.datasets.ImageFolder("asl_images", transform=transform_pt4)

data_total_len = len(images_data_pt4)

index_list = []
for i in range(data_total_len):
    index_list.append(i)

split_index = int(np.floor(0.8*data_total_len))
np.random.shuffle(index_list)

train_index = index_list[:split_index]
valid_index = index_list[split_index:]
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)

# for i in images_data:
#     image_train.append(i[0].numpy())
#     labels.append(i[1])


train_loader_pt4 = DataLoader(images_data_pt4 , batch_size=batch_size, sampler=train_sampler)
valid_loader_pt4 = DataLoader(images_data_pt4 , batch_size=len(valid_index), sampler=valid_sampler)

print("train_sampler", train_loader_pt4 )

image_train_pt4 = []
labels_train_pt4 = []
image_valid_pt4 = []
labels_valid_pt4 = []

labels = pd.DataFrame(labels,columns=["letter"])

oneh_encoder = OneHotEncoder(categories="auto")
label_one_hot = oneh_encoder.fit_transform(labels["letter"].values.reshape(-1, 1)).toarray()

train_data = ImgDataset(image_train, label_one_hot)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)



# =============== Part 4 ===============

def mean_std(dataloader):
    mean = []
    std = []
    for i, data in enumerate(dataloader):
        inputs, labels = data
        for j in range(labels.shape[0]):
            mean += [torch.mean(inputs[j])]
            std += [torch.std(inputs[j])]
    return np.mean(mean),np.mean(std)
        
def part4(batch_size, lr, epochs, eval_every, neuron, loss_function):
    print("Starting...")
    
    optimizer = torch.optim.SGD(neuron.parameters(),lr=lr)
    seed = 1
    torch.manual_seed(seed)

    # Calculate mean and std
    transform_pt4 = transforms.Compose([transforms.ToTensor()])
    images_data_pt4 = torchvision.datasets.ImageFolder("asl_images", transform=transform_pt4)
    dataloader = DataLoader(images_data_pt4, shuffle=False, batch_size=batch_size)
    mean, std = mean_std(dataloader)
    print("mean: ", mean, " std: ", std)

    transform_pt4 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(mean,mean,mean), std=(std,std,std))])
    images_data_pt4 = torchvision.datasets.ImageFolder("asl_images", transform=transform_pt4)
    dataloader = DataLoader(images_data_pt4, shuffle=False, batch_size=batch_size)
    mean, std = mean_std(dataloader)
    print("mean: ", mean, " std: ", std)

    image_train_valid = []
    labels = []
    for i in images_data_pt4:
        image_train_valid.append(i[0].numpy())
        labels.append(i[1])

    labels = pd.DataFrame(labels,columns=["letter"])

    oneh_encoder = OneHotEncoder(categories="auto")
    label_one_hot = oneh_encoder.fit_transform(labels["letter"].values.reshape(-1, 1)).toarray()

    # Part 4.1 Data splitting 
    data_total_len = len(images_data_pt4)

    index_list = []
    for i in range(data_total_len):
        index_list.append(i)

    split_index = int(np.floor(0.8*data_total_len))
    np.random.shuffle(index_list)

    train_index = index_list[:split_index]
    valid_index = index_list[split_index:]
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    train_valid_data = ImgDataset(image_train_valid, label_one_hot)
    train_loader_pt4 = DataLoader(train_valid_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader_pt4 = DataLoader(train_valid_data, batch_size=len(valid_index), sampler=valid_sampler)

    mean, std = mean_std(train_loader_pt4)
    print("training mean: ", mean, " std: ", std)

    # Part 4.3
    lossRec = []
    vlossRec = []
    nRec = []
    trainAccRec = []
    validAccRec = [] 

    start = time()

    for i in range(epochs):
        running_loss = []
        running_accuracy = []
        running_valid_loss = []
        running_valid_accuracy = []
        for j, data in enumerate(train_loader_pt4):
            feats, labels = data
            optimizer.zero_grad()
            predict = neuron(feats)
            loss = loss_function(input=predict.squeeze(), target=labels.float())
            loss.backward()  
            optimizer.step()  

            _, predicted = torch.max(predict.data, 1)
            trainAcc = (labels.nonzero()[:,1] == predicted).sum().item() / batch_size

            if j % eval_every == 0:
                print("=== epoch: ", i+1, " ===")
                print("loss: ", f'{loss:.4f}'," trainAcc: ", f'{trainAcc:.4f}')
            
            running_accuracy.append(trainAcc)
            running_loss.append(loss)

        for k, data in enumerate(valid_loader_pt4):
            feats, labels = data
            predict = neuron(feats)
            vloss = loss_function(input=predict.squeeze(), target=labels.float())
            _, predicted = torch.max(predict.data, 1)
            validAcc = (labels.nonzero()[:,1] == predicted).sum().item() / len(valid_index)

            if k % eval_every == 0:
                print("validAcc: ", f'{validAcc:.4f}')
            
            running_valid_accuracy.append(validAcc)
            running_valid_loss.append(vloss)


        trainAcc = sum(running_accuracy) / float(len(running_accuracy))
        loss = sum(running_loss) / float(len(running_loss))
        validAcc = sum(running_valid_accuracy) / float(len(running_valid_accuracy))
        vloss = sum(running_valid_loss) / float(len(running_valid_loss))
            
        lossRec.append(loss)
        vlossRec.append(vloss)
        nRec.append(i)
        trainAccRec.append(trainAcc)
        validAccRec.append(validAcc)

    end = time()
    summary(neuron, (3, 56, 56))
    print("Time: ", end - start)
    print("trainAcc: ", max(trainAccRec))
    print("validAcc: ", max(validAccRec))

    plt.subplot(1, 2, 1)
    plt.plot(nRec,lossRec, label='Train')
    plt.plot(nRec,vlossRec, label='Valid')
    plt.title('Training Loss vs. epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(nRec,trainAccRec, label='Train')
    plt.plot(nRec,validAccRec, label='Valid')
    plt.title('Training Accuracy vs. epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.ion()
    plt.show()

       
def part5(batch_size, lr, epochs, eval_every, neuron, loss_function):
    print("Starting...")
    
    optimizer = torch.optim.SGD(neuron.parameters(),lr=lr)
    seed = 1
    torch.manual_seed(seed)

    # Calculate mean and std
    transform_pt4 = transforms.Compose([transforms.ToTensor()])
    images_data_pt4 = torchvision.datasets.ImageFolder("asl_images", transform=transform_pt4)
    dataloader = DataLoader(images_data_pt4, shuffle=False, batch_size=batch_size)
    mean, std = mean_std(dataloader)

    transform_pt4 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(mean,mean,mean), std=(std,std,std))])
    images_data_pt4 = torchvision.datasets.ImageFolder("asl_images", transform=transform_pt4)
    dataloader = DataLoader(images_data_pt4, shuffle=False, batch_size=batch_size)
    mean, std = mean_std(dataloader)

    image_train_valid = []
    labels = []
    for i in images_data_pt4:
        image_train_valid.append(i[0].numpy())
        labels.append(i[1])

    data_total_len = len(images_data_pt4)

    index_list = []
    for i in range(data_total_len):
        index_list.append(i)

    split_index = int(np.floor(0.8*data_total_len))
    np.random.shuffle(index_list)

    train_index = index_list[:split_index]
    valid_index = index_list[split_index:]
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    train_valid_data = ImgDataset(image_train_valid, labels)
    train_loader_pt4 = DataLoader(train_valid_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader_pt4 = DataLoader(train_valid_data, batch_size=len(valid_index), sampler=valid_sampler)

    # Part 4.3
    lossRec = []
    vlossRec = []
    nRec = []
    trainAccRec = []
    validAccRec = [] 

    start = time()

    for i in range(epochs):
        running_loss = []
        running_accuracy = []
        running_valid_loss = []
        running_valid_accuracy = []
        for j, data in enumerate(train_loader_pt4):
            feats, labels = data
            optimizer.zero_grad()
            predict = neuron(feats)
            loss = loss_function(input=predict.squeeze(), target=labels.long())
            loss.backward()  
            optimizer.step()  

            _, predicted = torch.max(predict.data, 1)
            trainAcc = (labels == predicted).sum().item() / batch_size

            if j % eval_every == 0:
                print("=== epoch: ", i+1, " ===")
                print("loss: ", f'{loss:.4f}'," trainAcc: ", f'{trainAcc:.4f}')
            
            running_accuracy.append(trainAcc)
            running_loss.append(loss)

        for k, data in enumerate(valid_loader_pt4):
            feats, labels = data
            predict = neuron(feats)
            vloss = loss_function(input=predict.squeeze(), target=labels.long())
            _, predicted = torch.max(predict.data, 1)
            validAcc = (labels== predicted).sum().item() / len(valid_index)

            if k % eval_every == 0:
                print("validAcc: ", f'{validAcc:.4f}')
            
            running_valid_accuracy.append(validAcc)
            running_valid_loss.append(vloss)


        trainAcc = sum(running_accuracy) / float(len(running_accuracy))
        loss = sum(running_loss) / float(len(running_loss))
        validAcc = sum(running_valid_accuracy) / float(len(running_valid_accuracy))
        vloss = sum(running_valid_loss) / float(len(running_valid_loss))
            
        lossRec.append(loss)
        vlossRec.append(vloss)
        nRec.append(i)
        trainAccRec.append(trainAcc)
        validAccRec.append(validAcc)

    end = time()
    summary(neuron, (3, 56, 56))
    print("Time: ", end - start)
    print("trainAcc: ", max(trainAccRec))
    print("validAcc: ", max(validAccRec))

    plt.subplot(1, 2, 1)
    plt.plot(nRec,lossRec, label='Train')
    plt.plot(nRec,vlossRec, label='Valid')
    plt.title('Training Loss vs. epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(nRec,trainAccRec, label='Train')
    plt.plot(nRec,validAccRec, label='Valid')
    plt.title('Training Accuracy vs. epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.ion()
    plt.show()


      
def part6(batch_size, lr, epochs, eval_every, neuron, loss_function):
    print("Starting...")
    
    optimizer = torch.optim.SGD(neuron.parameters(),lr=lr)
    seed = 1
    torch.manual_seed(seed)

    # Calculate mean and std
    transform_pt4 = transforms.Compose([transforms.ToTensor()])
    images_data_pt4 = torchvision.datasets.ImageFolder("asl_images", transform=transform_pt4)
    dataloader = DataLoader(images_data_pt4, shuffle=False, batch_size=batch_size)
    mean, std = mean_std(dataloader)

    transform_pt4 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(mean,mean,mean), std=(std,std,std))])
    images_data_pt4 = torchvision.datasets.ImageFolder("asl_images", transform=transform_pt4)
    dataloader = DataLoader(images_data_pt4, shuffle=False, batch_size=batch_size)
    mean, std = mean_std(dataloader)

    image_train_valid = []
    labels = []
    for i in images_data_pt4:
        image_train_valid.append(i[0].numpy())
        labels.append(i[1])

    data_total_len = len(images_data_pt4)

    index_list = []
    for i in range(data_total_len):
        index_list.append(i)

    split_index = int(np.floor(0.8*data_total_len))
    np.random.shuffle(index_list)

    train_index = index_list[:split_index]
    valid_index = index_list[split_index:]
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    train_valid_data = ImgDataset(image_train_valid, labels)
    train_loader_pt4 = DataLoader(train_valid_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader_pt4 = DataLoader(train_valid_data, batch_size=len(valid_index), sampler=valid_sampler)

    lossRec = []
    vlossRec = []
    nRec = []
    trainAccRec = []
    validAccRec = [] 

    start = time()

    for i in range(epochs):
        running_loss = []
        running_accuracy = []
        running_valid_loss = []
        running_valid_accuracy = []
        for j, data in enumerate(train_loader_pt4):
            feats, labels = data
            optimizer.zero_grad()
            predict = neuron(feats)
            loss = loss_function(input=predict.squeeze(), target=labels.long())
            loss.backward()  
            optimizer.step()  

            _, predicted = torch.max(predict.data, 1)
            trainAcc = (labels == predicted).sum().item() / batch_size

            if j % eval_every == 0:
                print("=== epoch: ", i+1, " ===")
                print("loss: ", f'{loss:.4f}'," trainAcc: ", f'{trainAcc:.4f}')
            
            running_accuracy.append(trainAcc)
            running_loss.append(loss)

        for k, data in enumerate(valid_loader_pt4):
            feats, labels = data
            predict = neuron(feats)
            vloss = loss_function(input=predict.squeeze(), target=labels.long())
            _, predicted = torch.max(predict.data, 1)
            validAcc = (labels== predicted).sum().item() / len(valid_index)

            if k % eval_every == 0:
                print("validAcc: ", f'{validAcc:.4f}')
            
            running_valid_accuracy.append(validAcc)
            running_valid_loss.append(vloss)


        trainAcc = sum(running_accuracy) / float(len(running_accuracy))
        loss = sum(running_loss) / float(len(running_loss))
        validAcc = sum(running_valid_accuracy) / float(len(running_valid_accuracy))
        vloss = sum(running_valid_loss) / float(len(running_valid_loss))
            
        lossRec.append(loss)
        vlossRec.append(vloss)
        nRec.append(i)
        trainAccRec.append(trainAcc)
        validAccRec.append(validAcc)

    confusion = None
    
    # Confusion matrix
    for k, data in enumerate(valid_loader_pt4):
        feats, labels = data
        predict = neuron(feats)
        vloss = loss_function(input=predict.squeeze(), target=labels.long())
        _, predicted = torch.max(predict.data, 1)
        validAcc = (labels== predicted).sum().item() / len(valid_index)
        confusion = confusion_matrix(labels, predicted)
        print("confusion", confusion)
    
    end = time()
    summary(neuron, (3, 56, 56))
    print("Time: ", end - start)
    print("validAcc: ", max(validAccRec))
    ipdb.set_trace()


    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=200)
    args = parser.parse_args()


    model_pt4 = ConvNN_pt4()
    model_pt5 = ConvNN_pt5()
    model_pt6 = ConvNN_pt6()
    model_pt6_small = ConvNN_pt6_small()

    loss_mse = torch.nn.MSELoss() 
    loss_celoss = torch.nn.CrossEntropyLoss()

    # part4(batch_size, lr, epochs, eval_every, model_pt4, loss_mse)

    # =============== Part 5 ===============

    # Just batch normalization
    # part4(batch_size, lr, epochs, eval_every, model_pt5, loss_mse)

    # Just cross entropy loss
    # part5(batch_size, lr, epochs, eval_every, model_pt4, loss_celoss)

    # Both batch normalization and cross entropy loss
    # part5(batch_size, lr, epochs, eval_every, model_pt5, loss_celoss)

    # =============== Part 6 ===============
    # part6(batch_size, lr, epochs, eval_every, model_pt6, loss_celoss)

    # part6(batch_size, lr, epochs, eval_every, model_pt6_small, loss_celoss)



if __name__ == "__main__":
    main()


