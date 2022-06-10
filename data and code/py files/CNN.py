'''all imports goes at top'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2,glob
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch.optim as optim

'''
this void function should be used directly, gives an example of X and Y respectively
'''

def data_retrieval():
    from google.colab import drive
    import pickle

    # This will prompt for authorization.
    drive.mount('/content/drive')

    DATA_PATH = "/content/drive/My Drive/Team 6/processed-data"
    features= open(DATA_PATH+'/features-test.pkl','rb')
    labels= open(DATA_PATH+'/labels-test.pkl','rb')
    features_list = pickle.load(features)
    labels_list = pickle.load(labels)


    print("features_list","\n", features_list[:5])
    print("length", len(features_list)) 
    print("width", len(features_list[0]))
    print("-----------------")
    print("labels_list","\n",labels_list[:5])
    print("length", len(labels_list)) 
    print("width", labels_list[0])


'''
this function is for join X1 and Y1 as a pair

list1 = np.array([['a', 'b'], ['c','d'],['e','f']])
list2 = np.array([[True], [False], [True]])
list3 = crossJoin (list1,list2)
list3
>>>[[array(['a', 'b'], dtype='<U1'), array([1])],
    [array(['c', 'd'], dtype='<U1'), array([0])],
    [array(['e', 'f'], dtype='<U1'), array([1])]]

'''
def crossJoin(list1,list2):
  crossJoined_list = []
  
  for i in range(0,len(list1)):
    inner_list = []
    for j in range(0,1):
      inner_list.append(list1[i])
      inner_list.append(list2[i])
    crossJoined_list.append(inner_list)

  return crossJoined_list

def train_test_split(tensor_list,labels_list):
    #1. convert a list of images(np.arrays) to a 3D tensor
    tensor_list = [torch.from_numpy(item) for item in features_list]
    tensor_list = tensor_list
    labels_list = labels_list
    #2. add one singleton axis (you can use np.expand_dims for that) to get 4D array with channels dimension equal to 1
    ##?? since we are only look into 1 channel so no need to make it 4d?
    tensor = np.expand_dims(tensor_list,axis=1)
    #3. use train_test_split from sklearn because it allows to shuffle your data before splitting
    X_train,X_test,Y_train,Y_test = train_test_split(tensor_list, labels_list, test_size=0.1, train_size=0.8, random_state=1, shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=1)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def get_dataloader(X_train, Y_train, X_test, Y_test, X_val, Y_val):
    train_data = crossJoin(X_train, Y_train)
    val_data = crossJoin(X_val, Y_val)
    test_data = crossJoin(X_test,Y_test)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 4, 5)
        self.fc1 = nn.Linear(2175,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def training(epoches, train_dataloader, loss_fnc, optimizer, criterion):
    for epoch in range(epoches):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = CNN(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 4 == 3:    # print every 4 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 4:.3f}')
                running_loss = 0.0

    print('Finished Training')


'''

NEEDS MODIFICATION

'''
def prediction(loader, model):

  correct = 0
  total = 0
  losses = 0

  for i, (images, labels) in enumerate(loader):
    if use_gpu:
      # switch tensor type to GPU
      images = images.cuda()
      labels = labels.cuda()
    
    outputs = model(images)
    
    loss = criterion(outputs, labels)
  
    _, predictions = torch.max(outputs, 1)
  
    correct += torch.sum(labels == predictions).item()
    total += labels.shape[0]
    
    losses += loss.data.item()
    
  return losses/len(list(loader)), 1 - correct/total # we need to normalize loss with respect to the number of batches 



'''

NEEDS MODIFICATION

'''
def validating():
    return 0


'''

NEEDS MODIFICATION

'''
def testing():
    return 0

'''

NEEDS MODIFICATION

'''

def plotting():
    return 0