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
import pickle
import os, sys

from torch.utils.data import WeightedRandomSampler
from collections import Counter

from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 2, stride=2)
        self.conv3 = nn.Conv2d(6, 6, 2, stride=2)
        self.fc1 = nn.Linear(72, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluation(loader, model, criterion):

  y_true_per_epoch = []
  y_hats_per_epoch = []

  for i, (images, labels) in enumerate(loader):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
      # switch tensor type to GPU
      images = images.cuda()
      labels = labels.cuda()

      
    #print(image.shape, 'test')
    outputs = model(images)
    predictions = outputs[:,1].clone()
    #print('output size: ', predictions.size())
    
    loss = criterion(outputs, labels)
  
    _, predictions = torch.max(outputs, 1)

    correct += torch.sum(labels == predictions).item() #predictions <class 'torch.Tensor'>
    total += labels.shape[0]
    
    losses += loss.data.item()
    
    #keep track of each batch
    for j in range(0,len(labels)):
      y_true_per_epoch.append(int(labels[j]))
      y_hats_per_epoch.append(outputs[j][1])

  #ap_score_per_epoch <- float type
  ap_score_per_epoch = average_precision_score(y_true_per_epoch, y_hats_per_epoch)

  #print(ap_score_per_epoch)

  return losses/len(list(loader)), 1 - correct/total, ap_score_per_epoch # we need to normalize loss with respect to the number of batches 

def train(CNN, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=5):
  train_losses = []
  test_losses = []

  train_error_rates = []
  test_error_rates = []

  ap_score_list = []

  use_gpu = True
  if use_gpu:
    # switch model to GPU
    CNN.cuda()

  for epoch in range(num_epochs): 
    train_loss = 0 
    n_iter = 0 
    total = 0
    correct = 0
    
    for i, (images, labels) in enumerate(train_dataloader): 
      optimizer.zero_grad() 

      if use_gpu: 
        images = images.cuda()
        labels = labels.cuda()

      #print(images.shape, 'train')
      outputs = CNN(images)
      
      # to compute the train_error_rates  
      _, predictions = torch.max(outputs, 1)
      correct += torch.sum(labels == predictions).item()
      total += labels.shape[0]
      
      # compute loss
      loss_bs = criterion(outputs, labels)
      # compute gradients
      loss_bs.backward()
      # update weights
      optimizer.step()

      train_loss += loss_bs.detach().item()

      n_iter += 1

    train_error_rate = 1 - correct/total
    
    ap_score = 0

    with torch.no_grad():
      test_loss, test_error_rate, ap_score = evaluation(val_dataloader, CNN)

    train_error_rates.append(train_error_rate)
    test_error_rates.append(test_error_rate)
    train_losses.append(train_loss/n_iter)
    test_losses.append(test_loss)

    ap_score_list.append(ap_score)

    from collections import defaultdict

    m = defaultdict(list)

    if epoch%1 == 0:
      print('Epoch: {}/{}, Loss: {:.4f}, Error Rate: {:.1f}%, Average_precision_score: {:.1f}'.format(epoch+1, num_epochs, train_loss/n_iter, 100*train_error_rate, ap_score))

  for i in range(0, num_epochs):
    m["Num_epochs"].append(i+1)
    m["Test_loss"].append(train_losses[i])
    m["Error_Rate"].append(100*train_error_rates[i])
    m["Average_precision_score"].append(ap_score_list[i])

  print("finished")
  return m 

def initialize():
  return CNN()