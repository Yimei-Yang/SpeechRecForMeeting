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

from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score


class model(nn.Module):
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

def prediction(loader, model):

  correct = 0
  total = 0
  losses = 0

  for i, (images, labels) in enumerate(loader):
    if use_gpu:
      # switch tensor type to GPU
      images = images.cuda()
      labels = labels.cuda()
       
    #print(image.shape, 'test')
    outputs = model(images)
    
    loss = criterion(outputs, labels)
  
    _, predictions = torch.max(outputs, 1)
  
    correct += torch.sum(labels == predictions).item()
    total += labels.shape[0]
    
    losses += loss.data.item()
    
  return losses/len(list(loader)), 1 - correct/total # we need to normalize loss with respect to the number of batches 

def train(model, x_train, y_train):
  #evaluation
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(CNN.parameters(), lr=0.01, momentum=0.9)

  train_losses = []
  test_losses = []

  train_error_rates = []
  test_error_rates = []

  y_hat = []


  if use_gpu:
    # switch model to GPU
    CNN.cuda()

  num_epochs = 15

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
      #print(outputs.shape)
      #print(labels.shape)
      #print(labels) 
      loss_bs = criterion(outputs, labels)
      # compute gradients
      loss_bs.backward()
      # update weights
      optimizer.step()

      train_loss += loss_bs.detach().item()

      n_iter += 1

    train_error_rate = 1 - correct/total

    with torch.no_grad():
      test_loss, test_error_rate = prediction(val_dataloader, CNN)

    train_error_rates.append(train_error_rate)
    test_error_rates.append(test_error_rate)
    train_losses.append(train_loss/n_iter)
    test_losses.append(test_loss)

    y_hat.append(1-test_error_rate)

    if epoch%1 == 0:
      print('Epoch: {}/{}, Loss: {:.4f}, Error Rate: {:.1f}%'.format(epoch+1, num_epochs, train_loss/n_iter, 100*train_error_rate))
  
  return y_hat

def evaluate(y_true, y_hat):
  y_hat_class = [1 if x >= 0.5 else 0 for x in y_hat]  # convert probability to class for classification report

  #report_string += classification_report(y_true, y_hat_class)
  roc_auc = roc_auc_score(y_true, y_hat)
  precision = precision_score(y_true, y_hat)
  recall = recall_score(y_true, y_hat)
  accuracy = accuracy_score(y_true, y_hat) 
    
  return roc_auc, precision, recall, accuracy
