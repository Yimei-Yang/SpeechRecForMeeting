import glob, os, sys, contextlib, re
import xml.etree.ElementTree as et
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import random
from sklearn.model_selection import train_test_split

import mlflow
from getpass import getpass

from pydub import AudioSegment
import wave, librosa

google = False
if google:
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir("/content/drive/My Drive/Team 6")
    rootPath = "/content/drive/My Drive/Team 6"
else:
    rootPath = './data and code'

# # DagsHub set-up --------------------------------
# os.environ['MLFLOW_TRACKING_USERNAME'] = input('Enter your DAGsHub username: ')
# os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')
# os.environ['MLFLOW_TRACKING_PROJECTNAME'] = input('Enter your DAGsHub project name: ') #speechRecForMeeting

mlflow.set_tracking_uri(f'https://dagshub.com/Viv-Crowe/speechRecForMeeting.mlflow')

sys.path.append(rootPath + '/py files')
from data_preprocessing import *

# Pre-processing

# [segment_full_paths, df_timestamps] = processSignals("Signals-10M", rootPath)
# prepareDataset(segment_full_paths, df_timestamps, frac_interp, p)

# df_timestamps = getInputSegmentTimes(audio_file, segment_length, overlap_length)

# [features, df_timestamps] = processSegments("Signals-10M")
# diag_acts_path = processDialogueActs(path2all_xml_files)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

# # Load dataset
p = {}
DATA_PATH = rootPath + "/processed-data"
with open(DATA_PATH + '/features-10M.pkl','rb') as f:
    features_list = pickle.load(f)
with open(DATA_PATH + '/labels-10M.pkl','rb') as f:
    labels_list = pickle.load(f)

datafile = open('workfile', 'w')
datafile.write("Using just the first 10 meetings\n")

datafile.write("Number of samples: {}\n".format(len(features_list)))
datafile.write("Shape of a sample: {}\n\n".format(features_list[0].size()))

p['num of samples'] = len(features_list)
p['shape of X'] = features_list[0].size()

# from logistic_model import *

# [model, features] = initialize(features)
p['test_size'] = 0.1
p['train_size'] = 0.8
p['test_val_split'] = 0.5
p['seed'] = 1
p['lr'] = 0.01
p['momentum'] = 0.9

X_train,X_test,Y_train,Y_test = train_test_split(features_list, labels_list, test_size=p['test_size'], train_size=p['train_size'], random_state=p['seed'], shuffle=True)

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=p['test_val_split'], random_state=p['seed'])

train_data = crossJoin(X_train, Y_train)
val_data = crossJoin(X_val, Y_val)
test_data = crossJoin(X_test,Y_test)

print("Train/val/test split done\n")
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.fc1 = nn.Linear(26622, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

CNN = CNN()

#evaluation
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CNN.parameters(), lr=p['lr'], momentum=p['momentum'])

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
print("Data loaders ready\n")

def prediction(loader, model):

  correct = 0
  total = 0
  losses = 0

  for i, (images, labels) in enumerate(loader):
    if use_gpu:
      # switch tensor type to GPU
      images = images.cuda()
      labels = labels.cuda()
      
    # print(image.shape, 'test')
    outputs = model(images)
    
    loss = criterion(outputs, labels)
  
    _, predictions = torch.max(outputs, 1)
  
    correct += torch.sum(labels == predictions).item()
    total += labels.shape[0]
    
    losses += loss.data.item()
    
  return losses/len(list(loader)), 1 - correct/total, predictions # we need to normalize loss with respect to the number of batches 

with mlflow.start_run(run_name="CNN on 10 meetings"):
    mlflow.log_params(p)
    mlflow.pytorch.autolog()

    use_gpu = torch.cuda.is_available()

    train_losses = []
    test_losses = []

    train_error_rates = []
    test_error_rates = []


    if use_gpu:
        # switch model to GPU
        CNN.cuda()

    num_epochs = 1

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

            # print(images.shape, 'train')
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
    mlflow.log_metric("train_error", train_error_rate)

    with torch.no_grad():
        test_loss, test_error_rate, _ = prediction(val_dataloader, CNN)

    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_error_rate", test_error_rate)

    train_error_rates.append(train_error_rate)
    test_error_rates.append(test_error_rate)
    train_losses.append(train_loss/n_iter)
    test_losses.append(test_loss)
    mlflow.pytorch.autolog()
    if epoch%1 == 0:
        print('Epoch: {}/{}, Loss: {:.4f}, Error Rate: {:.1f}%'.format(epoch+1, num_epochs, train_loss/n_iter, 100*train_error_rate))


print('Finished Training')

# # # Train and evaluate model
# model = train(model, x_train, y_train)

# results = evaluate(model, x_test, y_test)

f.close()