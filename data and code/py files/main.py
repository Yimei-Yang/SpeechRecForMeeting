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

google = True
if google:
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir("/content/drive/My Drive/Team 6")
    rootPath = "/content/drive/My Drive/Team 6"
else:
    rootPath = './data and code'

# # DagsHub set-up --------------------------------
os.environ['MLFLOW_TRACKING_USERNAME'] = 'team-token'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'f01653d37636d9488c48cd922f6ab83881d2cf4a'
os.environ['MLFLOW_TRACKING_PROJECTNAME'] = 'speechRecForMeeting' #speechRecForMeeting

mlflow.set_tracking_uri(f'https://dagshub.com/Viv-Crowe/speechRecForMeeting.mlflow')

sys.path.append(rootPath + '/py files')
from data_preprocessing import *
from data_loader import *
from CNN import *

## Data pre-processing ##

# [segment_full_paths, df_timestamps] = processSignals("Signals-10M", rootPath)
# prepareDataset(segment_full_paths, df_timestamps, frac_interp, p)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

## Load dataset ##
DATA_PATH = rootPath + "/processed-data"
pickle_file = DATA_PATH + "/dataset-10M.pkl"
train_dataloader, val_dataloader, test_dataloader, p = prepareData(pickle_file)
examineBatches(train_dataloader, val_dataloader, test_dataloader)

## Train dataset ##
CNN = initialize()
criterion = nn.CrossEntropyLoss()
p['lr'] = 0.01
p['momentum'] = 0.9
optimizer = optim.SGD(CNN.parameters(), lr=p['lr'], momentum=p['momentum'])
with mlflow.start_run(run_name="CNN on 10 meetings"):
    use_gpu = torch.cuda.is_available()
    m = train(CNN, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=5)
    
    # Log parameters + metrics
    mlflow.log_params(p)
    mlflow.log_param('CNN parameters',CNN.parameters())
    mlflow.log_metrics(m)
    
## Evaluate model ##
# results = evaluate(model, x_test, y_test)