
import glob, os, sys, contextlib, re
# import xml.etree.ElementTree as et
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
# from sklearn.model_selection import train_test_split
# from torch.utils.data import WeightedRandomSampler
# from collections import Counter

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import cv2
# import random
# from sklearn.model_selection import train_test_split

# import mlflow
# from getpass import getpass

from pydub import AudioSegment
import wave, librosa
print(f"Running in {os.getcwd()}")
print("External packages imported\n")

# Select where you are running this script -----------------#

AWS = False
google = False
if google:
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir("/content/drive/My Drive/Team 6")
    rootPath = "/content/drive/My Drive/Team 6"
    dataPath = rootPath

else:
    rootPath = '/speechRecForMeeting'
    dataPath = ''

# -----------------------------------------------------------

from data_preprocessing import *

# # DagsHub set-up --------------------------------
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'team-token'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'f01653d37636d9488c48cd922f6ab83881d2cf4a'
# os.environ['MLFLOW_TRACKING_PROJECTNAME'] = 'speechRecForMeeting'

# mlflow.set_tracking_uri(f'https://dagshub.com/Viv-Crowe/speechRecForMeeting.mlflow')

sys.path.append(rootPath + '/py files')
from data_preprocessing import *
from data_loader import *
from CNN import *

## Data pre-processing ##

with open('dialogue-acts-prepped.pkl','rb') as f:
        df_diag_acts = pickle.load(f)

with open('processed-data/df_timestamps.pkl','rb') as f:
        df_timestamps = pickle.load(f)

# Get segment ids from df_timestamps
idx = df_timestamps.index
seg_ids = [df_timestamps['meeting_id'][j] + '_' + str(j) for j in idx]

# Prepare dataset
# [segment_full_paths, df_timestamps] = processSignals("Signals-new-10", rootPath)
p = {'segment length': 10, 'overlap length': 1}
dataset_path, p = prepareDataset(segment_paths, df_diag_acts, df_timestamps, p, AWS)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

## Load dataset ##

pickle_file = rootPath + "/processed-data" + "/dataset-10M.pkl"
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
    tr = train(CNN, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=5, use_gpu)
    train_error_rates, train_losses, test_error_rates, test_losses = tr

    y_hat, y_true, losses, error = prediction(test_dataloader, CNN, criterion)

    roc_auc, precision, recall, accuracy, y_hat_class = evaluate(y_hat, y_true)

    # Log parameters + metrics
    mlflow.log_params(p)
    mlflow.log_param('CNN parameters',CNN.parameters())
    mlflow.log_metrics(m)
    # for i in epoch:
    #   mlflow.log_metrics(test_error, step=i)
    
## Evaluate model ##
# results = evaluate(model, x_test, y_test)
