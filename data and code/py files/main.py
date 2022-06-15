
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

sys.path.append(rootPath + '/py files')
from data_preprocessing import *


# # DagsHub set-up --------------------------------
# os.environ['MLFLOW_TRACKING_USERNAME'] = input('Enter your DAGsHub username: ')
# os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')
# os.environ['MLFLOW_TRACKING_PROJECTNAME'] = input('Enter your DAGsHub project name: ') #speechRecForMeeting

# mlflow.set_tracking_uri(f'https://dagshub.com/Viv-Crowe/speechRecForMeeting.mlflow')

# Pre-processing
# diag_acts_path = processDialogueActs(path2all_xml_files)
# [segment_full_paths, df_timestamps] = processSignals("Signals-10M", rootPath)

p = {'segment_length': 10, 'overlap_length': 1}
segment_paths = glob.glob('./segments-5/*.wav')

with open('./processed-data/dialogue-acts-prepped.pkl', "rb") as f:
    df_diag_acts = pickle.load(f)
df_diag_acts.reset_index(inplace=True)

with open('processed-data/df_timestamps.pkl', "rb") as f:
    df_timestamps = pickle.load(f)
print("Precomputed dataframes loaded.")

dataset_path = prepareDataset(segment_paths, df_diag_acts.loc[1:6, :], df_timestamps, p)


# device = "cuda" if torch.cuda.is_available() else "cpu"
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

# from logistic_model import *

# [model, features] = initialize(features)

# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

#     # # Balance dataset
#     count=Counter(y_train)
#     class_count=np.array([count[0],count[1]])
#     weight=1./class_count
#     print(weight)

#     samples_weight = np.array([weight[t] for t in y_train])
#     samples_weight = torch.from_numpy(samples_weight)

    # train_error_rate = 1 - correct/total
    # mlflow.log_metric("train_error", train_error_rate)

    # with torch.no_grad():
    #     test_loss, test_error_rate, _ = prediction(val_dataloader, CNN)

    # mlflow.log_metric("test_loss", test_loss)
    # mlflow.log_metric("test_error_rate", test_error_rate)

    # train_error_rates.append(train_error_rate)
    # test_error_rates.append(test_error_rate)
    # train_losses.append(train_loss/n_iter)
    # test_losses.append(test_loss)
    # mlflow.pytorch.autolog()
    # if epoch%1 == 0:
    #     print('Epoch: {}/{}, Loss: {:.4f}, Error Rate: {:.1f}%'.format(epoch+1, num_epochs, train_loss/n_iter, 100*train_error_rate))


print('Finished Training')

# # # Train and evaluate model
# model = train(model, x_train, y_train)

# results = evaluate(model, x_test, y_test)

# f.close()