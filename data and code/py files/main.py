import glob, os, sys, contextlib, re
import xml.etree.ElementTree as et
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from pydub import AudioSegment
import wave, librosa

google = True
if google:
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir("/content/drive/My Drive/Team 6")
    rootPath = "/content/drive/My Drive/Team 6"
else:
    rootPath = ''

sys.path.append(rootPath + '/py files')
from data_preprocessing import *

# Pre-processing

segment_full_paths = processSegments("Signals")
processFeatures(segments_full_paths)


# # Train and evaluate model

# device = "cuda" if torch.cuda.is_available() else "cpu"
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

# train_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('/files/', train=True, download=True),
#   batch_size=batch_size_train, **kwargs)

# test_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('files/', train=False, download=True),
#   batch_size=batch_size, **kwargs)


# epochs = 100
# learning_rate = 0.01

# from logistic_model import *

# [model, features] = initialize(features)

# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

# model = train(model, x_train, y_train)

# results = evaluate(model, x_test, y_test)

