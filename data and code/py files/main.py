import glob, os, sys, contextlib, re
import xml.etree.ElementTree as et
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

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

# os.chdir("Signals-test")
# audio_file = 'Bed004.interaction.wav'
# segment_length = 10 # must be an int
# overlap_length = 1 # must be an int
# print(os.getcwd())

# df_timestamps = getInputSegmentTimes(audio_file, segment_length, overlap_length)

# # segments_paths = getInputSegments(audio_file, df_timestamps, rootPath)

# os.chdir(rootPath + '/segments')
# segment_paths = glob.glob("*.wav")
# segment_full_paths = [rootPath + "/segments/" + s for s in segment_paths]

os.chdir(rootPath)
features = getFeatures(segment_full_paths)
with open('./processed-data/features.pkl', 'wb') as f:
  pickle.dump(features, f)

df_diag_acts = dialogueActsXMLtoPd(rootPath + '/dialogue-acts/*.xml')
with open('./processed-data/dialogue-acts.pkl', 'wb') as f:
  pickle.dump(df_diag_acts, f)
  
df_diag_acts.head()
df_diag_acts = addDAoIVariable(df_diag_acts)
labels = getLabels(df_timestamps, df_diag_acts)
with open('./processed-data/labels.pkl', 'wb') as f:
  pickle.dump(labels, f)