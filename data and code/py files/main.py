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
    pass

#sys.path.append(rootPath + '/py files')
#from data_preprocessing import *

os.chdir("Signals-10M")
segment_length = 10 # must be an int
overlap_length = 1 # must be an int
print(os.getcwd())
df_timestamps = pd.DataFrame()
segments_path = []
for audio_file in glob.glob('*.wav'):
  df_timestamps_t = getInputSegmentTimes(audio_file, segment_length, overlap_length)
  #segments_paths_t = getInputSegments(audio_file, df_timestamps_t, rootPath)
  df_timestamps = df_timestamps.append(df_timestamps_t)
  #segments_path.append(segments_paths_t)

os.chdir(rootPath + '/segments-10M')
segment_paths = glob.glob("*.wav")
segment_full_paths = [rootPath + "/segments-10M/" + s for s in segment_paths]

os.chdir(rootPath)
result = getFeatures(segment_full_paths, df_timestamps)
features = result[0]
df_timestamps = result[1]
print("length of df_timestamps", df_timestamps.shape[0])
with open('/content/drive/My Drive/Team 6/processed-data/features-10M.pkl', 'wb') as f:
  pickle.dump(features, f)

df_diag_acts = dialogueActsXMLtoPd(rootPath + '/dialogue-acts-10M/*.xml')
with open('/content/drive/My Drive/Team 6/processed-data/dialogue-acts-10M.pkl', 'wb') as f:
  pickle.dump(df_diag_acts, f)


df_diag_acts = addDAoIVariable(df_diag_acts)
labels = getLabels(df_timestamps, df_diag_acts)
with open('/content/drive/My Drive/Team 6/processed-data/labels-10M.pkl', 'wb') as f:
  pickle.dump(labels, f)