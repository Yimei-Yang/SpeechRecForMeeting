import glob, os, sys, contextlib, re
import xml.etree.ElementTree as et
from pathlib import Path

from pydub import AudioSegment
import wave, librosa
import numpy as np
import pandas as pd


google = True
if google:
    from google.colab import drive
    drive.mount('/content/drive')
    os.chdir("/content/drive/My Drive/Team 6")
    rootPath = "/content/drive/My Drive/Team 6"
else:
    rootPath = os.getcwd()

sys.path.append(rootPath + '/py files')
from data_preprocessing import *

os.chdir("Signals-test")
audio_file = 'Bed004.interaction.wav'
segment_length = 10 # must be an int
overlap_length = 1 # must be an int
print(os.getcwd())

df_timestamps = getInputSegmentTimes(audio_file, segment_length, overlap_length)

segments_paths = getInputSegments(audio_file, df_timestamps, rootPath)

features = getFeatures(segments_paths)

df_diag_acts = dialogueActsXMLtoPd('dialogue-acts')

df_diag_acts = addDAoIVariable(df_diag_acts)



# os.chdir("/content/drive/My Drive/Team 6/Signals-test")
# for file in glob.glob("*.wav"):
#   segment_list = getInputSegmentTimes(file, 10, 0)
#   output = getInputSegments(file, segment_list)