# ! pip install pydub wave regex pathlib librosa contextlib2 pickle-mixin mlflow

from pydub import AudioSegment
import wave
import contextlib
import numpy as np
import pandas as pd
import re
import xml.etree.ElementTree as et
from pathlib import Path
import torch
import glob, os
import librosa
import pickle

from torch.utils.data import Dataset


class dataset(Dataset):

    def __init__(self, features, labels, p = {}, df_timestamps = None):

        self.labels = labels
        self.features = features
        self.df_timestamps = df_timestamps
        self.p = p

    def __len__(self):
      if len(self.labels) == len(self.features):
        return len(self.labels)
      else:
        print("Feature size doesn't match label length.")

    def __getitem__(self, idx):
        sample = [self.features[idx], self.labels[idx]]
        return sample

def prepareDataset(segment_paths, df_diag_acts, df_timestamps, p, AWS=False):

  features, df_timestamps, p = getFeatures(segment_paths, df_timestamps, p)
  print("Feature size: {}".format(features[0].size()))
  print(f"Number of obs: {len(features)}")
  labels = getLabels(df_timestamps, df_diag_acts)
  print("Labels size: {}".format(len(labels)))
  p['# obs'] = len(labels)
  p['# interp.s'] = sum(labels)
  p['feature size'] = features[0].size(1)*features[0].size(2)
  print("Rows timestamps: ", df_timeshapes.shape[0])


  if AWS:
    data_key = 'dataset-4.pkl'
    bucket = 'ai4good-m6-2022'
    data_2 = dataset(features, df_timestamps, labels)
    
    pickle_data_32 = pickle.dumps(data_2) 
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket,key).put(Body=pickle_data_32)

  else:
    dataset_path = './processed-data/dataset-4.pkl'
    with open(dataset_path, 'wb') as f:
          print("Writing to {}".format(dataset_path))
          pickle.dump(data_whole, f)
    # -------------------------------------------------------
  
  return dataset_path, p

def processSignals(signals_folder, rootPath, AWS=False):
  '''
  inputs path (str) to 
  '''
  print("-----------------------")
  print("------Processing signals -------")
  print("-----------------------")
  p = {}

  p['segment_length'] = 10 # must be an int
  p['overlap_length'] = 1 # must be an int

  df_timestamps = df_timestamps = pd.DataFrame(columns=['meeting_id','st_time','ed_time'])
  segments_path = []

  # Fix
  if AWS:
    signals_paths = glob.glob('./75_whole_meetings/*.wav')
    print("Number of signals: ", len(signals_paths))
  else:
    signals_paths = glob.glob('./Signals/*.wav')
    print("Number of signals: ", len(signals_paths))

  
  for audio_file in signals_paths:
    df_timestamps_t = getInputSegmentTimes(audio_file, segment_length, overlap_length)
    segments_paths_t = getInputSegments(audio_file, df_timestamps_t, rootPath)
    df_timestamps = df_timestamps.append(df_timestamps_t)
    segments_path.append(segments_paths_t)
    print(f"{audio_file} segmented.\n")

  # segment_full_paths = [rootPath + "/segments-viv/" + s for s in segments_path]

  print("Number of segments: {}".format(len(segments_path)))
  print("df_timestamps shape: {}".format(df_timestamps.shape))

  return segments_path, df_timestamps, p

def processDialogueActs(path2all_xml_files):
  df_diag_acts = dialogueActsXMLtoPd(path2all_xml_files) # rootPath + '/dialogue-acts/*.xml'
  df_diag_acts = addDAoIVariable(df_diag_acts)
  df_diag_acts = df_diag_acts[df_diag_acts.DAoI]
  diag_acts_path = 'dialogue-acts-prepped.pkl'

  with open(diag_acts_path, 'wb') as f:
    print("Writing to {}".format(diag_acts_path))
    pickle.dump(df_diag_acts, f)
  return diag_acts_path

# ------------------------------------------------------------------- #

def getInputSegmentTimes(audio_file, segment_length, overlap_length):
    '''
    get a pd of [id, st_time, ed_time]
    '''
    list_of_timestamps = []
    # generate meeting name
  
    meeting_id = audio_file.partition('.interaction.wav')[0]
    #print(meeting_id)

    # get duration of meeting
    with contextlib.closing(wave.open(audio_file,'r')) as f:
      frames = f.getnframes()
      rate = f.getframerate()
      duration = frames / float(rate)

    l = list(range(int(duration)))

    # generate list of timestamps 
    for i in range(0, len(l), int(segment_length - overlap_length)):
      list_of_timestamps.append([meeting_id,i,i+segment_length])

    df_timestamps = pd.DataFrame(list_of_timestamps, columns=['meeting_id','st_time','ed_time'])
    df_timestamps = df_timestamps.reset_index(level=0)
    df_timestamps["seg_id"] = df_timestamps["meeting_id"]+ df_timestamps["index"].astype(str)
    return df_timestamps

def getInputSegments(audio_file, df_timestamps, rootPath, AWS=False):
    '''
    input: path to audio_file, df_timestamps['meeting_id','st_time','ed_time']
    output: list of paths to segment file names
    '''
    audio = AudioSegment.from_wav(audio_file)
  
    segments=[]
    count = 0
    for idx in df_timestamps.index:

      #break loop if at last element of list
      if idx == len(df_timestamps.index):
          break
      start = df_timestamps['st_time'][idx] * 1000
      end = df_timestamps['ed_time'][idx] * 1000
      #print(start, type(start), end, type(end))
      audio_segment = audio[start:end]
      #print("segmented")
      count = count + 1
      
      segment_path = "{}/segments-new-10/{}_{}.wav".format(dataPath, df_timestamps['meeting_id'][idx], count)
      absPath = os.path.abspath(segment_path)
      #print("Ready to export to: {}".format(absPath))
    # os.makedirs("./segments_viv") (not working, we created the folder manually)
      audio_segment.export(segment_path, format="wav")
      segments.append(segment_path)

    return segments

def getFeatures(segment_paths, df_timestamps, p, AWS=False):
  '''
  get a list melspecs (i.e. a 2D np_array), one melspec per segment
  '''
  p["nfft"] = 400
  p["hop_length"] = 200
  p["win_length"] = 400
  p["fmax"] = 400
  p["n_mels"] = 16
  print("-----------------------------------")
  print("Getting features")
  print("-----------------------------------")
  print("Number of segments: {}".format(len(segment_paths)))
  features = []
  result = []

  df_timestamps.reset_index(inplace=True)

  for idx, segment in enumerate(segment_paths):
    signal, sr = librosa.load(segment, sr=None)
    if signal is None or len(signal) == 0:
      print(f"segment {idx} didn't exist or was empty.")
      df_timestamps = df_timestamps.drop([idx])
    elif idx == (len(segment_paths)-1):
      df_timestamps = df_timestamps[:-1]
    else:
      melspect = librosa.feature.melspectrogram(signal, n_fft = p["nfft"], hop_length = p["hop_length"], win_length = p["win_length"], n_mels = p["n_mels"])
      feat = torch.Tensor(melspect)
      feat = feat.reshape(1, melspect.shape[0],melspect.shape[1])
      features.append(feat)
      print(f"Computed feature for segment {segment}")
  if len(features) > 0:
    shape = features[0].size()
    print(f"Shape of one feature: {shape}")
  else: print("No features in list")
  for idx,x in enumerate(features):
    if not x.size()==shape:
      df_timestamps = df_timestamps.drop([idx])
      print(f"Incorrect feature shape found: {x.size()}")
  print(f"features is a {type(features)} with {len(features)} elements: {features[0].size()}")

  return features, df_timestamps, p

def dialogueActsXMLtoPd(pathToDialogueActs):
  '''
  If using google drive, the path is "/content/drive/My Drive/Team 6/xml-audio/*.xml"
  '''
  filenames = glob.glob(pathToDialogueActs)
  filename = [] 
  li = []
  for filename in filenames :   
    global left
    parsed_xml = et.parse(filename)
    #I hard-coded here but i think it would be easier to modify the panda dataframe later
    dfcols = ['meeting_id', 'st_time', 'ed_time', 'type', 'adjacency', 'original-type', 'channel', 'participant']
    left = pd.DataFrame(columns=dfcols)

    root = parsed_xml.getroot()
  
    for diaAct in parsed_xml.findall('./dialogueact'):
      uId = diaAct.get('{http://nite.sourceforge.net/}id')
      sT = diaAct.get("starttime")
      eT = diaAct.get("endtime")
      tP = diaAct.get("type")
      aJ = diaAct.get("adjacency")
      oT = diaAct.get("original-type")
      ch = diaAct.get("channel")
      par = diaAct.get("participant")

      
      left = left.append(pd.Series([uId, sT, eT, tP, aJ, oT, ch, par], index=dfcols),ignore_index=True)
      li.append(left)
            
  df = pd.concat(li, axis=0, ignore_index=True)
  df.loc[:, 'st_time'] = pd.to_numeric(df.loc[:, 'st_time'])
  df.loc[:, 'ed_time'] = pd.to_numeric(df.loc[:, 'ed_time'])
  df = df.drop_duplicates(keep='first')
  df.reset_index(inplace=True)
  df['meeting_id'] = df['meeting_id'].str[:6]
  print("Finished converting dialogue acts XML files")
  return df

def selectSample(label_list, df_timestamps, feature_list):
  interrupted_list = []
  uninterrupted_list = []
  timestamps_list = []
  df_timestamps_in = pd.DataFrame()
  df_timestamps_un = pd.DataFrame()
  for idx, label in enumerate(label_list):
    if label == 1:
      interrupted_list.append((feature_list[idx],1))
      df_timestamps_in.append(df_timestamps.loc[idx])
    else:
      uninterrupted_list.append((feature_list[idx],0))
      df_timestamps_un.append(df_timestamps.loc[idx])
  timestamps_list.append(df_timestamps_in)
  timestamps_list.append(df_timestamps_un)
  return interrupted_list, uninterrupted_list

def addDAoIVariable(df_diag_acts):
  # Add the 'DAoI' (bool) variable
  df_diag_acts['DAoI'] = df_diag_acts['type'].str.fullmatch('.*%-')
  df_diag_acts['DAoI'] = df_diag_acts['DAoI'].astype(bool)
  
  # View what types are counted as Interruptions 
  # print(df.loc[df['DAoI'], 'type'])
  return df_diag_acts

def getLabels(df_timestamps, df_diag_acts):

  '''
  input: df_timestamps[], df_diag_acts['meeting_id','st_time','ed_time']
  output: boolean vector with the same number of rows as df_timestamps
  '''
  print("-----------------------------------")
  print("Getting labels")
  print("-----------------------------------")
  counts = np.zeros(df_timestamps.shape[0])
  seg_index = 0 # don't use the index fromdf.iterrows(), they are non-unique segment id's
  df_diag_acts.reset_index(inplace=True)
  for diag_acts_index, diag_acts_row in df_diag_acts.iterrows():
    #print("df_timestamps row length", seg_index)
    for _, seg_row in df_timestamps.iterrows():
      if seg_row['meeting_id'] != diag_acts_row['meeting_id']:
        continue
      elif seg_row['st_time'] < diag_acts_row['st_time'] and seg_row['ed_time'] > diag_acts_row['ed_time']:
        
        print(f"Found segment for iterruption {diag_acts_index}: {seg_index}")
        counts[seg_index] += 1
        counts[seg_index] = 1
      else: None

  labels = np.empty(len(counts))
  for idx in range(len(counts)):
    if counts[idx] == 0:
      labels[idx] = 0
    else: labels[idx] = 1
  return labels
