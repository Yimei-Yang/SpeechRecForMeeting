# ! pip install pydub wave regex pathlib librosa contextlib2 pickle-mixin

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
import torch
from torch.utils.data import Dataset

class dataset(Dataset):

    def __init__(self, features, df_timestamps, labels):
        self.labels = labels
        self.features = features
        self.df_timestamps = df_timestamps

    def __len__(self):
      if len(self.labels) == feature.size[0]:
        return len(self.labels)
      else:
        print("Feature size doesn't match label length.")

    def __getitem__(self, idx):
        # ask Jingming for help
        sample = [0]
        return sample

def prepareDataset(segment_paths, df_timestamps, df_diag_acts, p):

  features, df_timestamps, p = getFeatures(segment_full_paths, df_timestamps, p)
  print("Feature size: {}".format(features.size))
  labels = getLabels(df_timestamps, df_diag_acts)
  print("Labels size: {}".format(features.size))

  data = dataset(features, df_timestamps, labels)

  dataset_path = './processed-data/whole-dataset.pkl'

  with open(dataset_path, 'wb') as f:
    print("Writing to {}".format(dataset_path))
    pickle.dump(data, f)
  return dataset_path

def processSignals(signals_folder, rootPath):
  '''
  inputs path (str) to 
  '''
  os.chdir(signals_folder)
  segment_length, overlap_length = 10, 1 # must be an int
  p = [segment_length, overlap_length]

  df_timestamps = pd.DataFrame()
  segments_path = []
  for audio_file in glob.glob('*.wav'):
    df_timestamps_t = getInputSegmentTimes(audio_file, segment_length, overlap_length)
    segments_paths_t = getInputSegments(audio_file, df_timestamps_t, rootPath)
    df_timestamps = df_timestamps.append(df_timestamps_t)
    segments_path.append(segments_paths_t)

  os.chdir(rootPath)

  # segment_full_paths = [rootPath + "/segments-viv/" + s for s in segments_path]

  print("Number of segments: {}".format(len(segments_path)))
  print("df_timestamps shape: {}".format(df_timestamps.shape))

  return segments_path, df_timestamps, p

def processDialogueActs(path2all_xml_files):
  df_diag_acts = dialogueActsXMLtoPd(path2all_xml_files) # rootPath + '/dialogue-acts/*.xml'
  df_diag_acts = addDAoIVariable(df_diag_acts)
  df_diag_acts = df_diag_acts[df_diag_acts.DAoI]
  diag_acts_path = './processed-data/dialogue-acts-prepped.pkl'

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

    return df_timestamps

def getInputSegments(audio_file, df_timestamps, rootPath):
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
      
      segment_path = "{}/segments-viv/{}_{}.wav".format(rootPath, df_timestamps['meeting_id'][idx], count)
      absPath = os.path.abspath(segment_path)
      #print("Ready to export to: {}".format(absPath))
    # os.makedirs("./segments_viv") (not working, we created the folder manually)
      audio_segment.export(segment_path, format="wav")
      segments.append(segment_path)

    return segments

def getFeatures(segment_paths, df_timestamps, p):
  '''
  get a list melspecs (i.e. a 2D np_array), one melspec per segment
  '''
  nfft = 512
  hop_length = 512/2
  win_length = 512
  p = [p, nfft, hop_length, win_length]
  #print("Number of segments: {}".format(len(segments)))
  features = []
  result = []
  #print(df_timestamps.describe)
  #print("timestamp length", df_timestamps.shape[0])
  #print(df_timestamps.iloc[:, 0])
  df_timestamps.reset_index(inplace=True)
  #print(df_timestamps.iloc[:, 0])
  #print(df_timestamps.loc[[792]])
  #print("segments length", len(segments))
  for idx, segment in enumerate(segments):
    signal, sr = librosa.load(segment, sr=None)
    if signal is None or len(signal) == 0:
      df_timestamps = df_timestamps.drop([idx])
    elif idx == (len(segments)-1):
      df_timestamps = df_timestamps[:-1]
    else:
      melspect = librosa.feature.melspectrogram(signal, n_fft = nfft, hop_length = hop_length, win_length = win_length)
      #save all np.arrays(.wav) files into an array -> X dataset
      if features and not melspect.shape == features[0].shape :
        #print(df_timestamps.loc[[idx]])
        df_timestamps = df_timestamps.drop([idx])
      else:
        features.append(melspect)
  #print("Finished computing features")
  #print("length of the features", len(features))
  shape = features[0].shape
  for x in features:
    if not x.shape==shape:
      print(x.shape)
  features = np.stack(features)
  features = torch.Tensor(features)
  features = features.reshape(features.shape[0], 1, features.shape[1], features.shape[2])
  #print(features.shape)
  #print("length of feature list", features.shape[0])
  #print(features[0][0].shape)
  #print("length of timestamps", df_timestamps.shape[0])
  df_timestamps.reset_index(inplace=True)
  result.append(features)
  result.append(df_timestamps)
  return result

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
  
  df['meeting_id'] = df['meeting_id'].str[:6]
  print("Finished converting dialogue acts XML files")
  return df

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
  
  counts = np.zeros(df_timestamps.shape[0])
  #print("length of df_timestamps", df_timestamps.shape[0])
  df_it = df_diag_acts.loc[df_diag_acts['DAoI'] == True]
  #print("whole shape", df_diag_acts.shape[0])
  #print("true shape", df_it.shape[0])
  #print(df_timestamps)
  for seg_index, seg_row in df_timestamps.iterrows():
    #print("df_timestamps row length", seg_index)
    for diag_acts_index, diag_acts_row in df_it.iterrows():
      if seg_row['meeting_id'] != diag_acts_row['meeting_id']:
        continue
      elif seg_row['st_time'] < diag_acts_row['st_time'] and seg_row['ed_time'] > diag_acts_row['ed_time']:
        counts[seg_index] += 1
      else:
        counts[seg_index] = 0

  # label as True if there's at least one entire interruption in the segment
  for idx, num in enumerate(counts):
    if num>0:
      counts[idx] = int(1)
    else:
      counts[idx] = int(0)
  A = [int(counts) for counts in counts]
  #print("type of A", type(A[0]))
  #non = 0
  #yes=0
  #for x in A:
    #if x == 0:
      #non = non+1
    #else:
      #yes = yes+1
  #print(len(A))
  #print("non", non, "yes", yes)
  #print("Finished getting labels")
  #print(A)
  #print("type of lable list is", type(A))
  #print("type of lable is", type(A[0]))
  return A