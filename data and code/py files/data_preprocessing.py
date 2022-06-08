# ! pip install pydub wave contextlib2 regex pathlib librosa

from pydub import AudioSegment
import wave
import contextlib
import numpy as np
import pandas as pd
import re
import xml.etree.ElementTree as et
from pathlib import Path

import glob, os
import librosa


def getInputSegmentTimes(audio_file, segment_length, overlap_length):
    '''
    get a pd of [id, st_time, ed_time]
    '''
    list_of_timestamps = []
    # generate meeting name
  
    meeting_id = audio_file.partition('.interaction.wav')[0]
    print(meeting_id)

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
    print("audio file path {}".format(audio_file))
    audio = AudioSegment.from_wav(audio_file)
    print("audio file loaded")
  
    segments=[]
    count = 0
    for idx in df_timestamps.index:

      #break loop if at last element of list
      if idx == len(df_timestamps.index):
          break
      start = df_timestamps['st_time'][idx] * 1000
      end = df_timestamps['ed_time'][idx] * 1000 #pydub works in millisec
      audio_segment = audio[start:end]
      count = count + 1
      segment_path = "{}/vivs_segments/{}_{}.wav".format(rootPath, df_timestamps['meeting_id'][idx], count)
      absPath = os.path.abspath(segment_path)
      print("Ready to export to: {}".format(absPath))

      audio_segment.export(segment_path, format="wav")
      segments.append(segment_path)

    return segments

def getFeatures(segments):
  '''
  get a list melspecs (i.e. a 2D np_array), one melspec per segment
  '''
  features = []
  for segment in segments:
    signal,sr = librosa.load(segment,sr=None)
    if signal is None or len(signal) == 0:
      continue
    else:
      melspect = librosa.feature.melspectrogram(signal)
      #save all np.arrays(.wav) files into an array -> X dataset
      features.append(melspect)
    return features  

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
    dfcols = ['Id', 'st_time', 'ed_time', 'type', 'adjacency', 'original-type', 'channel', 'participant']
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

  return df

def addDAoIVariable(df_diag_acts):
  # Add the 'Interruption' (bool) variable
  df_diag_acts['DAoI'] = df_diag_acts['type'].str.contains('%-', regex = False)
  df_diag_acts['DAoI'] = df_diag_acts['Interruption'].astype(bool)

  # View what types are counted as Interruptions 
  # print(df.loc[df['Interruption'], 'type'])
  return df_diag_acts

def getLabels(df_timestamps, df_diag_acts):
  '''
  input: df_timestamps[], df_diag_acts['meeting_id','st_time','ed_time', 'daoi']
  output: boolean vector with the same number of rows as df_timestamps
  '''
  counts = np.empty(df_timestamps.shape[0])

  for seg_index, seg_row in df_timestamps.iterrows():
    for diag_acts_index, diag_acts_row in df_diag_acts.iterrows():
      if seg_row['meeting_id'] != diag_acts_row['meeting_id']:
        continue
      elif seg_row['st_time'] < diag_acts_row['st_time'] and seg_row['ed_time'] > diag_acts_row['ed_time']:
        counts[seg_index] += 1
      else:
        counts[seg_index] = 0

  # label as True if there's at least one entire interruption in the segment
  labels = counts > 0 
  return labels