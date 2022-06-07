from pydub import AudioSegment
import wave
import contextlib
import numpy as np
import pandas as pd
import re

import glob, os
import librosa

#get a pd of [id, st_time, ed_time]
def getInputSegmentTimes(list_of_timestamps, audio_file, segment_length, overlap_length):
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
    for i in range(0, len(l), segment_length - overlap_length):
      list_of_timestamps.append([meeting_id,i,i+segment_length])

    df_timestamps = pd.DataFrame(list_of_timestamps, columns=['meeting_id','st_time','ed_time'])

    return df_timestamps


def getInputSegments(audio_file, df_timestamps):
    '''
    input: path to audio_file, df_timestamps
    output: [path/audio_chunk_{}, ..., path/audio_chunk_{}]
    '''
    audio = AudioSegment.from_wav(audio_file)
    segments=[]
    count = 0
    for idx in df_timestamps.index:

      #break loop if at last element of list
      if idx == len(df_timestamps.index):
          break
      start = df_timestamps['st_time'][idx] * 1000
      end = df_timestamps['ed_time'][idx] * 1000 #pydub works in millisec
      audio_chunk=audio[start:end]
      count = count + 1
      segment_name="/content/drive/My Drive/Team 6/segments/{}_{}.wav".format(df_timestamps['meeting_id'][idx], count)
      audio_chunk.export(segment_name, format="wav")
      segments.append(segment_name)

    return segments

#list of np_array(2d matrix)
def getFeatures(output, np_list):
  for segment in output:
    signal,sr = librosa.load(segment,sr=None)
    if signal is None or len(signal) == 0:
      continue
    else:
      melspect = librosa.feature.melspectrogram(signal)
      #save all np.arrays(.wav) files into an array -> X dataset
      np_list.append(melspect)

def getLabels(segments, interruptions):
    counts = np.empty(segments.shape[0])

    for seg_index, seg_row in segments.iterrows():
      for inter_index, inter_row in interruptions.iterrows():
        if seg_row['st_time'] < inter_row['st_time'] and seg_row['ed_time'] > inter_row['ed_time']:
          counts[seg_index] += 1
        else:
          counts[seg_index] = 0

    # label as True if there's at least one entire interruption in the segment
    labels = counts > 0 
    return labels