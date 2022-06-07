#preprocessing the data
#import functions from data processing
import glob, os
os.chdir("/content/drive/My Drive/Team 6/Signals-test")
for file in glob.glob("*.wav"):
  segment_list = getInputSegmentTimes(file, 10, 0)
  output = getInputSegments(file, segment_list)