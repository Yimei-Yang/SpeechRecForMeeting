
from data_preprocessing import *
import glob, os

#preprocessing the data

try:
    getInputSegments
except NameError:
    print("getInputSegments doesn't exist")
else:
    print("getInputSegments exists")

# os.chdir("/content/drive/My Drive/Team 6/Signals-test")
# for file in glob.glob("*.wav"):
#   segment_list = getInputSegmentTimes(file, 10, 0)
#   output = getInputSegments(file, segment_list)