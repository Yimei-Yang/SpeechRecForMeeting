'''all imports goes at top'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2,glob
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import os, sys

from torch.utils.data import WeightedRandomSampler
from collections import Counter

from torch.utils.data import DataLoader
import torch.optim as optim

def crossJoin (list1,list2):
  crossJoined_list = []
  for i in range(0,len(list1)):
    inner_list = []
    for j in range(0,1):
      inner_list.append(list1[i])
      inner_list.append(list2[i])
    crossJoined_list.append(inner_list)
  return crossJoined_list

def prepareData(pickle_file):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    
    features_list = data.features
    labels_list = data.labels

    X_train, X_test, Y_train, Y_test = train_test_split(features_list, labels_list, test_size=0.3, train_size=0.7, random_state=1, shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=1)

    train_data = crossJoin(X_train, Y_train)
    val_data = crossJoin(X_val, Y_val)
    test_data = crossJoin(X_test,Y_test)
    print("Train/val/test split done\n")

    ## Balancing Dataset ##
    # 1. Balancing training set
    count=Counter(Y_train)
    class_count=np.array([count[0],count[1]])
    print("0:1 Interruption Ratio - ", count[0],":",count[1])
    weight=1./class_count
    print("Weight:", weight)

    samples_weight = np.array([weight[t] for t in Y_train])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight)) #replacement=False 
    print("Sampler loaded.\n")

    # 2. Balancing validation set
    count_val=Counter(Y_val)
    class_count_val=np.array([count_val[0],count_val[1]])
    print("0:1 Interruption Ratio - ", count_val[0],":",count_val[1])
    weight_val=1./class_count_val
    print("Weight:", weight_val)

    samples_weight_val = np.array([weight_val[t] for t in Y_val])
    samples_weight_val = torch.from_numpy(samples_weight_val)

    sampler_val = WeightedRandomSampler(samples_weight_val, len(samples_weight_val)) #replacement=False
    print("Validation sampler loaded.\n")

    ## Load Data ##
    train_dataloader = DataLoader(train_data, batch_size=64, sampler = sampler)
    val_dataloader = DataLoader(val_data, batch_size=64, sampler = sampler_val)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    print("Train/val/test data loading complete\n")
    return train_dataloader, val_dataloader, test_dataloader, data.p

def examineBatches(train_dataloader, val_dataloader, test_dataloader):
    # Examine batch distribution
    for i, (data, label) in enumerate(train_dataloader):
        count=Counter(label.numpy())
        print("test-batch-{}, 0/1: {}/{}".format(i, count[0], count[1]))

    for i, (data, label) in enumerate(val_dataloader):
        count=Counter(label.numpy())
        print("val-batch-{}, 0/1: {}/{}".format(i, count[0], count[1]))
    
    for i, (data, label) in enumerate(test_dataloader):
        count=Counter(label.numpy())
        print("test-batch-{}, 0/1: {}/{}".format(i, count[0], count[1]))

    return


