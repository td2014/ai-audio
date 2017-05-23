#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:52:18 2017

Create NN model and evaluate.

@author: anthonydaniell
"""

from keras.layers import Conv1D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np
import io
import soundfile as sf
import matplotlib.pyplot as plt
import pickle


#
# Setup some directories and files
#


outputDir_pickle_files='/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/pickle_files/'

data_train_pickle=outputDir_pickle_files+'data_train_pickle.out'
label_train_pickle=outputDir_pickle_files+'label_train_pickle.out'
file_train_pickle=outputDir_pickle_files+'file_train_pickle.out'

data_valid_pickle=outputDir_pickle_files+'data_valid_pickle.out'
label_valid_pickle=outputDir_pickle_files+'label_valid_pickle.out'
file_valid_pickle=outputDir_pickle_files+'file_valid_pickle.out'

data_test_pickle=outputDir_pickle_files+'data_test_pickle.out'
label_test_pickle=outputDir_pickle_files+'label_test_pickle.out'
file_test_pickle=outputDir_pickle_files+'file_test_pickle.out'

#
# Load pickle files
#

pf=open(data_train_pickle, 'rb')
data_train = np.load(pf)
pf.close()

pf=open(label_train_pickle, 'rb')
label_train = np.load(pf)
pf.close()

pf=open(file_train_pickle, 'rb')
file_train = pickle.load(pf)
pf.close()

###
pf=open(data_valid_pickle, 'rb')
data_valid = np.load(pf)
pf.close()

pf=open(label_valid_pickle, 'rb')
label_valid = np.load(pf)
pf.close()

pf=open(file_valid_pickle, 'rb')
file_valid = pickle.load(pf)
pf.close()

###
pf=open(data_test_pickle, 'rb')
data_test = np.load(pf)
pf.close()

pf=open(label_test_pickle, 'rb')
label_test = np.load(pf)
pf.close()

pf=open(file_test_pickle, 'rb')
file_test = pickle.load(pf)
pf.close()


#
# End of script
#
