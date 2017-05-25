#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:40:21 2017

Code to create some diagnotics of the learning process.

@author: anthonydaniell
"""
from keras.layers import Conv1D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np
import io
import soundfile as sf
import matplotlib.pyplot as plt
import pickle
import sounddevice as sd

#
# Read in and preprocess data similar to what is done for the
# nn training
#

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
subset_len=1000000 # reduce full data fields
pf=open(data_train_pickle, 'rb')
data_train_full = np.load(pf)
data_train = data_train_full[:,:,:subset_len]
pf.close()

pf=open(label_train_pickle, 'rb')
label_train = np.load(pf)
pf.close()

pf=open(file_train_pickle, 'rb')
file_train = pickle.load(pf)
pf.close()

###
pf=open(data_valid_pickle, 'rb')
data_valid_full = np.load(pf)
data_valid = data_valid_full[:,:,:subset_len]
pf.close()

pf=open(label_valid_pickle, 'rb')
label_valid = np.load(pf)
pf.close()

pf=open(file_valid_pickle, 'rb')
file_valid = pickle.load(pf)
pf.close()

###
pf=open(data_test_pickle, 'rb')
data_test_full = np.load(pf)
data_test = data_test_full[:,:,:subset_len]
pf.close()

pf=open(label_test_pickle, 'rb')
label_test = np.load(pf)
pf.close()

pf=open(file_test_pickle, 'rb')
file_test = pickle.load(pf)
pf.close()

#
# Train NN to classifier vector into US vs. outside US speakers
#
data_len_max = data_train.shape[2] # Common max length

np.random.seed(42)

#
# Scramble the training data (it is all USA then all OTHER in the file)
#

perm = np.random.permutation(len(data_train))
data_train_scramble = np.reshape(data_train[perm],(-1,data_len_max,1))
label_train_scramble = label_train[perm]

data_valid_reshape = np.reshape(data_valid,(-1,data_len_max,1))
data_test_reshape = np.reshape(data_test,(-1,data_len_max,1))

#
# Load weights file and recreate earlier processing stages
# Overlay with data timeseries.
#
audio_model = Sequential()
audio_model.load_weights('saved_models/weights.best_v3.audio.hdf5')

# get index of predicted dog breed for each image in test set
audio_predictions = [np.argmax(audio_model.predict(np.expand_dims(feature, axis=0))) for feature in data_test_reshape]

# report test accuracy
test_accuracy = 100*np.sum(np.array(audio_predictions)==np.argmax(label_test, axis=1))/len(audio_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

#
# Compute some diagnostics
#

ma_test = np.zeros([data_len_max])

for idx in range(data_len_max-20):
    ma_test[idx]=np.mean(data_test[97,0,idx:idx+20])

#
# End of script
#