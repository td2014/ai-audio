#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:40:21 2017

Code to create some diagnotics of the learning process.

@author: anthonydaniell
"""
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
#single thread
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#
from keras.layers import Conv1D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import scipy.signal as sig
import io
import soundfile as sf
import matplotlib.pyplot as plt
import pickle
import sounddevice as sd
#
from custom_layers import Dense_FreezeKernel

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
subset_len=100000 # reduce full data fields
downsample_factor=10
pf=open(data_train_pickle, 'rb')
data_train_full = np.load(pf)
data_train = sig.decimate(data_train_full[:,:,:subset_len],downsample_factor,axis=2)
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
data_valid = sig.decimate(data_valid_full[:,:,:subset_len],downsample_factor,axis=2)
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
data_test = sig.decimate(data_test_full[:,:,:subset_len],downsample_factor,axis=2)
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

###np.random.seed(42)

#
# Scramble the training data (it is all USA then all OTHER in the file)
#

perm_diag = np.random.permutation(len(data_train))
data_train_scramble = np.reshape(data_train[perm_diag],(-1,data_len_max,1))
label_train_scramble = label_train[perm_diag]

data_valid_reshape = np.reshape(data_valid,(-1,data_len_max,1))
data_test_reshape = np.reshape(data_test,(-1,data_len_max,1))

# for testing/development
subset_min=0
subset_max=6 #data_train.shape[0]  # 6 for diagnostic deep dive
subset_time_min=0
subset_time_max=data_len_max#=10000
data_train_scramble_subset=data_train_scramble[subset_min:subset_max,subset_time_min:subset_time_max,:]
label_train_scramble_subset=label_train_scramble[subset_min:subset_max,:]

#
# Load weights file and recreate earlier processing stages
# Overlay with data timeseries.
#

# Load weights from previous training session
audio_model_diag = load_model('saved_models/config_audio_v40_named.hdf5')

# get index of predicted accent for each image in test set
###audio_predictions_diag_test = [np.argmax(audio_model_diag.predict(np.expand_dims(feature, axis=0))) for feature in data_test_reshape]
# report test accuracy
###accuracy_diag_test = 100*np.sum(np.array(audio_predictions_diag_test)==np.argmax(label_test, axis=1))/len(audio_predictions_diag_test)
###print('Test accuracy (diag): %.4f%%' % accuracy_diag_test)

# Report train accuracy
# get index of predicted accent for each image in test set
###audio_predictions_diag_train = [np.argmax(audio_model_diag.predict(np.expand_dims(feature, axis=0))) for feature in data_train_scramble_subset]
# report test accuracy
###accuracy_diag_train = 100*np.sum(np.array(audio_predictions_diag_train)==np.argmax(label_train_scramble_subset, axis=1))/len(audio_predictions_diag_train)
###print('Train accuracy (diag): %.4f%%' % accuracy_diag_train)

#
# Compute some diagnostics
#

#
# Rebuild relevant sections of the model we are
# interested in exploring.
#
# new model
###fname = 'saved_models/weights.best_audio_v4_named.hdf5'
model = Sequential()
model.add(audio_model_diag.layers[0])  # layer 0 will be loaded from original model
###model.add(audio_model.layers[1])  # will be loaded
model.summary()
model_prediction_diag = model.predict(data_train_scramble_subset)

#
# Create an image which shows output after the maxpooling layer
# superimposed with the audio waveform
#
# -----------------------------
#
#   Image
#
#
#
# -----------------------------
#
#  ~~~~~~waveform~~~~~~~~~~~~~~
#
#

fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
plt.tight_layout()

# extract the desired image
img_idx=0
istart=0
iend=10000
#ax.plot(4000*data_train_scramble[0,istart1:iend1,:],alpha=1.0)
###img = ax.imshow(np.rot90(model_prediction_diag[img_idx,istart:iend,:],1),
###                   cmap='hot', extent=[istart, iend,
###                                       0, model_prediction_diag.shape[2]],
###                   alpha=1.0)

img = ax.imshow(np.rot90(model_prediction_diag[img_idx,istart:iend,:],1),
                   cmap='hot',
                   alpha=1.0)

ax.set_aspect('auto')
plt.colorbar(img)
# Create line plot
center_plot=model_prediction_diag.shape[2]/2 # center line plot on number of filters/2
scale=center_plot*1.0 # adjust scaling to stretch to max range 
ax.plot(scale*data_train_scramble_subset[img_idx,istart:iend,:]+center_plot,alpha=0.4)
# plot weights
conv1D_weights = model.get_weights()[0]
###weight_scale=scale
###ax.plot(scale*data_train_scramble_subset[img_idx,istart:iend,:]+center_plot,alpha=0.4)

# Create a concatenated array of all the training samples:
flat_train = np.ndarray.flatten(data_train_scramble)
# Create a concatenated array of all the test samples:
flat_test = np.ndarray.flatten(data_test_reshape)
#
# End of script
#
