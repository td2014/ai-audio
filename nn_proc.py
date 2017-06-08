#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:52:18 2017

Create NN model and evaluate.

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
from keras.initializers import glorot_uniform
from keras.constraints import unit_norm
from keras.layers import Conv1D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import scipy.signal as sig
import io
import matplotlib.pyplot as plt
import pickle
import sounddevice as sd
#
from custom_layers import Dense_FreezeKernel

#
# print out some random numbers to verify we are set up consistently 
# from run to run
#

###print('numpy random (start) = ', np.random.uniform())
###print('python random (start) = ', rn.uniform(0,1))
###sess = tf.Session()
###with sess.as_default():
####print('tensorflow random (start) = ', K.random_uniform_variable((2,3), 0, 1).eval(session=sess))

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

savedmodel_path= \
'/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/CodeBase/ai_audio/saved_models/'

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

#
# Scramble the training data (it is all USA then all OTHER in the file)
#

perm = np.random.permutation(len(data_train))
data_train_scramble = np.reshape(data_train[perm],(-1,data_len_max,1))
label_train_scramble = label_train[perm]

data_valid_reshape = np.reshape(data_valid,(-1,data_len_max,1))
data_test_reshape = np.reshape(data_test,(-1,data_len_max,1))

# for testing/development
subset_min=0
subset_max=data_train.shape[0]  # 6 for diagnostic deep dive
subset_time_min=0
subset_time_max=data_len_max #=10000
data_train_scramble_subset=data_train_scramble[subset_min:subset_max,subset_time_min:subset_time_max,:]
label_train_scramble_subset=label_train_scramble[subset_min:subset_max,:]

#
# The architecture should be
#
# 1:  num_sound_samples x 512 filters
#
# 2:  Add the 1D convolution
#
# 3:  Max pool down to 256 (number of characters in snippet) x 512 phonemes
#
# 4:  Pool sum -> 1x512  (Sum across the entire field)
#
try:
    audio_model.get_weights()
    del audio_model
    print ('cleanup audio model from before.')
except:
    print ('no audio model from before.')

model_name='audio_v43_named'
audio_model = Sequential()

# Detect speech component in waveforms.
audio_model.add(Conv1D(4, kernel_size=1024, strides=1, activation='relu', 
                       input_shape=(data_len_max,1), name='conv_1'))

# Collapse down to number of characters (should be based on time of snippet)

audio_model.add(GlobalMaxPooling1D(name='maxpool_1')) #100 time samples per letter
# Include a two layer multilayer perceptron for the classifier backend
audio_model.add(Dense(16, activation='relu', name='dense_1'))
audio_model.add(Dropout(0.5, name='dropout_1'))

# We want probabilities over the accent classes (2 for now)
audio_model.add(Dense(2,activation='softmax', name='dense_3'))  # Baseline
# for testing - force the input filters to exactly replicate the input signals
###def my_init(shape, dtype=None):
###    init_weights = np.zeros([6,2],dtype=dtype)
###    init_weights[0,1] = 1
###    init_weights[1,1] = 1
###    init_weights[2,1] = 1
###    init_weights[3,0] = 1
###    init_weights[4,0] = 1
###    init_weights[5,0] = 1
###    return init_weights

###audio_model.add(Dense_FreezeKernel(2,activation='linear', name='dense_3', 
###                      kernel_initializer=my_init,
###                      use_bias=False))

# Output architecture and compile
audio_model.summary()
audio_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
#
# Compute Cross-Validation Accuracy
#
###modelweights_filepath=savedmodel_path+'weights.best_'+model_name+'.hdf5'
###checkpointer = ModelCheckpoint(filepath=modelweights_filepath, 
###                              verbose=1, save_best_only=False)

###audio_model.fit(data_train_scramble, label_train_scramble, 
###          validation_data=(data_valid_reshape, label_valid),
###          epochs=1, batch_size=20, callbacks=[checkpointer], shuffle=False, verbose=0)
shuffle_val=False

###audio_model.fit(data_train_scramble, label_train_scramble, 
###          validation_data=(data_valid_reshape, label_valid),
###          epochs=20, batch_size=20, shuffle=shuffle_val, verbose=1)

# subset testing
audio_model.fit(data_train_scramble_subset, 
                label_train_scramble_subset, 
                epochs=20, batch_size=subset_max-subset_min, 
                shuffle=shuffle_val, verbose=1)


###audio_model.save(savedmodel_path+'config_'+model_name+'.hdf5')
# make sure we got a good saved model.
###del audio_model 
#
# Compute Test Accuracy
#
###audio_model = load_model(savedmodel_path+'config_'+model_name+'.hdf5')

# get index of predicted accent for each image in test set
audio_predictions = [np.argmax(audio_model.predict(np.expand_dims(feature, axis=0))) for feature in data_test_reshape]

# report test accuracy
test_accuracy = 100*np.sum(np.array(audio_predictions)==np.argmax(label_test, axis=1))/len(audio_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

#dump out weights

print()
print('shuffle above = ', shuffle_val)
print('audio_model.get_weights()[0][0] = ', audio_model.get_weights()[0][0])
print()

#
# Save full model
#
full_model_savename = savedmodel_path+'config_'+model_name+'.hdf5'
###audio_model.save(full_model_savename)

print('model = ',model_name)
print('model file = ', full_model_savename)
print('====')

#
# Check random state at end of processing
#

###print('numpy random (end) = ', np.random.uniform())
###print('python random (end) = ', rn.uniform(0,1))
##sess = tf.Session()
##with sess.as_default():
##    print('tensorflow random (end) = ', tf.random_uniform(shape=(1,)).eval())


#
# End of script
#
