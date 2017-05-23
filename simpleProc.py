#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:29:49 2017

Script to load up data, and run through NN training and test evaluation.

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

inputDir_SpeakerMetadata = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/speaker_metadata/'
inputDir_WAV = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/wav/'
speaker_attributes_split_pickle = 'speaker_attributes_details_split.out'
soundfile_metadata_pickle = 'soundFileMetadata.out'

#
# Load metadata with class tags appended
#

pf = open(inputDir_SpeakerMetadata+speaker_attributes_split_pickle,'rb')
speaker_attributes_split = pickle.load(pf)
pf.close()

#
# Load metadata for sound files.  Scan for max values
#

pf = open(inputDir_WAV+soundfile_metadata_pickle,'rb')
soundfile_metadata = pickle.load(pf)
pf.close()

#
# Create common dictionary
#

processDict = {}

#
# Loop over data, extracting appropriate files and grouping
# into train, validation, and test sets with appropriate labels
# Skip over the "IGNORE" class.
#
# Add padding on sounddata to take into account convolution and
# to make each dataset the same length.
#

#
# Preread to get counts
#
# Keep track of number of each label
USA_class_count=0 
OTHER_class_count=0

for iRecord in speaker_attributes_split:
    for idx in range(len(iRecord)):
        if iRecord[idx]=='speaker_tag:':
            keyname = iRecord[idx+1] # key for dictionary 
        if iRecord[idx]=='class_tag:':
            if iRecord[idx+1]=='USA':
                USA_class_count=USA_class_count+1
                break
            elif iRecord[idx+1]=='OTHER':
                OTHER_class_count=OTHER_class_count+1
                break
            else:
                break
    processDict[keyname]=iRecord  # add to dictionary
# Find minimum of each label to create balances train/test sets            
minCommon = min(USA_class_count,OTHER_class_count)

#
# Preread to get boundaries
#
global_max_samplerate=float('-inf') # maximum sample rate of any data
global_min_samplerate=float('inf') # minimum sample rate of any data
global_max_length=-1  # maximum length of all data
global_max_data_val=float('-inf') # maximum amplitude of any data
global_min_data_val=float('inf') # minimum amplitude of any data

for iRecord in soundfile_metadata:
    for idx in range(len(iRecord)):
        if iRecord[idx]=='filename:':
            keyfile=iRecord[idx+1]  # use for joining data
        if iRecord[idx]=='sample_rate:':
            if iRecord[idx+1] > global_max_samplerate:
                global_max_samplerate=iRecord[idx+1]
            elif iRecord[idx+1] < global_min_samplerate:
                global_min_samplerate=iRecord[idx+1]
            else:
                continue
        elif iRecord[idx]=='data_length:':
            if iRecord[idx+1] > global_max_length:
                global_max_length=iRecord[idx+1]
            else:
                continue
        elif iRecord[idx]=='max_data_val:':
            if iRecord[idx+1] > global_max_data_val:
                global_max_data_val=iRecord[idx+1]
            else:
                continue
        elif iRecord[idx]=='min_data_val:':
            if iRecord[idx+1] < global_min_data_val:
                global_min_data_val=iRecord[idx+1]
            else:
                continue
        else:
            continue

#
# Add to corresponding dictionary
#

    keylookup=keyfile[:len(keyfile)-4]+','  #keyname has comma added
    tempVal = processDict[keylookup]
    for iMini in range(len(iRecord)):
        tempVal.append(iRecord[iMini])
    processDict[keylookup]=tempVal  # update dictionary

# do some boundary checks
if global_min_data_val < -1.0:
    print('Data needs to be rescaled.  Min less than -1.0.')
if global_max_data_val > 1.0:
    print('Data needs to be rescaled.  Max greater than 1.0')


#
# Do second pass, now loading up sound data and padding.
# Only use data which is 44100 Hz sample rate
#

USA_count=0
OTHER_count=0

for dataIdx in range(len(speaker_attributes_split)):
    print(speaker_attributes_split[dataIdx])
    print(soundfile_metadata[dataIdx])


# training data
####currFile = inputPath+inputFile_train
####data_train, samplerate_train = sf.read(currFile)
# validation data
####currFile = inputPath+inputFile_valid
####data_valid, samplerate_valid = sf.read(currFile)


#
# Split data into train, validation, test sets
#
# Two Categories:  US English Speaker vs. Outside US English Speaker
# 
###data_len_max=800000 # set maximum input size to NN
####train_audio = np.zeros([1,data_len_max,1])
####train_labels = np.zeros([1,1])
#
####valid_audio = np.zeros([1,data_len_max,1])
####valid_labels = np.zeros([1,1])

# Set up training data
####train_audio[0,:,0] = data_train[:min(data_len_max,len(data_train))]
####train_labels[0,0] = 1  # US Based speaker.
# Set up validation data
####valid_audio[0,:,0] = data_valid[:min(data_len_max,len(data_valid))]
####valid_labels[0,0] = 0  # Outside US Based speaker.
#
# Convert to one-hot encoding
#
####train_labels_onehot = np_utils.to_categorical(train_labels, 2)
####valid_labels_onehot = np_utils.to_categorical(valid_labels, 2)
#
# 
#
####print('train_audio.shape = ', train_audio.shape)
####print('valid_audio.shape = ', valid_audio.shape)
#
####print('train_labels.shape = ', train_labels.shape)
####print('valid_labels.shape = ', valid_labels.shape)

####print('train_labels_onehot.shape = ', train_labels_onehot.shape)
####print('valid_labels_onehot.shape = ', valid_labels_onehot.shape)
####print('train_labels_onehot = ', train_labels_onehot)
####print('valid_labels_onehot = ', valid_labels_onehot)

####plt.plot(train_audio[0,:,0])
####plt.show()
####plt.plot(valid_audio[0,:,0])
####plt.show()

#
# Train NN to classifier vector into US vs. outside US speakers
#
####np.random.seed(42)
####audio_model = Sequential()
#
# The architecture should be
#
# 1:  num_sound_samples x 512 filters
#
# Add the 1D convolution
#
# Max pool down to 256 (number of characters in snippet) x 512 phonemes
#
# Pool sum -> 1x512  (Sum across the entire field)
#

####audio_model.add(Conv1D(128, kernel_size=4096, strides=1, activation='relu', input_shape=(data_len_max,1)))

# Collapse down to number of characters (should be based on time of snippet)
####audio_model.add(MaxPooling1D(4096)) #100 time samples per letter
# Include a two layer MLP for the classifier backend
####audio_model.add(Dense(512, activation='relu'))
####audio_model.add(Dropout(0.2))
####audio_model.add(Dense(512, activation='relu'))
####audio_model.add(Dropout(0.2))

# Collapse down to 1,512
####audio_model.add(GlobalAveragePooling1D())
# We want probabilities over the accent classes (2 for now)
####audio_model.add(Dense(2,activation='softmax'))

# Output architecture and compile
####audio_model.summary()
####audio_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# Compute Cross-Validation Accuracy
#
####checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.audio.hdf5', 
####                               verbose=1, save_best_only=True)

####audio_model.fit(train_audio, train_labels_onehot, 
####          validation_data=(valid_audio, valid_labels_onehot),
####          epochs=3, batch_size=1, callbacks=[checkpointer], verbose=1)

#
# Compute Test Accuracy
#
###audio_model.load_weights('saved_models/weights.best.audio.hdf5')

# get index of predicted dog breed for each image in test set
###audio_predictions = [np.argmax(audio_model.predict(np.expand_dims(feature, axis=0))) for feature in test_audio]

# report test accuracy
###test_accuracy = 100*np.sum(np.array(audio_predictions)==np.argmax(test_targets, axis=1))/len(audio_predictions)
###print('Test accuracy: %.4f%%' % test_accuracy)

#
# End of script
#