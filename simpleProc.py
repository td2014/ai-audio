#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:29:49 2017

@author: anthonydaniell
"""
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import numpy as np
import io
import soundfile as sf
import matplotlib.pyplot as plt

#
# Load some data
#
inputPath = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/wiki_IPA_examples/'
inputFile = 'Postalveolar_approximant.ogg'
inputFile2 = 'Voiced_labiodental_fricative.ogg'

currFile = inputPath+inputFile
data, samplerate = sf.read(currFile)

currFile2 = inputPath+inputFile2
data2, samplerate2 = sf.read(currFile2)

##rms = [np.sqrt(np.mean(block**2)) for block in
##       sf.blocks('myfile.wav', blocksize=1024, overlap=512)]

print('samplerate = ', samplerate)
print('data[0] = ', data[0])
print('samplerate2 = ', samplerate2)

#
# Extract a phoneme filter
#
postalV_approximant = data[63000:68000]

#
# Plot the data
#

plt.plot(postalV_approximant)
plt.show()
plt.plot(data2[20000:25000])
plt.show()

#
# Compute a correlation between a filter and the full data set
#

sampleIndex = 0
audCorrSum=np.zeros(len(data)-len(postalV_approximant))
while sampleIndex+len(postalV_approximant) < len(data):
    audCorr = np.multiply(data[sampleIndex:sampleIndex+len(postalV_approximant)],postalV_approximant)
    audCorrSum[sampleIndex] = np.sum(audCorr)
    sampleIndex = sampleIndex+1


#
# Plot result
#
plt.plot(audCorrSum)
plt.show()

#
# Pass list of phonemes to distribution calculator
#

#
# Split data into train, validation, test sets
#
# Two Categories:  US English Speaker vs. Outside US English Speaker
# 


#
# Train NN to classifier vector into US vs. outside US speakers
#
audio_model = Sequential()

# Include a two layer MLP for the classifier backend
audio_model.add(Dense(512, activation='relu'))
audio_model.add(Dropout(0.2))
audio_model.add(Dense(512, activation='relu'))
audio_model.add(Dropout(0.2))

# We want probabilities over the accent classes.
audio_model.add(Dense(2,activation='softmax'))

# Output architecture and compile
audio_model.summary()
audio_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# Compute Cross-Validation Accuracy
#
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.audio.hdf5', 
                               verbose=1, save_best_only=True)

audio_model.fit(train_audio, train_targets, 
          validation_data=(valid_audio, valid_targets),
          epochs=10, batch_size=20, callbacks=[checkpointer], verbose=1)

#
# Compute Test Accuracy
#
audio_model.load_weights('saved_models/weights.best.audio.hdf5')

# get index of predicted dog breed for each image in test set
audio_predictions = [np.argmax(audio_model.predict(np.expand_dims(feature, axis=0))) for feature in test_audio]

# report test accuracy
test_accuracy = 100*np.sum(np.array(audio_predictions)==np.argmax(test_targets, axis=1))/len(audio_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

#
# End of script
#