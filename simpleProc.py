#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:29:49 2017

@author: anthonydaniell
"""

import numpy as np
import io
import soundfile as sf
import matplotlib.pyplot as plt

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
# End of script
#