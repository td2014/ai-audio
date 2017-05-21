#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:15:36 2017

Download the soundfiles for the speakers.

@author: anthonydaniell
"""

import urllib.request
import pickle
import time

#
# Get metadata and set up output directory
#

inputDir = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/speaker_metadata/'

pickleFile = inputDir+'speaker_attributes_details.out'
pf = open(pickleFile,'rb')
sf_data = pickle.load(pf)
pf.close()

outputDir = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/mp3/'

#
# Loop over sf_data and download each soundfile
#
object_count = 0
for speaker_object in sf_data:
    print('object count = ', object_count)
    
    # look for the soundfile keyword, the next item is the corresponding url
    for index in range(len(speaker_object)):
        if speaker_object[index] != 'speaker_soundfile:':
            continue
        else:
            soundfile_url = speaker_object[index+1]
            print('soundfile_url = ', soundfile_url)
            break
   
    soundtracks_idx = soundfile_url.index('soundtracks/')
    savefile = outputDir+soundfile_url[soundtracks_idx+12:] # get item after soundtracks
    print ('savefile = ', savefile)
    urllib.request.urlretrieve(soundfile_url,savefile)
    object_count = object_count+1
    time.sleep(0.2) # avoid overwhelming server

#
# Download the examples given their locations from the main webpage
#

#
# Transform from mp3 to wav (easier to work with in Python)
#

#
# Determine key audio parameters and add to information.
#

#
# Pickle metadata files for later use by NN processing code.
#

###pickleFile = outputDir+'speaker_attributes_details.out'
###pf = open(pickleFile,'wb')
###pickle.dump(speaker_attributes_details, pf)
###pf.close()

#
# End of script
#