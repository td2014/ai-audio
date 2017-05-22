#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:31:26 2017

Preprocessing driver.  This script calls all the functions that
prepare the audio data for use with the neural network.

@author: anthonydaniell
"""

#
# Set up the environment
#

outputDir_SpeakerMetadata = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/speaker_metadata/'
outputDir_MP3 = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/mp3/'
outputDir_WAV = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/wav/'

# mp3 to wav conversion program
pathto_ffmpeg = '/Users/anthonydaniell/Downloads/Lion_Mountain_Lion_Mavericks_Yosemite_El-Captain_15.05.2017/ffmpeg'

# file setups
speaker_attributes_pickle = 'speaker_attributes_details.out'
speaker_attributes_CSV = 'speaker_attributes_details.csv'
soundfile_Metadata = outputDir_WAV+'soundFileMetadata.out'
#
# Download the webpage information.
#

# function version not tested - for documentation purposes
####res = webdata_preproc_fn(outputDir_SpeakerMetadata, speaker_attributes_pickle)

#
# Produce reformatted version for human readability
#

# function version not tested - for documentation purposes
###res = analysis_prep(outputDir_SpeakerMetadata, speaker_attributes_pickle, outputDir_SpeakerMetadata, speaker_attributes_CSV)

#
# Download the soundfiles
#

# function version not tested - for documentation purposes
###res = soundfile_download_fn(outputDir_SpeakerMetadata, speaker_attributes_pickle, outputDir_MP3)

#
# Preprocess and standardize the soundfiles
#

res = soundfile_preprocess(outputDir_MP3, outputDir_WAV, pathto_ffmpeg, soundfile_Metadata)

#
# End of script
#