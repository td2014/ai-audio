#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:31:26 2017

Preprocessing driver.  This script calls all the functions that
prepare the audio data for use with the neural network.

@author: anthonydaniell
"""
import analysis_prep as ap
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
# using webdata_preproc.py script.
#
# Produce reformatted version for human readability
#

res = ap.analysis_prep(outputDir_SpeakerMetadata, speaker_attributes_pickle, outputDir_SpeakerMetadata, speaker_attributes_CSV)

#
# Download the soundfiles using soundfile_download.py script.
#
# Preprocess and standardize the soundfiles
#

####res = soundfile_preprocess(outputDir_MP3, outputDir_WAV, pathto_ffmpeg, soundfile_Metadata)

#
# End of script
#