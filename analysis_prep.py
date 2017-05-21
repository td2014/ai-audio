#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:52:33 2017

Input speaker metadata and reformat for 
determining how to best partition the data.

@author: anthonydaniell
"""
import pickle

inputDir = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/speaker_metadata/'

pickleFile = inputDir+'speaker_attributes_details.out'
pf = open(pickleFile,'rb')
sf_data = pickle.load(pf)
pf.close()

#
# Loop over data and create a csv output
#
outFile = open(inputDir+'speaker_attributes_details.csv','wt')

headerSet = False
headerLine = ''
for iRecord in sf_data:
    currLine = ''
    for index in range(0, len(iRecord),2):
        keyname = iRecord[index]
        value = iRecord[index+1]
        if not headerSet:
            headerLine = headerLine+keyname+'\t'
        currLine = currLine+value+'\t'
    if headerSet==False:
        outFile.write(headerLine+'\r')
        headerSet=True
    
    outFile.write(currLine+'\r')

outFile.close()
#
# End of script
#