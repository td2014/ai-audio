#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:51:18 2017

Script to split the data into desired groups.

@author: anthonydaniell
"""
import pickle

inputDir_SpeakerMetadata = '/Users/anthonydaniell/Desktop/FilesToStay/OnlineCourses/AI_NanoDegree/Term2/CapstoneProject/RawData/speaker_metadata/'
speaker_attributes_pickle = 'speaker_attributes_details.out'

pf = open(inputDir_SpeakerMetadata+speaker_attributes_pickle,'rb')
data = pickle.load(pf)
pf.close()

#
# Loop over data and split based on critera.
#
USA_Count=0
OTHER_Count=0
IGNORE_Count=0
splitRecord = []
for iRecord in data:
    print (iRecord)
    USA_Born_Flag=False
    USA_Live=False
    
    for iScan in range(len(iRecord)):
        if iRecord[iScan] == 'birth place:' and iRecord[iScan+1].find('usa')>=0:
            USA_Born_Flag=True
        elif iRecord[iScan] == 'age, sex:':
            speaker_age = int(float(iRecord[iScan+1].split(',')[0])) #in case we have fractional years
            speaker_sex = iRecord[iScan+1].split(',')[1].strip()
        elif iRecord[iScan] == 'english residence:' and iRecord[iScan+1]==' usa':
            USA_Live =True
        elif iRecord[iScan] == 'length of english residence:':
            age_of_residence= int(float(iRecord[iScan+1].split(' ')[1])) #in case we have fractional years
        else:
            continue
              
    iRecord.append('class_tag:')
    if USA_Born_Flag and USA_Live and speaker_age==age_of_residence:
        iRecord.append('USA')
        USA_Count = USA_Count+1
    elif speaker_age==age_of_residence:
        iRecord.append('OTHER')
        OTHER_Count = OTHER_Count+1
    else:
        iRecord.append('IGNORE')
        IGNORE_Count = IGNORE_Count+1
        
    splitRecord.append(iRecord)

#
# End of script
#