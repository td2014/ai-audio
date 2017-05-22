#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def soundfile_preprocess(inputDir, outputDir, pathto_ffmpeg, pickleFile):
    """
    Created on Sun May 21 14:55:06 2017

    Preprocess the soundfiles in preparation for use with NN codes.
    This script takes the mp3 files and converts to wave (if not already done), 
    finds max length, sample rate, max and min value, and stores in metadata file.

    @author: anthonydaniell
    """

    from os import listdir, system
    from os.path import isfile, join
    import soundfile as sf
    import pickle

#
# Get list of files in input (only include mp3)
#
    filesToProcess = [f for f in listdir(inputDir) if isfile(join(inputDir, f)) and f[len(f)-4:]=='.mp3']

#
# Loop over list.  If equivalent wavefile exists in outputDir skip
#

    for iFile in filesToProcess:
        fullFile = outputDir+iFile[:len(iFile)-4] + '.wav'
        if isfile(fullFile):
            continue
        else:
            print('need to add file')
            command = pathto_ffmpeg + ' -i ' + inputDir+iFile + ' ' + fullFile
            print('command = ', command)
            res = system(command)
            print('res = ', res)

#
# Read in each wave file, and retain length, max, min value, and sample rate
#

    newInputDir = outputDir

    wav_filesToProcess = [f for f in listdir(newInputDir) if isfile(join(newInputDir, f)) and f[len(f)-4:]=='.wav']

# Load WaveFiles
    soundFileMetadata_full = []
    for iFile in wav_filesToProcess:
        soundFileMetadata = []
        currFile = newInputDir+iFile
        data, samplerate = sf.read(currFile)
    # Determine key parameters.
        soundFileMetadata.append('filename:')
        soundFileMetadata.append(iFile)
        soundFileMetadata.append('sample_rate:')
        soundFileMetadata.append(samplerate)
        soundFileMetadata.append('data_length:')
        soundFileMetadata.append(len(data))
        soundFileMetadata.append('max_data_val:')
        soundFileMetadata.append(data.max())
        soundFileMetadata.append('min_data_val:')
        soundFileMetadata.append(data.min())

    # Store information for this entire record
        soundFileMetadata_full.append(soundFileMetadata)
#
# Pickle metadata file
#

    pf = open(pickleFile,'wb')
    pickle.dump(soundFileMetadata_full, pf)
    pf.close()
    
    return 0  # normal exit

#
# End of script
#