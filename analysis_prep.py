#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def analysis_prep(inputDir, inputPickle, outputDir, outputCSV):
    """
    Created on Sun May 21 15:52:33 2017

    Input speaker metadata and reformat for 
    determining how to best partition the data.

    @author: anthonydaniell
    """
    import pickle

    pickleFile = inputDir+inputPickle
    pf = open(pickleFile,'rb')
    sf_data = pickle.load(pf)
    pf.close()

#
# Loop over data and create a tab seperated output (csv file)
#
    outFile = open(outputDir+outputCSV,'wt')

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
    
    return 0 # normal exit
#
# End of script
#