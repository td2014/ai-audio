#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 20:02:41 2017

Script to preprocess the data for use with the NN.

@author: anthonydaniell
"""
from lxml import html
import requests

#
# Load the webpage with the accent examples
#
rootpage = 'http://accent.gmu.edu/'
page = requests.get(rootpage+'browse_language.php?function=find&language=english')
tree = html.fromstring(page.content)

#
# Extract some key information
#
#This will obtain list of speakers:
maincontent_list = tree.xpath('//div[@class="content"]/p/node()')

# Split into information about speaker, and hyperlink to details page & soundfile
m_index=0
speaker_links = []
speaker_attributes = []
while m_index < len(maincontent_list):
    speaker_links.append(maincontent_list[m_index])
    m_index = m_index+1
    speaker_attributes.append(maincontent_list[m_index])
    m_index = m_index+1

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
# End of script
#