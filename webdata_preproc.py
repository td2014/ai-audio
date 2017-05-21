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
# The .attrib field contains the uri of the detail page
# The .text field contains the tag name, such as "english1" of the 
# speaker, which matches up with the .mp3 files
#
#
# Extract out the speaker details and soundfiles from the .attrb field
# of each speaker entry
#
speaker_attributes_details = []
speaker_soundfiles = []
speaker_links_test = []
speaker_links_test.append(speaker_links[0])
speaker_links_test.append(speaker_links[1])
for speaker_object in speaker_links_test:
    # extract the current link from the href attribute
    print('in here.')
    curr_url = speaker_object.attrib['href']
    # extract the current tag, which matches to the soundfile, e.g. "english1"
    curr_tag = speaker_object.text
    
    print('curr_url = ', curr_url)
    print('curr_tag = ', curr_tag)
    print('curr_page = ', rootpage+curr_url)
    # form the full url path for the current speaker and load the page
    page = requests.get(rootpage+curr_url)
    tree = html.fromstring(page.content)
    # Extract biographical details
    speaker_bio = tree.xpath('//ul[@class="bio"]/node()')
    
    #
    # Parse speaker_bio into human readable components
    #
    speaker_bio_human_readable = []
    #
    # Add join field(s)
    #
    speaker_bio_human_readable.append('speaker_tag:')
    speaker_bio_human_readable.append(curr_tag)
    speaker_bio_human_readable.append('speaker_page:')
    speaker_bio_human_readable.append(rootpage+curr_url)
    # Loop over elements in bio structure and parse out.
    for component in speaker_bio:
        try:
            # Extract out the keyword value pairs for the speaker bio items
            if component.tag == 'li':  # list item
                keyword = component.getchildren()[0].text
                value = component.getchildren()[0].tail
                speaker_bio_human_readable.append(keyword)
                speaker_bio_human_readable.append(value)
        except: #ignore spacing characters
            continue
    
    speaker_attributes_details.append(speaker_bio_human_readable)
    # Add soundfile link
    speaker_soundinfo = tree.xpath('//audio[@id="player"]/node()')
    sf_url = speaker_soundinfo[1].attrib['src']
    speaker_soundfiles.append(sf_url)

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