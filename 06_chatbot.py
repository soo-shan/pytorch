import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
import itertools

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

# # Data Preprocessing
lines_filepath = os.path.join('data/cornell_movie_dialogs_corpus','movie_lines.txt')

# print(lines_filepath)

# # visualize some lines
# with open(lines_filepath,'r',encoding = 'ISO-8859-14') as file:
#     lines = file.readlines()
# for line in lines[:8]:
#     print(line.strip())


# lineid  characterid   moviesid character_name     Dialogue
# L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
# L1044 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ They do to!
# L985 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ I hope so.
# L984 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ She okay?
# L925 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Let's go.
# L924 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Wow

# Split each line fo the file into a dictionary of fields(lineID, characterID,movieID,character,text)

line_fields = ['lineId','characterId','moviesId','character','text']
lines = {}
with open(lines_filepath,'r',encoding='ISO-8859-1') as f:
    for line in f:
        values = line.split(" +++$+++ ")
        # Extract fields
        lineObj = {}
        for i, field in enumerate(line_fields):
            lineObj[field] = values[i]
        lines[lineObj['lineId']] = lineObj

# print(list(lines.items())[0])

# Group fields of lines from 'loadlines' into conversations 
# based on movie_conversations.txt
conv_fields = ['character1Id','character2Id','movieId','utteranceIds']
conv_filepath = os.path.join('data/cornell_movie_dialogs_corpus','movie_conversations.txt')
# print(conv_filepath)

conversations = []
with open(conv_filepath,'r') as f:
    for line in f:
        values = line.split(' +++$+++ ')
        # Extract fields
        convObj = {}
        for i,field in enumerate(conv_fields):
            convObj[field] = values[i]
        # convert string result from split to list
        lineIds = eval(convObj['utteranceIds'])
        # Reassemble lines
        convObj['lines']=[]
        for lineId in lineIds:
            convObj['lines'].append(lines[lineId])
        conversations.append(convObj)
# print(conversations[0])

# Extract pairs of sentences from conversation
qa_pairs = []
for conversation in conversations:
    # iterate over all the lines of conversation
    for i in range(len(conversation['lines']) - 1):
        inputLine = conversation['lines'][i]['text'].strip()
        targetLine = conversation['lines'][i+1]['text'].strip()
        # Filter wrong samples(if any of the list is empty)
        if inputLine and targetLine:
            qa_pairs.append([inputLine,targetLine])

# Define path to new file
datafile = os.path.join('data/cornell_movie_dialogs_corpus','formatted_movie_lines.txt')
delimiter = '\t'
# unescape the delimiter
delimiter = str(codecs.decode(delimiter,'unicode_escape'))

# write new csv file
print('\nWriting newly formatted file...')
with open(datafile,'w',encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile,delimiter=delimiter)
    for pair in qa_pairs:
        writer.writerow(pair)
print('Done writing to file')

# Visualise some lines
datafile = os.path.join('data/cornell_movie_dialogs_corpus','formatted_movie_lines.txt')
with open(datafile,'rb') as file:
    lines = file.readlines()
for line in lines[:8]:
    print(line)












