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

# # Part 1 Data Preprocessing
lines_filepath = os.path.join('data/cornell_movie_dialogs_corpus','movie_lines.txt')
conv_filepath = os.path.join('data/cornell_movie_dialogs_corpus','movie_conversations.txt')
print(lines_filepath)
print(conv_filepath)

# visualize some lines
with open(lines_filepath,'r',encoding = 'ISO-8859-14') as file:
    lines = file.readlines()

for line in lines[:8]:
    print(line.strip())


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
            lineObj[]










