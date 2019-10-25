# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
import csv
import random
import os
import codecs
import itertools
import utils
from torch import cuda, device
from vocab import Vocabulary


CUDA = cuda.is_available()
device = device("cuda" if CUDA else "cpu")

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

# # write new csv file
# print('\nWriting newly formatted file...')
# with open(datafile,'w',encoding='utf-8') as outputfile:
#     writer = csv.writer(outputfile,delimiter=delimiter)
#     for pair in qa_pairs:
#         writer.writerow(pair)
# print('Done writing to file')

# # Visualise some lines
# datafile = os.path.join('data/cornell_movie_dialogs_corpus','formatted_movie_lines.txt')
# with open(datafile,'rb') as file:
#     lines = file.readlines()
# for line in lines[:8]:
#     print(line)

# Read the datafile and split into lines
print('Reading and processing file. \nPlease wait ...')
lines = open(datafile,encoding='utf-8').read().strip().split('\n')
# Split every line into pairs and normalize
pairs = [[utils.normalizeString(s) for s in pair.split('\t')] for pair in lines]
print('Done Reading!')

# Instantial a vocabulary class
voc = Vocabulary('Cornell Movie-Dialogue Corpus')

pairs = utils.filterPairs(pairs,MAX_LENGTH = 10)
print('After filtering, there are {} conversation pairs'.format(len(pairs)))

# Loop through each pair and add them to the vocabulary
for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])

print('Counted Words:',voc.num_words)

# Minimum word frequency threshold for trimming
MIN_COUNT = 3
# Trim vocabulary
voc.trim(min_count = MIN_COUNT)
# Filter out all pairs of sentences which have trimmed words
keep_pairs = []
for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True
    # check input sentence
    for word in input_sentence.split(' '):
        if word not in voc.word2index:
            keep_input = False
            break
    for word in output_sentence.split(' '):
        if word not in voc.word2index:
            keep_output = False
            break
    # Only keep the pairs that do not contain trimmed words in their input or output sentence
    if keep_input and keep_output:
        keep_pairs.append(pair)

print('Trimmed from {} pairs to {}, {:.2f}% of total remaining'.format(len(pairs),len(keep_pairs), 100*len(keep_pairs)/len(pairs)))

# If we are interested in speeding up training and/or would like to 
# leverage GPU parallelization capabilitites, we will need to train with 
# mini-batches. Using mini-batches also means that we must be mindful of the
# variation of sentence length in our batches. To accomodate sentences of different
# sizes in teh same batch, we will make our batched input tensor of 
# shape(max_length, batch size), where sentences shorter than the max_length are 
# zero padded after an EOS_token. If we simply convert our English sentences to 
# tensors by converting words to thier indexes(indexFromSentence) and zero-pad,
# our tesor would have shape(batch_size, max_length) and indexing the first dimension
# would return a full sequence across all time-steps. However, we need to be able
# to index our batch along time and across all sequences in teh batch. Therefore,
# we transpose our input batch shape to (max_length, batch_size) so that indexing
# across the first dimension returns a time step across all sentences in the batch.
# We handle this transpose implicitly in the zeroPadding function.

# indexes from sentences
# utils.indexesFromSentence(voc,keep_pairs[1][0]) # Testing function

# # Example for validation
# small_batch_size = 5
# batches = utils.batch2TrainData(voc, [random.choice(keep_pairs) for _ in range(small_batch_size)])
# input_variable, lengths, target_variable, mask, max_target_len = batches

# print('Input Variable: ')
# print(input_variable)
# print('Lengths: ', lengths)
# print('target_variable: ')
# print(target_variable)
# print('mask: ')
# print(mask)
# print('max_target_len:', max_target_len)















