import unicodedata
import re
import itertools
import torch

from vocab import PAD_token, SOS_token, EOS_token

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    
    # Replace any .?! by a whitespace + the character
    # ie '!' replaced by ' !' (without quotes)
    # \1 implies first bracketed group ie [,!?]
    # r is to escape backslash
    s = re.sub(r'([,!?])',r'\1',s)
    
    # Remove any character that is not alphanumeric
    # + means one or more
    s = re.sub(r'[^a-zA-Z.!?]',r' ',s)

    # Remove a sequence of whitespace characters
    s = re.sub(r'\s+',r' ',s).strip()
    return s

# Returns True if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p, MAX_LENGTH=10):
    return (len(p[0].split()) < MAX_LENGTH) and (len(p[1].split()) < MAX_LENGTH)

# Filter pairs using filterPair condition (above)
def filterPairs(pairs, MAX_LENGTH = 10):
    return [pair for pair in pairs if filterPair(pair,MAX_LENGTH)]

# convert data to be fed into RNN sequences
def zeroPadding(l, fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# create index list from sentence
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# convert tensor to binary
def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sentence tensor with a tensor of lengths for each of the sequences in the batch
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Return padded target sequence tensor, padding mask and marx targt length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Return all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    # Sort the questions in descending length
    pair_batch.sort(key=lambda  x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

