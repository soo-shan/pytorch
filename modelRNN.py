import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size' bcaz our input size is word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # input_seq: batch of input sentences; shape = (max_length,batch size)
        # input_lengths: list of sentence lenghts corresponding to each sentence in the batch
        # shape of hidden state: (n_layers * num_directions, batch_size, hidden_size)
        embedded = self.embedding(input_seq)
        # pack padded batch of sequences for RNN module 
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # unpack  padding
        
