import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.1):
        if not self.training or dropout == 0:
            return x
        else:
            mask = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout).requires_grad_(False).expand_as(x) / (1 - dropout)
        return mask * x

class LockedDropoutCell(nn.Module):
    def __init__(self):
        super(LockedDropoutCell, self).__init__()
        self.mask = None
        self.has_mask = False
    
    def reset_mask(self):
        self.mask = None
        self.has_mask = False

    def forward(self, x, dropout=0.1):
        if not self.training or dropout == 0:
            return x
        if not self.has_mask:
            # not mask set
            self.has_mask = True
            self.mask = x.new(1, x.size(1)).bernoulli_(1 - dropout).requires_grad_(False).expand_as(x) / (1 - dropout)
    
        return self.mask * x

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        arguments:
        query: N * query_size
        key: N * T * key_size
        value: N * T * value_size
        lens: N
        
        N - batch_size
        T - max length
        
        
        return:
        output: Attended Context
        attention_mask: Attention mask that can be plotted
        '''
        q = query
        k = key
        v = value
        
        # k: (N, T, key_size), q: (N, context_size), energy: (N, T)
        energy = torch.bmm(k, q.unsqueeze(2)).squeeze(2)
        
        # LHS: (1, T), RHS: (N, length), mask: (N, T)
        mask = torch.arange(key.size(1)).to(DEVICE).unsqueeze(0) >= lens.unsqueeze(1)
        
        # Set attention logits at padding positions to negative infinity.
        energy.masked_fill_(mask, -1e9)
        
        # energy: (N, T)
        attention = F.softmax(energy, dim=1)
        
        # attention: (N, T), v: (N, T, value_size)
        context = torch.bmm(attention.unsqueeze(1), v).squeeze(1)
        
        # context: (N, value_size), mask: (N, T)
        return context, mask

class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(pBLSTM, self).__init__()
        self.dropout = dropout
        self.lockeddropout = LockedDropout()
        self.blstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.blstm2 = nn.LSTM(input_size=hidden_dim*4, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.blstm3 = nn.LSTM(input_size=input_dim*4, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.blstm4 = nn.LSTM(input_size=input_dim*4, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

    def _reduce_by_2(self, out_padded, lengths):
        batch_size, max_length, dim = out_padded.shape
        # pad with zeros if length not dividable by 2
        if max_length % 2 != 0:
            out_padded = F.pad(out_padded, (0,0,0,1))
            max_length += 1
        out_reshaped = out_padded.view(batch_size, int(max_length/2), dim*2)
        out_padded = rnn.pack_padded_sequence(out_reshaped, lengths=(lengths+1)/2, batch_first=True, enforce_sorted=False)
        
        return out_padded, (lengths+1)/2
    
    def forward(self, x, lens):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''
        out1, _ = self.blstm1(x)
        out1_padded, out1_padded_length = rnn.pad_packed_sequence(out1, batch_first=True)
        out1_padded = self.lockeddropout(out1_padded, self.dropout)
        out1_reduced, lengths_2 = self._reduce_by_2(out1_padded, out1_padded_length)
       
        out2, _ = self.blstm2(out1_reduced)
        out2_padded, out2_padded_length = rnn.pad_packed_sequence(out2, batch_first=True)
        out2_padded = self.lockeddropout(out2_padded, self.dropout)
        out2_reduced, lengths_3 = self._reduce_by_2(out2_padded, out2_padded_length)

        out3, _ = self.blstm2(out2_reduced)
        out3_padded, out3_padded_length = rnn.pad_packed_sequence(out3, batch_first=True)
        out3_padded = self.lockeddropout(out3_padded, self.dropout)
        out3_reduced, lengths_4 = self._reduce_by_2(out3_padded, out3_padded_length)
        
        out4, _ = self.blstm2(out3_reduced)
        
        # out3: (N, T/(2^3), H)
        return out4, lengths_4
        

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128, key_size=128, dropout=0.1):
        super(Encoder, self).__init__()
        self.plstm = pBLSTM(40, hidden_dim, dropout=dropout)
        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens):
        rnn_inp = rnn.pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)

        outputs, lengths = self.plstm(rnn_inp, lens)

        linear_input, _ = rnn.pad_packed_sequence(outputs, batch_first=True)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, lens

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, sos_index=33, eos_index=34, dropout=0.1):
        super(Decoder, self).__init__()
        self.lockeddropout_cell_1 = LockedDropoutCell()
        self.lockeddropout_cell_2 = LockedDropoutCell()
        self.dropout = dropout
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.hidden_dim = hidden_dim
        self.value_size = value_size
        self.key_size = key_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, lengths, text=None, isTrain=True, isAttended=True, random=False):
        '''
        :param key :(N, T, key_size) Output of the Encoder Key projection layer
        :param values: (N, T, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        batch_size = key.shape[0]
        
        probs = torch.ones(batch_size, dtype=torch.float64).to(DEVICE)

        if (isTrain == True):
            max_len =  text.shape[1] - 1 # note that y is <sos> ... <eos>, but maxlen should be len(... <eos>)
            embeddings = self.embedding(text) # (N, text_len, embedding_size)
        else:
            max_len = 250

        # initialization
        predictions = []
        hidden_states = [[torch.zeros(batch_size, self.hidden_dim).to(DEVICE), torch.zeros(batch_size, self.hidden_dim).to(DEVICE)], 
                         [torch.zeros(batch_size, self.key_size).to(DEVICE), torch.zeros(batch_size, self.hidden_dim).to(DEVICE)]]
        
        prediction = torch.zeros(batch_size,1).to(DEVICE) + self.sos_index # first input is <sos>
        
        # reset lockedDropout mask
        self.lockeddropout_cell_1.reset_mask()
        self.lockeddropout_cell_2.reset_mask()
        
        if isAttended:
            attention_context = self.attention(hidden_states[1][0], key, values, lengths)[0]
        else:
            attention_context = torch.zeros(batch_size, self.value_size).to(DEVICE) # random value when no attention
            
            
        for i in range(max_len):
            if isTrain:
                # teacher forcing
                if np.random.random_sample() <= 0.2:
                    # auto-regressive
                    char_embed = self.embedding(prediction.argmax(-1)) 
                else:
                    # use true label
                    char_embed = embeddings[:, i, :]
            else:
                # auto-regressive
                if random == False:
                    char_embed = self.embedding(prediction.argmax(-1))
                    probs = probs * prediction.softmax(dim=1).max(-1)[0]
                else:
                    pred_probs = prediction.softmax(dim=1)
                    m = Categorical(probs=pred_probs)
                    sample_pred = m.sample()
                    char_embed = self.embedding(sample_pred)
                    probs = probs * pred_probs.gather(1, sample_pred.unsqueeze(1)).squeeze(1)
            
            inp = torch.cat([char_embed, attention_context], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            inp_2 = self.lockeddropout_cell_1(inp_2, dropout=self.dropout)
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
            if isAttended:
                attention_context = self.attention(hidden_states[1][0], key, values, lengths)[0]
            else:
                attention_context = torch.zeros(batch_size, self.value_size).to(DEVICE) # random value when no attention
                
            output = hidden_states[1][0]
            output = self.lockeddropout_cell_2(output, dropout=self.dropout)

            # use output and attention_context to predict
            prediction = self.character_prob(torch.cat([output, attention_context], dim=1)) # (N, vocab_size)
            predictions.append(prediction.unsqueeze(1)) # (N, 1, vocab_size)
                
        return torch.cat(predictions, dim=1), probs # (N, T, vocab_size), (N,)

class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=True, sos_index=33, eos_index=34, dropout=0.1):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, value_size=value_size, key_size=key_size, dropout=dropout)
        self.decoder = Decoder(vocab_size, hidden_dim, value_size=value_size, key_size=key_size, sos_index=sos_index, eos_index=eos_index, dropout=dropout)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True, isAttended=True, random=False):
        key, value, lengths = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value, lengths, text=text_input, isTrain=True, isAttended=isAttended, random=random)
        else:
            predictions = self.decoder(key, value, lengths, text=None, isTrain=False, isAttended=isAttended, random=random)
        return predictions
    
        