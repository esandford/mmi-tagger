from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class MMIModel(nn.Module):

    def __init__(self, num_word_types, num_char_types, num_labels, word_dim,
                 char_dim, width, num_lstm_layers):
        super(MMIModel, self).__init__() #initializes a nn.Module, which is the parent class of MMIModel, so we can use all the associated methods
        #print("num_word_types is {0}".format(num_word_types))      #7
        #print("num_char_types is {0}".format(num_char_types))      #12
        #print("num_labels is {0}".format(num_labels))              #3
        #print("word_dim is {0}".format(word_dim))                  #200
        #print("char_dim is {0}".format(char_dim))                  #100
        #print("width is {0}".format(width))                        #2
        #print("num_lstm_layers is {0}".format(num_lstm_layers))    #1
        self.wemb = nn.Embedding(num_word_types, word_dim, padding_idx=0)
        self.cemb = nn.Embedding(num_char_types, char_dim, padding_idx=0)
        self.num_labels = num_labels
        self.width = width

        self.loss = Loss()

        self.past = PastEncoder(self.wemb, width, num_labels)
        self.future = FutureEncoder(self.wemb, self.cemb, num_lstm_layers,
                                    num_labels)

    def forward(self, past_words, future_words, padded_chars, char_lengths,
                is_training=True):
        past_rep = self.past(past_words)
        future_rep = self.future(future_words, padded_chars, char_lengths)

        if is_training:
            loss = self.loss(past_rep, future_rep)
            return loss

        else:
            future_probs, future_indices = future_rep.max(1)
            return future_probs, future_indices


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.entropy = Entropy()

    def forward(self, past_rep, future_rep):
        pZ_Y = F.softmax(future_rep, dim=1)
        pZ = pZ_Y.mean(0)
        hZ = self.entropy(pZ)

        x = pZ_Y * F.log_softmax(past_rep, dim=1)  # B x m
        hZ_X_ub = -1.0 * x.sum(dim=1).mean()

        loss = hZ_X_ub - hZ
        return loss


class Entropy(nn.Module):

    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, probs):
        x = probs * torch.log(probs)
        entropy = -1.0 * x.sum()
        return entropy


class PastEncoder(nn.Module):

    def __init__(self, wemb, width, num_labels):
        # In the __init__() step, define what each layer is. In the forward() step,
        # define how the layers are connected.
        super(PastEncoder, self).__init__()
        self.wemb = wemb
        self.linear = nn.Linear(2 * width * wemb.embedding_dim, num_labels) # 800 x 3 = (2width x d_w) x num_labels

    def forward(self, words):
        wembs = self.wemb(words)                           # B x 2width x d_w
        #print(wembs)
        #print(wembs.shape)
        #print(words.shape)                                # 15 x 4 = B x 2width
        #print(wembs.shape)                                # 15 x 4 x 200 = B x 2width x d_w
        #print(words.shape[0])                             # 15
        #print(wembs.view(words.shape[0], -1).shape)       # 15 x 800 = B x (2width x d_w)
        rep = self.linear(wembs.view(words.shape[0], -1))  # 15 x 3 = B x num_labels
        #print(rep)
        print("PastEncoder shape is {0}".format(rep.shape))
        return rep


class FutureEncoder(nn.Module):

    def __init__(self, wemb, cemb, num_layers, num_labels):
        super(FutureEncoder, self).__init__()
        self.wemb = wemb
        self.cemb = cemb
        self.lstm = nn.LSTM(cemb.embedding_dim, cemb.embedding_dim, num_layers,
                            bidirectional=True)
        self.linear = nn.Linear(wemb.embedding_dim + 2 * cemb.embedding_dim,
                                num_labels)

    def forward(self, words, padded_chars, char_lengths):
        B = len(char_lengths)
        wembs = self.wemb(words)  # B x d_w

        packed = pack_padded_sequence(self.cemb(padded_chars), char_lengths)
        output, (final_h, final_c) = self.lstm(packed)

        final_h = final_h.view(self.lstm.num_layers, 2, B,
                               self.lstm.hidden_size)[-1]         # 2 x B x d_c
        cembs = final_h.transpose(0, 1).contiguous().view(B, -1)  # B x 2d_c

        rep = self.linear(torch.cat([wembs, cembs], 1))  # B x m
        print("FutureEncoder shape is {0}".format(rep.shape))
        return rep
