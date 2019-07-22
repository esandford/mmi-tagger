from __future__ import division, print_function
import os
import random
import torch

from collections import Counter
from torch.nn.utils.rnn import pad_sequence


class Data(object):

    def __init__(self, data_path):
        self.data_path = data_path #example.words
        self.PAD = '<pad>'
        self.UNK = '<unk>'

        self.sents = []   # Index sequences    # [[2, 3, 4, 2, 5], [2, 5, 4, 2, 6], [2, 6, 4, 2, 3]]
        self.golds = []                        # [['D', 'N', 'V', 'D', 'N'], ['D', 'N', 'V', 'D', 'N'], ['D', 'N', 'V', 'D', 'N']]
        self.w2i = {self.PAD: 0, self.UNK: 1}  # {'chased': 4, 'dog': 3, 'cat': 5, '<pad>': 0, 'the': 2, 'rat': 6, '<unk>': 1}
        self.i2w = [self.PAD, self.UNK]        # ['<pad>', '<unk>', 'the', 'dog', 'chased', 'cat', 'rat']
        self.c2i = {self.PAD: 0, self.UNK: 1}  # {'a': 9, 'c': 8, 'e': 4, 'd': 5, 'g': 7, 'h': 3, 's': 10, 'o': 6, '<pad>': 0, 'r': 11, 't': 2, '<unk>': 1}
        self.i2c = [self.PAD, self.UNK]        # ['<pad>', '<unk>', 't', 'h', 'e', 'd', 'o', 'g', 'c', 'a', 's', 'r']
        self.word_counter = []
        self.char_counter = []
        self.label_counter = Counter()

        self.get_data()

    def get_data(self):
        wcount = Counter()
        ccount = Counter()
        def add(w):
            wcount[w] += 1
            if w not in self.w2i:
                self.i2w.append(w)
                self.w2i[w] = len(self.i2w) - 1
            for c in w:
                ccount[c] += 1
                if c not in self.c2i:
                    self.i2c.append(c)
                    self.c2i[c] = len(self.i2c) - 1
            return self.w2i[w]

        with open(self.data_path, 'r') as data_file:
            for line in data_file:
                toks = line.split()
                if toks:
                    self.sents.append([add(tok) for tok in toks])

        self.word_counter = [wcount[self.i2w[i]] for i in range(len(self.i2w))]
        self.char_counter = [ccount[self.i2c[i]] for i in range(len(self.i2c))]

        gold_path = self.data_path[:-5] + 'tags'
        assert os.path.isfile(gold_path)
        self.get_golds(gold_path)

    def get_golds(self, gold_path):
        with open(gold_path, 'r') as f:
            index = 0
            for line in f:
                labels = line.split()
                if labels:
                    self.label_counter.update(labels)
                    self.golds.append(labels)
                    assert len(self.golds[index]) == len(self.sents[index])
                    index += 1
        assert len(self.golds) == len(self.sents)

    def get_batches(self, batch_size):
        pairs = []
        # i = index specifying sentence = 0, 1, 2
        # j = index specifying word in sentence = 0, 1, 2, 3, 4
        for i in range(len(self.sents)):
            pairs.extend([(i, j) for j in range(len(self.sents[i]))]) # pairs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
        random.shuffle(pairs)
        batches = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:min(i + batch_size, len(pairs))] # batch = [(1, 1), (0, 4), (2, 4), (2, 1), (2, 2), (0, 1), (2, 3), (0, 0), (1, 2), (1, 4), (1, 0), (0, 3), (1, 3), (0, 2), (2, 0)]
            sorted_batch = sorted(batch,
                                  key=lambda x:
                                  len(self.i2w[self.sents[x[0]][x[1]]]),
                                  reverse=True)
            batches.append(sorted_batch)                     # sorted_batch = [(2, 2), (1, 2), (0, 2), (1, 1), (0, 4), (2, 4), (2, 1), (0, 1), (2, 3), (0, 0), (1, 4), (1, 0), (0, 3), (1, 3), (2, 0)]
        return batches #only one batch for len(pairs) = 15, batch_size=80

    def tensorize_batch(self, batch, device, width):
        def get_context(i, j, width):
            left = [0 for _ in range(width - j)] + \
                   self.sents[i][max(0, j - width):j]
            right = [0 for _ in range((j + width) - len(self.sents[i]) + 1)] + \
                    self.sents[i][j + 1: min(len(self.sents[i]), j + width) + 1]
            # if e.g. i==0, j==2, width=2 (referring to sentence "the dog chased the cat", word "chased"): 
            #    left = [2,3] (signifying "the dog"); right = [2,5] (signifying "the cat")
            # if e.g. i==0, j==0, width=2 (referring to sentence "the dog chased the cat", word "the" (first time)):
            #    left = [0,0] (signifying "<pad> <pad>"); right = [3,4] (signifying "dog chased")
            # if e.g. i==0, j==4, width=2 (referring to sentence "the dog chased the cat", word "cat"):
            #    left = [4,2] (signifying "chased the"); right = [0,0] (signifying "<pad> <pad>")
            return left + right

        contexts = [get_context(i, j, width) for (i, j) in batch] # list of [left,right]s to make up X
        targets = [self.sents[i][j] for (i, j) in batch]          # list of individual words to make up Y
        seqs = [torch.LongTensor([self.c2i[c] for c in self.i2w[target]]) # characters in Ys
                for target in targets]

        X = torch.LongTensor(contexts).to(device)  # B x 2width, where B = batch size = 15 for basic example
        Y1 = torch.LongTensor(targets).to(device)  # B
        Y2 = pad_sequence(seqs, padding_value=0).to(device)  # T x B, where T = maximum number of characters in any word in Y = 6 in basic example (for "chased")
        lengths = torch.LongTensor([seq.shape[0] for seq in seqs]).to(device) #B, number of characters in each word in batch
        return X, Y1, Y2, lengths
