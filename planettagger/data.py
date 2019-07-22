from __future__ import division, print_function
import numpy as np
import random
import torch
import pickle

from collections import Counter

class Data(object):

    def __init__(self, num_planet_features, data_path):
        self.data_path = data_path
        self.PAD = np.zeros(num_planet_features)
        self.PAD = tuple(self.PAD)
        self.systems = []   # will represent each training set planetary system as a list of indices
                            # e.g. [1,2,3,4,5],[6,7],[8,9] (each planet is unique, so we don't expect repeats)
        
        # "planet" defined by a list: [Teff, logg, [Fe/H], Rp/R*, P]
        self.planet2i = {self.PAD: 0} #will contain each "planet" in the training set, uniquely
        self.i2planet = [self.PAD]

        self.label_counter = Counter()

        self.get_data()

    def get_data(self):
        #wcount = Counter()
        #ccount = Counter()
        def add(p):
            # "p" is a list: [Teff, logg, [Fe/H], Rp/R*, P]
            if tuple(p) not in self.planet2i:
                self.i2planet.append(p)
                self.planet2i[tuple(p)] = len(self.i2planet) - 1
            return self.planet2i[tuple(p)]
        
        with open(self.data_path, "rb") as picklefile:
            trainingDat = pickle.load(picklefile)
            for sys in trainingDat:
                self.systems.append([add(planet) for planet in sys])

    def get_batches(self, batch_size):
        pairs = []
        # i = index specifying system = 0, 1, 2, ..., 2656
        # j = index specifying planet in system = 0, 1, 2, ..., 3513 (except amend these numbers to exclude single-planet systems)
        for i in range(len(self.systems)):
            pairs.extend([(i, j) for j in range(len(self.systems[i]))]) # pairs = [(0,0),(0,1),...,(0,3513),...,(2656,0),...,(2656,3513)]
        random.shuffle(pairs)
        batches = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:min(i + batch_size, len(pairs))] 
            sorted_batch = sorted(batch,
                                  key=lambda x:
                                  len(self.i2planet[self.systems[x[0]][x[1]]]),
                                  reverse=True)
            batches.append(sorted_batch)
        return batches

    def tensorize_batch(self, batch, device, width):
        def get_context(i, j, width):
            left = [0 for _ in range(width - j)] + \
                   self.systems[i][max(0, j - width):j]
            right = [0 for _ in range((j + width) - len(self.systems[i]) + 1)] + \
                    self.systems[i][j + 1: min(len(self.systems[i]), j + width) + 1]

            left = [self.i2planet[k] for k in left]
            right = [self.i2planet[k] for k in right]

            # below for POS tagging example, but analogous for planets, except that each "planet" 
            #   is an array of data rather than an index to a word:
            # if e.g. i==0, j==2, width=2 (referring to sentence "the dog chased the cat", word "chased"): 
            #    left = [2,3] (signifying "the dog"); right = [2,5] (signifying "the cat")
            # if e.g. i==0, j==0, width=2 (referring to sentence "the dog chased the cat", word "the" (first time)):
            #    left = [0,0] (signifying "<pad> <pad>"); right = [3,4] (signifying "dog chased")
            # if e.g. i==0, j==4, width=2 (referring to sentence "the dog chased the cat", word "cat"):
            #    left = [4,2] (signifying "chased the"); right = [0,0] (signifying "<pad> <pad>")
            return left + right

        contexts = [get_context(i, j, width) for (i, j) in batch] # list of [left,right]s to make up X
        targets = [self.systems[i][j] for (i, j) in batch]        # list of individual words to make up Y
        targets = [self.i2planet[k] for k in targets]

        X = torch.LongTensor(contexts).to(device)  # B x 2width x num_planet_features, where B = batch size = 15 for basic example
        Y1 = torch.LongTensor(targets).to(device)  # B x num_planet_features
        return X, Y1
