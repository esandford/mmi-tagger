from __future__ import division, print_function
import numpy as np
import random
import torch
import pickle

from collections import Counter

class Data(object):

    def __init__(self, num_planet_features, num_stellar_features, data_path, truth_known):
        self.num_features = num_planet_features + num_stellar_features
        self.num_planet_features = num_planet_features
        self.num_stellar_features = num_stellar_features
        self.data_path = data_path
        self.PAD = np.zeros(self.num_features)
        self.PAD = tuple(self.PAD)
        self.systems = []   # will represent each training set planetary system as a list of indices
                            # e.g. [1,2,3,4,5],[6,7],[8,9] (each planet is unique, so we don't expect repeats)
        
        # "planet" defined by a list: [Teff, logg, [Fe/H], Rp/R*, P]
        self.planet2i = {self.PAD: 0} #will contain each "planet" in the training set, uniquely
        self.i2planet = [self.PAD]
        self.targetIdxs = []

        self.label_counter = Counter()

        self.truths = []
        
        self.get_data(truth_known)

    def get_data(self,truth_known):
        #wcount = Counter()
        #ccount = Counter()
        def add(p):
            # "p" is a list: [Rp/R*, P, Teff, logg, [Fe/H]]
            if tuple(p) not in self.planet2i:
                self.i2planet.append(p)
                self.planet2i[tuple(p)] = len(self.i2planet) - 1
            return self.planet2i[tuple(p)]
        
        with open(self.data_path, "rb") as picklefile:
            trainingDat = pickle.load(picklefile)
            for sys in trainingDat:
                self.systems.append([add(planet) for planet in sys])

        if truth_known:
            truthPath = "./{0}_truthsOrganized.txt".format(self.data_path[:-4])
            with open(truthPath, "rb") as truthfile:
                truthsDat = pickle.load(truthfile)
                for sys in truthsDat:
                    self.truths.append([truth for truth in sys])


    def get_batches(self, batch_size):
        pairs = []
        # i = index specifying system = 0, 1, 2, ..., 2656
        # j = index specifying planet in system = 0, 1, 2, 3, 4, 5 (up to 6-planet systems)
        #print(len(self.systems)) #2657
        #print(self.systems)      #[[1],[2],[3],[4],[5,6],[7,8],...,[3512,3513],[3514]]
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
        #print(len(batches))
        #print(len(batches[0]))
        #print(len(batches[1]))
        return batches

    def tensorize_batch(self, batch, device, width, truth_known):
        def get_context(i, j, width):
            # stop repeating stellar features!
            thisPlanet = self.systems[i][j]
            thisPlanet = self.i2planet[thisPlanet]
            
            thisStellar = list(thisPlanet[self.num_planet_features:])
            

            left = [0 for _ in range(width - j)] + \
                   self.systems[i][max(0, j - width):j]
            right = self.systems[i][j + 1: min(len(self.systems[i]), j + width) + 1] + \
                    [0 for _ in range((j + width) - len(self.systems[i]) + 1)]

            left = [self.i2planet[k][0:self.num_planet_features] for k in left]
            right = [self.i2planet[k][0:self.num_planet_features] for k in right]
            
            for ii in range(len(left)):
                for jj in range(len(left[ii])):
                    thisStellar.append(left[ii][jj])

            for ii in range(len(right)):
                for jj in range(len(right[ii])):
                    thisStellar.append(right[ii][jj])
            
            # below for POS tagging example, but analogous for planets, except that each "planet" 
            #   is an array of data rather than an index to a word:
            # if e.g. i==0, j==2, width=2 (referring to sentence "the dog chased the cat", word "chased"): 
            #    left = [2,3] (signifying "the dog"); right = [2,5] (signifying "the cat")
            # if e.g. i==0, j==0, width=2 (referring to sentence "the dog chased the cat", word "the" (first time)):
            #    left = [0,0] (signifying "<pad> <pad>"); right = [3,4] (signifying "dog chased")
            # if e.g. i==0, j==4, width=2 (referring to sentence "the dog chased the cat", word "cat"):
            #    left = [4,2] (signifying "chased the"); right = [0,0] (signifying "<pad> <pad>")
            return thisStellar


        targetIdxs = [self.systems[i][j] for (i, j) in batch]        # list of individual words to make up Y
        self.targetIdxs = targetIdxs
        
        #keep only planet features, not stellar features, for Y1
        targets = [self.i2planet[k][0:self.num_planet_features] for k in targetIdxs]

        if truth_known:
            targetTruths = [self.truths[i][j] for (i, j) in batch]
            contexts = [get_context(i, j, width) for (i, j) in batch]
            
        else:
            targetTruths = None
            contexts = [get_context(i, j, width) for (i, j) in batch] # list of [left,right]s to make up X
        
        X = torch.FloatTensor(contexts).to(device)  # B x (num_stellar_features + (2width x num_planet_features)), where B = batch size = 15 for basic example
        Y1 = torch.FloatTensor(targets).to(device)  # B x (num_planet_features + num_stellar_features)

        return X, Y1, targetTruths
