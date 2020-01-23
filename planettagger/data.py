import numpy as np
import random
import torch
import pickle

from collections import Counter

class Data(object):
    """
    An object which contains data to feed into the network.

    """
    def __init__(self, num_planet_features, num_stellar_features, data_path, truth_known):
        """
        Initialize the Data object.

        Parameters
        ---------
        num_planet_features : int
            The number of training features relating to planets (e.g. Rp, P)
        num_stellar_features : int
            The number of training features relating to the host star (e.g. Teff, logg, Fe/H)
        data_path : str
            The path to the data file
        truth_known : bool
            whether the true class of the planets in the training set is known 
            (as it would be for simulated data) or not (as it would be for real data)
        """

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
        """
        Get the data from the file and reformat it appropriately.
        
        Parameters
        ---------
        truth_known : bool
            whether the true class of the planets in the training set is known 
            (as it would be for simulated data) or not (as it would be for real data)
        """
        def add(p):
            """
            Add an individual planet from the data file to the Data object.

            Parameters
            ---------
            p : list
                List of planet properties, e.g. [Rp, P, Teff, logg, [Fe/H]]

            Returns
            ---------
            len(self.i2planet) - 1 : int
                number of unique planets now added to self.i2planet, including this one
            """

            # if this planet hasn't been handled already
            if tuple(p) not in self.planet2i:
                self.i2planet.append(p)
                self.planet2i[tuple(p)] = len(self.i2planet) - 1
            return self.planet2i[tuple(p)]
        
        # loop over planets in data file
        with open(self.data_path, "rb") as picklefile:
            trainingDat = pickle.load(picklefile)
            for sys in trainingDat:
                self.systems.append([add(planet) for planet in sys])

        # if truth is known, format the true classes appropriately
        if truth_known:
            truthPath = "./{0}_truthsOrganized.txt".format(self.data_path[:-4])
            with open(truthPath, "rb") as truthfile:
                truthsDat = pickle.load(truthfile)
                for sys in truthsDat:
                    self.truths.append([truth for truth in sys])

        return

    def get_batches(self, batch_size):
        """
        Organize the data into batches.

        Parameters
        ---------
        batch_size : int
            The number of planets per batch.

        Returns
        ---------
        batches : list
            A list of batches.
            Each batch is itself a list, where each entry is a tuple of form
            (planetary_system_index, planet_index)
        """

        pairs = []

        # i = index specifying system = 0, 1, 2, ..., 2656
        # j = index specifying planet in system = 0, 1, 2, 3, 4, 5 (up to 6-planet systems)
        # len(self.systems) is the number of planetary systems in the data file, e.g. 2656
        # self.systems is a nested list, each entry of which is 1 planetary system
        # self.systems is of form e.g. [[1],[2],[3],[4],[5,6],[7,8],...,[3512,3513],[3514]]
        
        # index into all planetary systems and planets
        for i in range(len(self.systems)):
            pairs.extend([(i, j) for j in range(len(self.systems[i]))]) 
        # e.g. if self.systems is [[1,2], [3], [4,5,6]],
        # pairs is of form [(0,0),(0,1), (1,0), (2,0),(2,1),(2,2)]
        # so each entry in pairs is a tuple of form (planetary_system_index, planet_index)

        # shuffle pairs randomly
        random.shuffle(pairs)

        # organize pairs into batches
        batches = []
        for i in range(0, len(pairs), batch_size):
            # grab batch_size planets
            batch = pairs[i:min(i + batch_size, len(pairs))] 

            # sort them 
            sorted_batch = sorted(batch,
                                  key=lambda x:
                                  len(self.i2planet[self.systems[x[0]][x[1]]]),
                                  reverse=True)
            batches.append(sorted_batch)

        return batches

    def tensorize_batch(self, batch, device, width, truth_known):
        """
        Convert batch of planets into torch tensor format.

        Parameters
        ---------
        batch : list
            A list, where each entry is tuple of form (planetary_system_index, planet_index)
        device : obj
            A torch.device ('cpu' or 'cuda')
        width : int
            The width of context to consider on each side of the target planet
        truth_known : bool
            whether the true class of the planets in the training set is known 
            (as it would be for simulated data) or not (as it would be for real data)

        Returns
        ---------
        X : torch.Tensor
            A torch.Tensor containing the context data,
            of shape [Batchsize, (2width * num_planet_features) + num_stellar_features)]
        Y : torch.Tensor
            A torch.Tensor containing the target data,
            of shape [Batchsize, num_planet_features]
        targetTruths: list
            A list of length Batchsize. The ith entry is an integer between 0
            and nClasses-1 which indicates the true class of target planet i in the batch
        """

        def get_context(i, j, width):
            """
            Get the context surrounding a particular planet.

            Parameters
            ---------
            i : int
                An index into planetary systems
            j : int
                An index into planets within a particular planetary system
            width : int
                The width of context to consider on each side of the target planet

            Returns
            ---------
            thisContext : list
                A list of [*stellar_props, *context_planet_-w_props, ..., *context_planet_-1_props,
                *context_planet_+1_props, ..., *context_planet_+w_props]
            """

            thisPlanet = self.systems[i][j]
            thisPlanet = self.i2planet[thisPlanet]
            
            # get stellar properties by themselves in a list
            thisContext = list(thisPlanet[self.num_planet_features:])
            
            # get indices into left and right contexts
            left = [0 for _ in range(width - j)] + \
                   self.systems[i][max(0, j - width):j]
            right = self.systems[i][j + 1: min(len(self.systems[i]), j + width) + 1] + \
                    [0 for _ in range((j + width) - len(self.systems[i]) + 1)]

            # use these indices to get appropriate arrays of planet properties
            left = [self.i2planet[k][0:self.num_planet_features] for k in left]
            right = [self.i2planet[k][0:self.num_planet_features] for k in right]
            
            # append these planet properties to thisContext, in the right order
            for ii in range(len(left)):
                for jj in range(len(left[ii])):
                    thisContext.append(left[ii][jj])

            for ii in range(len(right)):
                for jj in range(len(right[ii])):
                    thisContext.append(right[ii][jj])
            
            # below for POS tagging example, but analogous for planets, except that each "planet" 
            #   is an array of data rather than an index to a word:
            # if e.g. i==0, j==2, width=2 (referring to sentence "the dog chased the cat", word "chased"): 
            #    left = [2,3] (signifying "the dog"); right = [2,5] (signifying "the cat")
            # if e.g. i==0, j==0, width=2 (referring to sentence "the dog chased the cat", word "the" (first time)):
            #    left = [0,0] (signifying "<pad> <pad>"); right = [3,4] (signifying "dog chased")
            # if e.g. i==0, j==4, width=2 (referring to sentence "the dog chased the cat", word "cat"):
            #    left = [4,2] (signifying "chased the"); right = [0,0] (signifying "<pad> <pad>")

            return thisContext


        # get a list of targets to make up Y
        targetIdxs = [self.systems[i][j] for (i, j) in batch]
        # keep track of how we've reshuffled these into our batch
        self.targetIdxs = targetIdxs
        
        # keep only planet features, not stellar features, for Y
        targets = [self.i2planet[k][0:self.num_planet_features] for k in targetIdxs]

        # get true classes of both targets and context planets, if known
        if truth_known:
            targetTruths = [self.truths[i][j] for (i, j) in batch]
            contexts = [get_context(i, j, width) for (i, j) in batch]
            
        else:
            targetTruths = None
            # list of [left,right]s to make up X
            contexts = [get_context(i, j, width) for (i, j) in batch] 

        # Convert lists to torch tensors.
        # X is of shape batch_size x (num_stellar_features + (2width x num_planet_features))
        X = torch.FloatTensor(contexts).to(device)
        # Y is of shape batch_size x num_planet_features
        Y = torch.FloatTensor(targets).to(device)  

        return X, Y, targetTruths
