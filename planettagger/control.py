from __future__ import division, print_function
import data
import math
import random
import time
import torch
import torch.nn as nn
import numpy as np
import pickle

from datetime import timedelta

def calculateEntropy(probs):
    x = np.multiply(probs,np.log(probs))
    entro = -1. * np.sum(x)
    return entro

def calculateLoss(targetTruths,data_path):
    """
    Calculate the loss function if everything were labeled correctly, i.e. the "goal"
    loss function.

    #contextTruths = length Batchsize list, each entry of which is a list of length 2*width
    #                each sublist has entries of values between 0....(nClasses - 1)
    #                corresponding to the true class membership of that context planet.
    
    targetTruths = length Batchsize list, each entry of which is a value between 0...(nclasses-1)
                   corresponding tot he true class membership of that target planet.
    """
    #evaluate loss function for truths
    B = int(len(targetTruths))
    nClasses = int(np.max(np.array(targetTruths)) + 1)

    #make an array of class probabilities for the target planets
    #e.g. if there are classes 0,1,2 and planet i is of class 0, the ith 
    # row in the array should be [1 0 0]
    pZ_Y = np.zeros((B,nClasses))

    for i in range(B):
        trueClass_i = int(targetTruths[i][0])
        pZ_Y[i][trueClass_i] = 1

    pZ = np.mean(pZ_Y,axis=0)
    hZ = calculateEntropy(pZ)

    #if everything is working perfectly, the class labels
    # predicted by the contexts should be the same as those
    # predicted by the target planets themselves. 

    hZ_X_ub = -1.0*np.mean(np.sum(pZ_Y,axis=1))
    loss = hZ_X_ub - hZ

    return loss

class Control(nn.Module):

    def __init__(self, model, model_path, batch_size, device, logger, truth_known):
        super(Control, self).__init__()
        self.model = model
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.logger = logger
        self.truth_known = truth_known

    def train(self, data, data_path, lr, epochs):
        self.log_data(data)
        self.logger.log('[TRAINING]')
        self.logger.log('   num_labels:    %d' % self.model.num_labels)
        self.logger.log('   batch_size:    %d' % self.batch_size)
        self.logger.log('   lr:            %g' % lr)
        self.logger.log('   epochs:        %d' % epochs)
        self.logger.log('')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        start_time = time.time()

        epochLog = []
        lossLog = []

        try:
            for epoch in range(1, epochs + 1):
                avg_loss, epoch_time = self.do_epoch(data, optimizer)
                #print("avg_loss is {0}".format(avg_loss))
                #print("epoch_time is {0}".format(epoch_time))
                bits = (- avg_loss) * math.log(math.e,2)
                self.logger.log('| epoch {:3d} | loss {:6.12f} | {:6.12f} bits | '
                                'time {:10s}'.format(
                                    epoch, avg_loss, bits,
                                    str(timedelta(seconds=int(epoch_time)))))

                epochLog.append(epoch)
                lossLog.append(avg_loss)
                with open(self.model_path, 'wb') as f:
                    state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                     'optimizer': optimizer.state_dict()}
                    torch.save(state, f)
                    #torch.save(self.model, f)

        except KeyboardInterrupt:
            self.logger.log('-' * 89)
            self.logger.log('Exiting from training early')

        self.logger.log('\nTraining time {:10s}'.format(
            str(timedelta(seconds=int(time.time() - start_time)))))

        #self.load_model(self.lr)
        self.logger.log('=' * 89)
        self.logger.log('=' * 89)

        epochLosses = np.vstack((np.array(epochLog).T,np.array(lossLog).T)).T
        np.save("./{0}_losses.npy".format(data_path[:-4]),epochLosses)

        return 

    def do_epoch(self, data, optimizer):
        self.model.train()
        #print(self.model)
        avg_loss = 0
        epoch_start_time = time.time()
        batches = data.get_batches(self.batch_size)
        #print(batches)
        #print(type(batches)) #list, each element of which is 1 batch 
        for batch in batches:
            self.model.zero_grad()
            X, Y1, contextTruths, targetTruths = data.tensorize_batch(batch, self.device, self.model.width, self.truth_known)
            #print("X shape is {0}".format(X.shape))    # Batchsize x 2width x numPlanetFeatures
            #print("Y1 shape is {0}".format(Y1.shape))  # Batchsize x numPlanetFeatures
            loss = self.model(X, Y1, is_training=True) # runs MMIModel.forward(X, Y1, is_training=True)
            #print("loss is {0}".format(loss))
            avg_loss += loss.item() / len(batches)
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start_time
        
        return avg_loss, epoch_time

    def classify(self, data_path, data, num_labels):
        self.model.eval()
        batches = data.get_batches(self.batch_size)
        zseqs = [[False for w in sys] for sys in data.systems]
        clustering = [{} for z in range(self.model.num_labels)]

        all_future_probs = np.zeros((1,self.model.num_labels))
        all_idxs = np.zeros((1,1))
        avg_truth_loss = 0.
        with torch.no_grad():
            for batch in batches:
                X, Y1, contextTruths, targetTruths = data.tensorize_batch(batch, self.device, self.model.width, self.truth_known)
                #print(type(Y1))
                #print(Y1.shape)
                #print(type(data.targetIdxs))
                #print(len(data.targetIdxs))
                if self.truth_known:
                    truth_loss = calculateLoss(targetTruths,data_path)
                    #print("loss is {0}".format(loss))
                    avg_truth_loss += truth_loss / len(batches)


                future_probs, future_max_probs, future_indices = self.model(X, Y1, is_training=False)
                all_future_probs = np.vstack((all_future_probs,future_probs.numpy()))
                all_idxs = np.vstack((all_idxs,np.atleast_2d(np.array(data.targetIdxs)).T))
                for k, (i, j) in enumerate(batch):
                    z = future_indices[k].max()
                    zseqs[i][j] = z
                    clustering[z][data.systems[i][j]] = True

        all_future_probs = all_future_probs[1:]
        all_idxs = all_idxs[1:].astype(int)
        all_idxs = all_idxs[:,0] - 1 # 1-indexing to 0-indexing
        #all_future_probs = all_future_probs[all_idxs]

        np.save("./{0}_classprobs.npy".format(data_path[:-4]),all_future_probs)
        np.save("./{0}_idxs.npy".format(data_path[:-4]),all_idxs)
        np.save("./{0}_optimalLoss.npy".format(data_path[:-4]),avg_truth_loss)
        
        maxMI = (- avg_truth_loss) * math.log(math.e,2)
        print("avg_truth_loss is {0}; max MI is {1}".format(avg_truth_loss,maxMI))
        
        return future_probs, zseqs, clustering


    def load_model(self,lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        with open(self.model_path, 'rb') as f:
            #self.model = torch.load(f)
            checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(f, checkpoint['epoch']))

    def log_data(self, data):
        self.logger.log('-' * 89)
        self.logger.log('[DATA]')
        self.logger.log('   data:          %s' % data.data_path)
        self.logger.log('   # planets:     %d' % sum(len(sys) for sys in data.systems))
        self.logger.log('-' * 89)
