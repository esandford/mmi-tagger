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

    targetTruths = length Batchsize list, each entry of which is a value between 0...(nclasses-1)
                   corresponding tot he true class membership of that target planet.
    """
    #evaluate loss function for truths
    B = int(len(targetTruths))
    nClasses = int(np.max(np.array(targetTruths)) + 1)

    #make an array of class probabilities for the target planets
    #e.g. if there are classes 0,1,2 and planet i is of class 0, the ith 
    # row in the array should be [1 0 0]
    pZ_Y_ideal_softmax = np.zeros((B,nClasses))
    pZ_Y_ideal_logsoftmax = 0.00001*np.ones((B,nClasses))

    for i in range(B):
        trueClass_i = int(targetTruths[i][0])
        pZ_Y_ideal_softmax[i][trueClass_i] = 1.
        pZ_Y_ideal_logsoftmax[i][trueClass_i] = 1. - 2.*0.00001

    pZ_Y_ideal_logsoftmax = np.log(pZ_Y_ideal_logsoftmax)
        
    pZ = np.mean(pZ_Y_ideal_softmax,axis=0)
    hZ = calculateEntropy(pZ)

    #if everything is working perfectly, the class labels
    # predicted by the contexts should be the same as those
    # predicted by the target planets themselves. 
    # But remember to treat softmaxs and logsoftmaxs correctly!

    X = np.multiply(pZ_Y_ideal_softmax, pZ_Y_ideal_logsoftmax)
    hZ_X_ub = -1.0*np.mean(np.sum(X,axis=1))
    loss = hZ_X_ub - hZ

    return loss

class Control(nn.Module):

    def __init__(self, model, model_path, batch_size, device, logger, truth_known, seed):
        super(Control, self).__init__()
        self.model = model
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.logger = logger
        self.truth_known = truth_known
        self.seed = seed

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

            for epoch in range(1):
                avg_loss, epoch_time = self.do_epoch(data, optimizer)
                
            
        except KeyboardInterrupt:
            self.logger.log('-' * 89)
            self.logger.log('Exiting from training early')

        self.logger.log('\nTraining time {:10s}'.format(
            str(timedelta(seconds=int(time.time() - start_time)))))

        #self.load_model(self.lr)
        self.logger.log('=' * 89)
        self.logger.log('=' * 89)

        epochLosses = np.vstack((np.array(epochLog).T,np.array(lossLog).T)).T
        np.save("./{0}_losses_{1}.npy".format(data_path[:-4],self.seed),epochLosses)


        return 

    def do_epoch(self, data, optimizer):
        self.model.train()
        avg_loss = 0
        epoch_start_time = time.time()
        batches = data.get_batches(self.batch_size)
        #print(type(batches)) #list, each element of which is 1 batch 
        for batch in batches:
            self.model.zero_grad()
            X, Y1, targetTruths = data.tensorize_batch(batch, self.device, self.model.width, self.truth_known)
            #print("X shape is {0}".format(X.shape))    # Batchsize x 2width x (num_planet_features + num_stellar_features)
            #print("Y1 shape is {0}".format(Y1.shape))  # Batchsize x (num_planet_features + num_stellar_features)
            loss = self.model(X, Y1, is_training=True) # runs MMIModel.forward(X, Y1, is_training=True)

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
        all_future_context_probs = np.zeros((1,self.model.num_labels))
        all_idxs = np.zeros((1,1))
        all_context_idxs = np.zeros((1,1))
        avg_truth_loss = 0.
        with torch.no_grad():
            for batch in batches:
                X, Y1, targetTruths = data.tensorize_batch(batch, self.device, self.model.width, self.truth_known)

                if self.truth_known:
                    truth_loss = calculateLoss(targetTruths,data_path)
                    avg_truth_loss += truth_loss / len(batches)

                future_probs, future_max_probs, future_indices, future_context_probs, future_max_context_probs, future_context_indices = self.model(X, Y1, is_training=False)
                all_future_probs = np.vstack((all_future_probs,future_probs.numpy()))
                all_idxs = np.vstack((all_idxs,np.atleast_2d(np.array(data.targetIdxs)).T))

                all_future_context_probs = np.vstack((all_future_context_probs,future_context_probs.numpy()))
                
                for k, (i, j) in enumerate(batch):
                    z = future_indices[k].max()
                    zseqs[i][j] = z
                    clustering[z][data.systems[i][j]] = True

        all_future_probs = all_future_probs[1:]
        all_future_context_probs = all_future_context_probs[1:]
        all_idxs = all_idxs[1:].astype(int)
        all_idxs = all_idxs[:,0] - 1 # 1-indexing to 0-indexing

        np.save("./{0}_classprobs_softmax_{1}.npy".format(data_path[:-4],self.seed),all_future_probs)
        np.save("./{0}_classprobs_fromcontext_logsoftmax_{1}.npy".format(data_path[:-4],self.seed),all_future_context_probs)
        np.save("./{0}_idxs_{1}.npy".format(data_path[:-4],self.seed),all_idxs)
        np.save("./{0}_optimalLoss_{1}.npy".format(data_path[:-4],self.seed),avg_truth_loss)
        
        maxMI = (- avg_truth_loss) * math.log(math.e,2)
        print("avg_truth_loss is {0}; max MI is {1}".format(avg_truth_loss,maxMI))
        
        return future_probs, zseqs, clustering

    def cross_validate(self, CVdata_path, CVdata):
        self.model.eval()
        batches = CVdata.get_batches(self.batch_size)
        
        all_future_probs = np.zeros((1,self.model.num_labels))
        all_future_context_probs = np.zeros((1,self.model.num_labels))
        all_idxs = np.zeros((1,1))
        all_context_idxs = np.zeros((1,1))
        avg_truth_loss = 0.
        with torch.no_grad():
            for batch in batches:
                X, Y1, targetTruths = CVdata.tensorize_batch(batch, self.device, self.model.width, self.truth_known)

                if self.truth_known:
                    truth_loss = calculateLoss(targetTruths,CVdata_path)
                    avg_truth_loss += truth_loss / len(batches)

                future_probs, future_max_probs, future_indices, future_context_probs, future_max_context_probs, future_context_indices = self.model(X, Y1, is_training=False)
                all_future_probs = np.vstack((all_future_probs,future_probs.numpy()))
                all_idxs = np.vstack((all_idxs,np.atleast_2d(np.array(CVdata.targetIdxs)).T))

                all_future_context_probs = np.vstack((all_future_context_probs,future_context_probs.numpy()))
                
        all_future_probs = all_future_probs[1:]
        all_future_context_probs = all_future_context_probs[1:]
        all_idxs = all_idxs[1:].astype(int)
        all_idxs = all_idxs[:,0] - 1 # 1-indexing to 0-indexing

        np.save("./{0}_classprobs_softmax_{1}.npy".format(CVdata_path[:-4],self.seed),all_future_probs)
        np.save("./{0}_classprobs_fromcontext_logsoftmax_{1}.npy".format(CVdata_path[:-4],self.seed),all_future_context_probs)
        np.save("./{0}_idxs_{1}.npy".format(CVdata_path[:-4],self.seed),all_idxs)
        np.save("./{0}_optimalLoss_{1}.npy".format(CVdata_path[:-4],self.seed),avg_truth_loss)
        
        return future_probs

    def predict_missing(self, CVdata_path, CVdata):
        self.model.eval()
        batches = CVdata.get_batches(1)
        
        predicted_props = np.zeros((1,self.model.num_planet_features))
        all_idxs = np.zeros((1,1))

        with torch.no_grad():
            for batch in batches:
                X, Y1, targetTruths = CVdata.tensorize_batch(batch, self.device, self.model.width, self.truth_known)

                future_probs, future_max_probs, future_indices, future_context_probs, future_max_context_probs, future_context_indices = self.model(X, Y1, is_training=False)
                
                predicted_props_ = self.model.reverse_planet(Context_logSoftmax_Output=future_context_probs, n_iter=100)
                
                predicted_props = np.vstack((predicted_props,predicted_props_.numpy()))
                all_idxs = np.vstack((all_idxs,np.atleast_2d(np.array(CVdata.targetIdxs)).T))

        predicted_props = predicted_props[1:]
        
        all_idxs = all_idxs[1:].astype(int)
        all_idxs = all_idxs[:,0] - 1 # 1-indexing to 0-indexing

        np.save("./{0}_prediction_idxs_{1}.npy".format(CVdata_path[:-4],self.seed),all_idxs)
        np.save("./{0}_predicted_props_{1}.npy".format(CVdata_path[:-4],self.seed),predicted_props)
        
        return

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
