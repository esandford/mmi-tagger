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
    """
    calculate information entropy, S = -sum_i[ p_i * ln(p_i) ]

    Parameters
    ---------
    probs : np.array
        The class membership probabilities of a target planet. Length n_classes

    Returns
    ---------
    entro : float
        The information entropy of this array of probabilities
    """
    x = np.multiply(probs,np.log(probs))
    entro = -1. * np.sum(x)
    return entro

def calculateLoss(targetTruths):
    """
    Calculate the loss function if everything were labeled correctly, i.e. the "goal"
    loss function.
    
    Parameters
    ---------
    targetTruths : list 
        list of length Batchsize, each entry of which is a value between 0...(nclasses-1)
        corresponding to the true class membership of that target planet.
    
    Returns
    ---------
    loss : float
        The value of the (ideal) loss function
    """

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
    """
    A class inheriting from torch.nn.Module, the base class of pytorch neural network modules.
    Has functions to train our network, evaluate it on the training set, evaluate it on
    a holdout CV set, etc.

    """

    def __init__(self, model, model_path, batch_size, device, logger, truth_known, seed, results_path):
        """
        Initialize the Control object.

        Parameters
        ---------
        model : obj
            a model.MMIModel object, which inherits from torch.nn.Module
        model_path : str
            a path specifying where the model will be saved
        batch_size : int
            the size of each training batch
        device : obj
            a torch.device object ('cuda' or 'cpu') 
        logger : obj
            a logger.Logger object which can write to stdout and our log file
        truth_known : bool
            whether the true class of the planets in the training set is known 
            (as it would be for simulated data) or not (as it would be for real data)
        seed : int
            the random seed of the model
        """
        super(Control, self).__init__()
        self.model = model
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.logger = logger
        self.truth_known = truth_known
        self.seed = seed
        self.results_path = results_path

    def train(self, data, data_path, lr, epochs):
        """
        Train self.model (which is specified at initialization of the Control object)
        for a specified number of epochs.

        Parameters
        ---------
        data : obj
            a data.Data object containing the training data
        data_path : str
            a path to the training data file
        lr : float
            the learning rate to be used by the Adam optimizer
        epochs : int
            the number of training epochs to do 
        """

        # Write metadata to the log file
        self.log_data(data)
        self.logger.log('[TRAINING]')
        self.logger.log('   num_labels:    %d' % self.model.num_labels)
        self.logger.log('   batch_size:    %d' % self.batch_size)
        self.logger.log('   lr:            %g' % lr)
        self.logger.log('   epochs:        %d' % epochs)
        self.logger.log('')

        # Initialize the model optimizer. use the Adam algorithm, which is for
        # "first-order, gradient-based optimization of stochastic objective
        # functions" (see https://arxiv.org/abs/1412.6980)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        start_time = time.time()

        epochLog = []
        lossLog = []

        try:
            for epoch in range(1, epochs + 1):
                # is it the last epoch?
                if epoch == epochs:
                    last_epoch = True
                else:
                    last_epoch = False
                # calculate the average value of the loss function over a training epoch.
                # also keep track of how long that training epoch took.
                avg_loss, epoch_time = self.do_epoch(data, optimizer, last_epoch)

                # calculate the number of bits of information learned
                bits = (- avg_loss) * math.log(math.e,2)

                # write the results of this training epoch to stdout and the log file
                self.logger.log('| epoch {:3d} | loss {:6.12f} | {:6.12f} bits | '
                                'time {:10s}'.format(
                                    epoch, avg_loss, bits,
                                    str(timedelta(seconds=int(epoch_time)))))

                epochLog.append(epoch)
                lossLog.append(avg_loss)

                # save the state of the model after this training epoch
                with open(self.model_path, 'wb') as f:
                    state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                     'optimizer': optimizer.state_dict()}
                    torch.save(state, f)

        except KeyboardInterrupt:
            self.logger.log('-' * 89)
            self.logger.log('Exiting from training early')

        # write time elapsed during training to stdout and the log file
        self.logger.log('\nTraining time {:10s}'.format(
            str(timedelta(seconds=int(time.time() - start_time)))))
        self.logger.log('=' * 89)
        self.logger.log('=' * 89)

        # write the value of the loss function at each epoch to a .npy file, 
        # for easier analysis later (i.e. so we don't have to parse the log file)
        epochLosses = np.vstack((np.array(epochLog).T,np.array(lossLog).T)).T

        np.save("{0}{1}_losses_{2}.npy".format(self.results_path,data_path.split("/")[-1][:-4],self.seed),epochLosses)

        return 

    def do_epoch(self, data, optimizer, last_epoch):
        """
        Do one training epoch.

        Parameters
        ---------
        data : obj
            a data.Data object containing the training data
        optimizer : obj
            a torch.optim.Adam object

        Returns
        ---------
        avg_loss : float
            The average value of the loss function over the batches.
        epoch_time : float
            How long it took to do this training epoch, in seconds
        last_epoch : bool
            Whether this is the last training epoch or not
        """

        # set the model (an object which inherits from torch.nn.Module) in training mode
        # set the initial value of the average loss at 0 and record the clock time
        # when the training epoch began
        self.model.train()
        avg_loss = 0
        epoch_start_time = time.time()

        # organize the data into batches
        batches = data.get_batches(self.batch_size)
        
        for batch in batches:
            # set gradients of all model parameters to 0
            self.model.zero_grad()

            # convert the data in the batch into torch tensor format
            # X is of type torch.Tensor; it contains the context data
            #   and is of shape [Batchsize, (2width * num_planet_features) + num_stellar_features)]
            # Y is of type torch.Tensor; it contains the target data
            #    and is of shape [Batchsize, num_planet_features]
            # targetTruths is a list of length Batchsize. The ith entry is an integer between 0
            #    and nClasses-1 which indicates the true class of target planet i in the batch
            X, Y, targetTruths = data.tensorize_batch(batch, self.device, self.model.width, self.truth_known)

            # run MMIModel.forward(X, Y, is_training=True)
            loss = self.model(X, Y, batch_size=self.batch_size, last_epoch=last_epoch, is_training=True)

            # Add the value of the loss function for this batch
            # into the average loss over all the batches
            avg_loss += loss.item() / len(batches)

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Calculate how long this training epoch took, in seconds
        epoch_time = time.time() - epoch_start_time
        
        return avg_loss, epoch_time

    def classify(self, data_path, data):
        """
        Evaluate the (trained) model on some data (either training data or holdout test data)

        Parameters
        ---------
        data_path : str
            The path to the data file
        data : obj
            a data.Data object containing the data
        """

        # set the model (an object which inherits from torch.nn.Module) in evaluation mode
        self.model.eval()

        # organize the data to be evaluated into batches
        batches = data.get_batches(self.batch_size)

        # make empty arrays to hold the network's predictions based on target data
        # and context data
        all_target_probs = np.zeros((1,self.model.num_labels))
        all_context_probs = np.zeros((1,self.model.num_labels))

        # make empty arrays to keep track of how the network has reshuffled the data
        all_idxs = np.zeros((1,1))

        # if TRUTH_KNOWN, we want to calculate the *ideal* loss function, i.e.
        # what the loss function would be if the data were perfectly classified.
        # set the initial value of this ideal average loss at 0
        if self.truth_known:
            avg_loss_ideal = 0.

        # in any case, we want to calculate the loss function given the network's
        # actual classifications. set the inital value of this to 0
        avg_loss_actual = 0.

        # temporarily set the torch requires_grad flag to False, to stop autograd
        # from tracking the history on Tensors here. (we're not training, so 
        # we don't need that history.)
        with torch.no_grad():
            for batch in batches:
                # convert the data in the batch into torch tensor format
                # X is of type torch.Tensor; it contains the context data
                #   and is of shape [Batchsize, (2width * num_planet_features) + num_stellar_features)]
                # Y is of type torch.Tensor; it contains the target data
                #    and is of shape [Batchsize, num_planet_features]
                # targetTruths is a list of length Batchsize. The ith entry is an integer between 0
                #    and nClasses-1 which indicates the true class of target planet i in the batch
                X, Y, targetTruths = data.tensorize_batch(batch, self.device, self.model.width, self.truth_known)

                # if truth is known, calculate the value the loss function
                # would have if all the data were classified perfectly
                if self.truth_known:
                    loss_ideal = calculateLoss(targetTruths)
                    avg_loss_ideal += loss_ideal / len(batches)

                # run the target & context models forward. get the 
                # class probabilities of the target planets from the target network
                # (target_probs) and from the context network (context_probs).
                # Both are torch tensors of shape (Batchsize, nClasses).
                target_probs, context_probs = self.model(X, Y, batch_size=self.batch_size, last_epoch=False, is_training=False)

                # calculate the avg value of the loss function over this batch
                pZ = target_probs.mean(0)
                pZ_temp = pZ * torch.log(pZ)
                hZ = -1.0 * pZ_temp.sum()
                x = target_probs * context_probs
                hZ_X_ub = -1.0 * x.sum(dim=1).mean() # scalar

                loss_actual = hZ_X_ub - hZ
                avg_loss_actual += loss_actual / len(batches)

                # stack this batch's probabilities in an array for the entire data set
                all_target_probs = np.vstack((all_target_probs,target_probs.numpy()))
                all_context_probs = np.vstack((all_context_probs,context_probs.numpy()))

                # stack this batch's reshuffled indices in an array for the entire data set
                all_idxs = np.vstack((all_idxs,np.atleast_2d(np.array(data.targetIdxs)).T))
        
        # get rid of first entry, which is 0
        all_target_probs = all_target_probs[1:]
        all_context_probs = all_context_probs[1:]
        all_idxs = all_idxs[1:].astype(int)
        all_idxs = all_idxs[:,0] - 1 # 1-indexing to 0-indexing

        # save class probabilities from target network (softmax form)
        np.save("{0}{1}_classprobs_softmax_{2}.npy".format(self.results_path,data_path.split("/")[-1][:-4],self.seed),all_target_probs)
        
        # save class probabilities from context network (log-softmax form)
        np.save("{0}{1}_classprobs_fromcontext_logsoftmax_{2}.npy".format(self.results_path,data_path.split("/")[-1][:-4],self.seed),all_context_probs)
        
        # save reshuffling indices
        np.save("{0}{1}_idxs_{2}.npy".format(self.results_path,data_path.split("/")[-1][:-4],self.seed),all_idxs)
        
        if self.truth_known:
            # save optimal loss function value
            np.save("{0}{1}_optimalLoss_{2}.npy".format(self.results_path,data_path.split("/")[-1][:-4],self.seed),avg_loss_ideal)
        
            maxMI_ideal = (-avg_loss_ideal) * math.log(math.e,2)
            print("avg_loss_ideal is {0}; max MI is {1}".format(avg_loss_ideal,maxMI_ideal))
        
        # print actual loss function value to stdout
        maxMI_actual = (-avg_loss_actual) * math.log(math.e,2)
        print("avg_loss_actual is {0}; max MI is {1}".format(avg_loss_actual,maxMI_actual))
        
        return

    def load_model(self,lr):
        """
        Load a previous state of the MMIModel.

        Parameters
        ---------
        lr : float
            the learning rate to be used by the Adam optimizer (from now on)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        with open(self.model_path, 'rb') as f:
            checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(f, checkpoint['epoch']))

        return

    def log_data(self, data):
        """
        Log basic facts about the data to the log file and stdout.

        Parameters
        ---------
        data : obj
            a data.Data object
        """
        self.logger.log('-' * 89)
        self.logger.log('[DATA]')
        self.logger.log('   data:          %s' % data.data_path)
        self.logger.log('   # planets:     %d' % sum(len(sys) for sys in data.systems))
        self.logger.log('-' * 89)

        return
