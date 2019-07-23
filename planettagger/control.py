from __future__ import division, print_function
import data
import math
import random
import time
import torch
import torch.nn as nn

from datetime import timedelta

class Control(nn.Module):

    def __init__(self, model, model_path, batch_size, device, logger):
        super(Control, self).__init__()
        self.model = model
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.logger = logger

    def train(self, data, lr, epochs):
        self.log_data(data)
        self.logger.log('[TRAINING]')
        self.logger.log('   num_labels:    %d' % self.model.num_labels)
        self.logger.log('   batch_size:    %d' % self.batch_size)
        self.logger.log('   lr:            %g' % lr)
        self.logger.log('   epochs:        %d' % epochs)
        self.logger.log('')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        start_time = time.time()

        try:
            for epoch in range(1, epochs + 1):
                avg_loss, epoch_time = self.do_epoch(data, optimizer)
                #print("avg_loss is {0}".format(avg_loss))
                #print("epoch_time is {0}".format(epoch_time))
                bits = (- avg_loss) * math.log(math.e,2)
                self.logger.log('| epoch {:3d} | loss {:6.2f} | {:6.2f} bits | '
                                'time {:10s}'.format(
                                    epoch, avg_loss, bits,
                                    str(timedelta(seconds=int(epoch_time)))))
                with open(self.model_path, 'wb') as f:
                    torch.save(self.model, f)

        except KeyboardInterrupt:
            self.logger.log('-' * 89)
            self.logger.log('Exiting from training early')

        self.logger.log('\nTraining time {:10s}'.format(
            str(timedelta(seconds=int(time.time() - start_time)))))

        self.load_model()
        self.logger.log('=' * 89)
        self.logger.log('=' * 89)

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
            X, Y1 = data.tensorize_batch(batch, self.device, self.model.width)
            #print("X shape is {0}".format(X.shape))    # Batchsize x 2width x numPlanetFeatures
            #print("Y1 shape is {0}".format(Y1.shape))  # Batchsize x numPlanetFeatures
            loss = self.model(X, Y1, is_training=True) # runs MMIModel.forward(X, Y1, is_training=True)
            #print("loss is {0}".format(loss))
            avg_loss += loss.item() / len(batches)
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start_time
        
        return avg_loss, epoch_time

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = torch.load(f)

    def log_data(self, data):
        self.logger.log('-' * 89)
        self.logger.log('[DATA]')
        self.logger.log('   data:          %s' % data.data_path)
        self.logger.log('   # planets:     %d' % sum(len(sys) for sys in data.systems))
        self.logger.log('-' * 89)
