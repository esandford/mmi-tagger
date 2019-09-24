from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MMIModel(nn.Module):

    def __init__(self, num_planet_features, num_stellar_features, num_labels, width, dropout_prob):
        # In the __init__() step, define what each layer is. In the forward() step,
        # define how the layers are connected.
        super(MMIModel, self).__init__()
        self.num_planet_features = num_planet_features
        self.num_stellar_features = num_stellar_features
        self.num_total_features = (num_planet_features + num_stellar_features)
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob
        self.width = width
        self.loss = Loss()
        #self.iteration = 0

        self.planetContext = ContextRep(width, num_planet_features, num_stellar_features, num_labels, dropout_prob)
        self.indivPlanet = PlanetRep(num_planet_features, num_labels, dropout_prob)
        #print(type(self.planetContext))
        #print(type(self.indivPlanet))
    def forward(self, planetContextData, indivPlanetData, is_training=True, softmax_scale=0.0005):
        context_rep = self.planetContext(planetContextData)
        planet_rep = self.indivPlanet(indivPlanetData)
        if is_training:
            loss = self.loss(context_rep, planet_rep)
            return loss

        else:
            future_max_probs, future_indices = planet_rep.max(1)
            return F.softmax(softmax_scale*planet_rep, dim=1), future_max_probs, future_indices


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.entropy = Entropy()

    def forward(self, planetContextRep, indivPlanetRep, softmax_scale=0.0005):
        #print("indivPlanetRep shape is {0}".format(indivPlanetRep.shape))      # Batchsize x num_labels
        #print("planetContextRep shape is {0}".format(planetContextRep.shape))  # Batchsize x num_labels
        #print("softmax_scale is {0}".format(softmax_scale))
        pZ_Y = F.softmax(softmax_scale*indivPlanetRep, dim=1) # B x num_labels
        #print(pZ_Y.shape)
        #print(type(pZ_Y))
        #print(pZ_Y)
        pZ = pZ_Y.mean(0) # num_labels
        #print(pZ.shape)
        #print("pZ shape is {0}".format(pZ.shape))
        hZ = self.entropy(pZ) # scalar
        #print("hZ shape is {0}".format(hZ.shape))

        #print("pZ_Y shape is {0}".format(pZ_Y.shape))
        #print("log_softmax shape is {0}".format(F.log_softmax(softmax_scale*planetContextRep, dim=1).shape))
        x = pZ_Y * F.log_softmax(softmax_scale*planetContextRep, dim=1)  # B x num_labels * B x num_labels (elementwise multiplication)
        #print("x shape is {0}".format(x.shape))
        #print("x is {0}".format(x))
        hZ_X_ub = -1.0 * x.sum(dim=1).mean() # scalar
        #print("hZ_X_ub shape is {0}".format(hZ_X_ub.shape))

        loss = hZ_X_ub - hZ
        #print("loss is {0}".format(loss))
        return loss

class Entropy(nn.Module):

    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, probs):
        """
        H(x) = -sum (p(x) * ln(p(x)))
        """
        x = probs * torch.log(probs)
        entropy = -1.0 * x.sum()
        return entropy

class ContextRep(nn.Module):

    def __init__(self, width, num_planet_features, num_stellar_features, num_labels, dropout_prob):
        # In the __init__() step, define what each layer is. In the forward() step,
        # define how the layers are connected.
        super(ContextRep, self).__init__()
        #self.linear = nn.Linear(2 * width * num_planet_features, num_labels) #(in_features, out_features) 
                                                                             # = (2width x numPlanetFeatures, numLabels)
                                                                             # = e.g. (4x5, numLabels)
                                                                             # = (20, numLabels)
        self.linear1 = nn.Linear(2 * width * (num_planet_features + num_stellar_features), 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, num_labels)
        self.drop_layer = nn.Dropout(p=dropout_prob)
        self.iteration = 0
        self.linear1weights = np.zeros((2*width*(num_planet_features + num_stellar_features)*20))
        self.linear1biases = np.zeros((20))


    def forward(self, contextPlanetData):
        #contextPlanets will be of the shape: Batchsize x 2width x numPlanetFeatures, e.g. shape (15, 4, 5)
        # self.linear1 wants to operate on something of shape (Batchsize, 2width*numPlanetFeatuers), e.g. (15, 20)
        rep = self.drop_layer(self.linear1(contextPlanetData.view(contextPlanetData.shape[0], -1)))  # returns Batchsize x numLabels
        rep = self.drop_layer(F.relu(self.linear2(rep)))
        rep = self.drop_layer(F.relu(self.linear3(rep)))
        #print(rep.shape)
        #print(type(rep))
        #rint("ContextRep:")
        #print(rep)
        if self.iteration % 100 == 0:
            #print(self.iteration)
            #print(list(self.linear1.parameters()))
            weights = list(self.linear1.parameters())[0].data
            biases = list(self.linear1.parameters())[1].data

            #print("weights: ")
            #print(weights)
            #print("biases: ")
            #print(biases)
            #print(type(weights.data))
            #print(type(biases.data))
            #print(weights.data)
            #print(biases.data)
            #print(np.shape(weights.numpy()))
            #print(weights)
            #print(np.shape(biases.numpy()))
            #print(biases)
            self.linear1biases = np.vstack((self.linear1biases,biases))
            self.linear1weights = np.vstack((self.linear1weights,np.ravel(weights)))
            #print(np.shape(self.linear1biases))
            np.save("./ContextRep_linear1biases.npy",self.linear1biases)
            np.save("./ContextRep_linear1weights.npy",self.linear1weights)

        self.iteration += 1

        return rep


class PlanetRep(nn.Module):

    def __init__(self, num_planet_features, num_labels,dropout_prob):
        # In the __init__() step, define what each layer is. In the forward() step,
        # define how the layers are connected.
        super(PlanetRep, self).__init__()
        #self.linear = nn.Linear(num_planet_features,num_labels) # e.g. (5, numLabels)
        self.linear1 = nn.Linear(num_planet_features,20)
        self.linear2 = nn.Linear(20,10)
        self.linear3 = nn.Linear(10,num_labels)
        self.drop_layer = nn.Dropout(p=dropout_prob)
        self.iteration = 0
        self.linear1weights = np.zeros((num_planet_features*20))
        self.linear1biases = np.zeros((20))

    def forward(self, indivPlanetData):
        #self.linear wants to operate on something of shape (Batchsize, num_planet_features)
        #print(indivPlanetData.shape)
        #print(type(indivPlanetData)) # torch.Tensor
        #print(indivPlanetData.shape) # Batchsize x numPlanetFeatures
        #print(indivPlanetData.view(indivPlanetData.shape[0],-1).shape) #also Batchsize x numPlanetFeatures. in fact same as indivPlanetData
        #rep = self.linear(indivPlanetData.view(indivPlanetData.shape[0],-1))  # B x num_labels
        rep = self.drop_layer(self.linear1(indivPlanetData)) #shape Batchsize x num_labels
        rep = self.drop_layer(F.relu(self.linear2(rep)))
        rep = self.drop_layer(F.relu(self.linear3(rep)))
        #print(type(rep)) #Torch tensor full of nans---but only full of nans because something is bad about the loss or entropy functions.
        #print(rep.shape)
        #print("PlanetRep:")
        #print(rep)
        if self.iteration % 100 == 0:
            #print(list(self.linear1.parameters()))
            weights = list(self.linear1.parameters())[0].data
            biases = list(self.linear1.parameters())[1].data

            #print("weights: ")
            #print(weights)
            #print("biases: ")
            #print(biases)
            #print(type(weights.data))
            #print(type(biases.data))
            #print(weights.data)
            #print(biases.data)
            #print(np.shape(weights.numpy()))
            #print(weights)
            #print(np.shape(biases.numpy()))
            #print(biases)
            self.linear1biases = np.vstack((self.linear1biases,biases))
            self.linear1weights = np.vstack((self.linear1weights,np.ravel(weights)))

            np.save("./PlanetRep_linear1biases.npy",self.linear1biases)
            np.save("./PlanetRep_linear1weights.npy",self.linear1weights)

        self.iteration += 1

        return rep

