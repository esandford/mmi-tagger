from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMIModel(nn.Module):

    def __init__(self, num_planet_features, num_labels, width):
        # In the __init__() step, define what each layer is. In the forward() step,
        # define how the layers are connected.
        super(MMIModel, self).__init__()
        self.num_planet_features = num_planet_features
        self.num_labels = num_labels
        self.width = width
        self.loss = Loss()

        self.planetContext = ContextRep(width, num_planet_features, num_labels)
        self.indivPlanet = PlanetRep(num_planet_features, num_labels)

    def forward(self, planetContextData, indivPlanetData, is_training=True):
        context_rep = self.planetContext(planetContextData)
        planet_rep = self.indivPlanet(indivPlanetData)
        if is_training:
            loss = self.loss(context_rep, planet_rep)
            return loss

        else:
            future_probs, future_indices = planet_rep.max(1)
            return future_probs, future_indices


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.entropy = Entropy()

    def forward(self, planetContextRep, indivPlanetRep):
        #pZ_Y = F.softmax(indivPlanetRep, dim=1)
        #pZ = pZ_Y.mean(0)
        #hZ = self.entropy(pZ)

        #x = pZ_Y * F.log_softmax(planetContextRep, dim=1)  # B x m
        #hZ_X_ub = -1.0 * x.sum(dim=1).mean()

        #loss = hZ_X_ub - hZ
        #return loss
        return 1

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

    def __init__(self, width, num_planet_features, num_labels):
        # In the __init__() step, define what each layer is. In the forward() step,
        # define how the layers are connected.
        super(ContextRep, self).__init__()
        self.linear = nn.Linear(2 * width * num_planet_features, num_labels) #(in_features, out_features) 
                                                                             # = (2width x numPlanetFeatures, numLabels)
                                                                             # = e.g. (4x5, numLabels)
                                                                             # = (20, numLabels)

    def forward(self, contextPlanetData):
        #contextPlanets will be of the shape: Batchsize x 2width x numPlanetFeatures, e.g. shape (15, 4, 5)
        # self.linear wants to operate on something of shape (Batchsize, 2width*numPlanetFeatuers), e.g. (15, 20)
        rep = self.linear(contextPlanetData.view(contextPlanetData.shape[0], -1))  # returns Batchsize x numLabels
        #print(rep.shape)
        #print(type(rep))
        print("ContextRep:")
        print(rep)
        return rep


class PlanetRep(nn.Module):

    def __init__(self, num_planet_features, num_labels):
        # In the __init__() step, define what each layer is. In the forward() step,
        # define how the layers are connected.
        super(PlanetRep, self).__init__()
        self.linear = nn.Linear(num_planet_features,num_labels) # e.g. (5, numLabels)

    def forward(self, indivPlanetData):
        #self.linear wants to operate on something of shape (Batchsize, num_planet_features)
        #print(indivPlanetData.shape)
        #print(type(indivPlanetData)) # torch.Tensor
        #print(indivPlanetData.shape) # Batchsize x numPlanetFeatures
        #print(indivPlanetData.view(indivPlanetData.shape[0],-1).shape) #also Batchsize x numPlanetFeatures. in fact same as indivPlanetData
        #rep = self.linear(indivPlanetData.view(indivPlanetData.shape[0],-1))  # B x num_labels
        rep = self.linear(indivPlanetData) #shape Batchsize x num_labels
        #print(type(rep)) #Torch tensor full of nans---but only full of nans because something is bad about the loss or entropy functions.
        #print(rep.shape)
        print("PlanetRep:")
        print(rep)
        return rep

