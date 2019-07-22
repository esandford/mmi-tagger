from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMIModel(nn.Module):

    def __init__(self, num_planet_features, num_labels, width):
        super(MMIModel, self).__init__()
        self.num_planet_features = num_planet_features
        self.num_labels = num_labels
        self.width = width
        self.loss = Loss()

        self.planetContext = ContextHandler(width, num_planet_features, num_labels)
        self.indivPlanet = PlanetHandler(num_planet_features, num_labels)

    def forward(self, planetContexts, indivPlanets, is_training=True):
        context_rep = self.planetContext(planetContexts)
        planet_rep = self.indivPlanet(indivPlanets)
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

    def forward(self, planetContexts, indivPlanets):
        pZ_Y = F.softmax(indivPlanets, dim=1)
        pZ = pZ_Y.mean(0)
        hZ = self.entropy(pZ)

        x = pZ_Y * F.log_softmax(planetContexts, dim=1)  # B x m
        hZ_X_ub = -1.0 * x.sum(dim=1).mean()

        loss = hZ_X_ub - hZ
        return loss


class Entropy(nn.Module):

    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, probs):
        x = probs * torch.log(probs)
        entropy = -1.0 * x.sum()
        return entropy

class ContextHandler(nn.Module):

    def __init__(self, width, num_planet_features, num_labels):
        super(ContextHandler, self).__init__()
        self.linear = nn.Linear(2 * width * num_planet_features, num_labels) #(in_features, out_features) 
                                                                             # = (2width x numPlanetFeatures, numLabels)
                                                                             # = e.g. (4x5, numLabels)
                                                                             # = (20, numLabels)

    def forward(self, contextPlanets):
        #contextPlanets will be of the shape: Batchsize x 2width x numPlanetFeatures, e.g. shape (15, 4, 5)
        # self.linear wants to operate on something of shape (Batchsize, 2width*numPlanetFeatuers), e.g. (15, 20)
        rep = self.linear(contextPlanets.view(contextPlanets.shape[0], -1))  # returns Batchsize x numLabels
        return rep


class PlanetHandler(nn.Module):

    def __init__(self, num_planet_features, num_labels):
        super(PlanetHandler, self).__init__()
        self.linear = nn.Linear(num_planet_features,num_labels) # e.g. (5, numLabels)

    def forward(self, planets):
        #self.linear wants to operate on something of shape (Batchsize, num_planet_features)
        rep = self.linear(planets.view(planets.shape[0],-1))  # B x num_planet_features
        return rep

