import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from viz import plot_net

class MMIModel(nn.Module):

    def __init__(self, num_planet_features, num_stellar_features, num_labels, width, dropout_prob, feature_names, plot, saveplot):
        # In the __init__() step, define what each layer is. In the forward() step,
        # define how the layers are connected.
        super(MMIModel, self).__init__()
        self.num_planet_features = num_planet_features
        self.num_stellar_features = num_stellar_features
        self.feature_names = feature_names
        self.num_total_features = (num_planet_features + num_stellar_features)
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob
        self.width = width
        self.loss = Loss()
        self.plot = plot
        self.saveplot = saveplot
        self.iteration = 0

        self.planetContext = ContextRep(width, num_planet_features, num_stellar_features, num_labels, dropout_prob)
        self.indivPlanet = PlanetRep(num_planet_features, num_labels, dropout_prob)

        self.planet_plottedWeights = []
        self.planet_plottedBiases = []
        self.planet_fig = None
        self.planet_cb = None

        self.context_plottedWeights = []
        self.context_plottedBiases = []
        self.context_fig = None
        self.context_cb = None

    def forward(self, planetContextData, indivPlanetData, is_training=True,softmax_scale=1): #softmax_scale=0.0005):
        context_rep, context_weights, context_biases = self.planetContext(planetContextData)
        
        #context_weights is a list of numpy arrays: [ (8, 20), (20, 10), (10, 3) ]
        #context_biases is a list of numpy arrays: [ (20), (10), (3) ]
        
        planet_rep, planet_weights, planet_biases = self.indivPlanet(indivPlanetData)
        #planet_weights is a list of numpy arrays: [ (2, 20), (20, 10), (10, 3) ]
        #planet_biases is a list of numpy arrays: [ (20), (10), (3) ]

        if is_training:
            loss = self.loss(context_rep, planet_rep)

            if self.plot is True:
                if self.iteration % 500 == 0:
                    """
                    self.planet_fig, self.planet_cb, self.planet_plottedWeights, self.planet_plottedBiases = plot_net(self.planet_fig,
                                                                        self.planet_cb,
                                                                        self.planet_plottedWeights, 
                                                                        self.planet_plottedBiases, 
                                                                        planet_weights, 
                                                                        planet_biases, 
                                                                        net_name="planet representation",
                                                                        n_planet_features=self.num_planet_features,
                                                                        n_stellar_features=self.num_stellar_features,
                                                                        feature_names=self.feature_names,
                                                                        context_rep=False,
                                                                        showplot=True,
                                                                        pause_time=0.01,
                                                                        save=False)
                    
                    """
                    self.context_fig, self.context_cb, self.context_plottedWeights, self.context_plottedBiases = plot_net(self.context_fig,
                                                                        self.context_cb,
                                                                        self.context_plottedWeights, 
                                                                        self.context_plottedBiases, 
                                                                        context_weights, 
                                                                        context_biases, 
                                                                        net_name="context representation",
                                                                        n_planet_features=self.num_planet_features,
                                                                        n_stellar_features=self.num_stellar_features,
                                                                        feature_names=self.feature_names,
                                                                        context_rep=True,
                                                                        context_width=2,
                                                                        showplot=True,
                                                                        pause_time=0.01,
                                                                        save=False)
                    
            if self.saveplot is True:
                if self.iteration % 500 == 0:
                    self.planet_fig, self.planet_cb, self.planet_plottedWeights, self.planet_plottedBiases = plot_net(self.planet_fig,
                                                                            self.planet_cb,
                                                                            self.planet_plottedWeights, 
                                                                            self.planet_plottedBiases, 
                                                                            planet_weights, 
                                                                            planet_biases, 
                                                                            net_name="planet representation",
                                                                            n_planet_features=self.num_planet_features,
                                                                            n_stellar_features=self.num_stellar_features,
                                                                            feature_names=self.feature_names,
                                                                            context_rep=False,
                                                                            pause_time=0.01,
                                                                            showplot=False,
                                                                            save=True,
                                                                            figname="./simulatedPlanets/crossValidation/fake_grammaticalSystems_Rp_uniformP_sigma=0.005_planetWeights")
                                
                              
                    self.context_fig, self.context_cb, self.context_plottedWeights, self.context_plottedBiases = plot_net(self.context_fig,
                                                                            self.context_cb,
                                                                            self.context_plottedWeights, 
                                                                            self.context_plottedBiases, 
                                                                            context_weights, 
                                                                            context_biases, 
                                                                            net_name="context representation",
                                                                            n_planet_features=self.num_planet_features,
                                                                            n_stellar_features=self.num_stellar_features,
                                                                            feature_names=self.feature_names,
                                                                            context_rep=True,
                                                                            context_width=2,
                                                                            pause_time=0.01,
                                                                            showplot=False,
                                                                            save=True,
                                                                            figname="./simulatedPlanets/crossValidation/fake_grammaticalSystems_Rp_uniformP_sigma=0.005_contextWeights")
            
            self.iteration += 1
            return loss#, planet_weights, planet_biases, context_weights, context_biases

        else:
            future_max_probs, future_indices = planet_rep.max(1)
            future_max_context_probs, future_context_indices = context_rep.max(1)
            return F.softmax(softmax_scale*planet_rep, dim=1), future_max_probs, future_indices, F.log_softmax(softmax_scale*context_rep, dim=1), future_max_context_probs, future_context_indices

class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.entropy = Entropy()

    def forward(self, planetContextRep, indivPlanetRep,softmax_scale=1.):# ,softmax_scale=0.0005):
        #indivPlanetRep is Batchsize x num_labels
        #planetContextRep is Batchsize x num_labels
        pZ_Y = F.softmax(softmax_scale*indivPlanetRep, dim=1) # B x num_labels

        pZ = pZ_Y.mean(0) # num_labels

        hZ = self.entropy(pZ) # scalar

        x = pZ_Y * F.log_softmax(softmax_scale*planetContextRep, dim=1)  # B x num_labels * B x num_labels (elementwise multiplication)

        hZ_X_ub = -1.0 * x.sum(dim=1).mean() # scalar

        loss = hZ_X_ub - hZ

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

        self.architecture = [num_stellar_features + (2*width*num_planet_features), 100, 10, num_labels]
        
        self.linear1 = nn.Linear(self.architecture[0],self.architecture[1])
        self.linear2 = nn.Linear(self.architecture[1],self.architecture[2])
        self.linear3 = nn.Linear(self.architecture[2],self.architecture[3])
        self.layers = [self.linear1, self.linear2, self.linear3]

        self.drop_layer = nn.Dropout(p=dropout_prob)

    def forward(self, contextPlanetData):
        #contextPlanets will be of the shape: Batchsize x 2width x numPlanetFeatures, e.g. shape (15, 4, 5)
        # self.linear1 wants to operate on something of shape (Batchsize, 2width*numPlanetFeatuers), e.g. (15, 20)
        rep = self.drop_layer(self.linear1(contextPlanetData.view(contextPlanetData.shape[0], -1)))  # returns Batchsize x numLabels
        rep = self.drop_layer(F.relu(self.linear2(rep)))
        rep = self.drop_layer(F.relu(self.linear3(rep)))

        weights = []
        biases = []

        for layer in self.layers:
            weights.append(list(layer.parameters())[0].data.numpy().T)
            biases.append(list(layer.parameters())[1].data.numpy().T)

        return rep, weights, biases

class PlanetRep(nn.Module):

    def __init__(self, num_planet_features, num_labels,dropout_prob):
        # In the __init__() step, define what each layer is. In the forward() step,
        # define how the layers are connected.
        super(PlanetRep, self).__init__()

        self.architecture = [num_planet_features, 100, 10, num_labels]
        self.linear1 = nn.Linear(self.architecture[0], self.architecture[1])
        self.linear2 = nn.Linear(self.architecture[1], self.architecture[2])
        self.linear3 = nn.Linear(self.architecture[2], self.architecture[3])
        self.layers = [self.linear1, self.linear2, self.linear3]
        
        self.drop_layer = nn.Dropout(p=dropout_prob)

    def forward(self, indivPlanetData):
        #self.linear wants to operate on something of shape (Batchsize, num_planet_features)
        #rep = self.linear(indivPlanetData.view(indivPlanetData.shape[0],-1))  # B x num_labels
        rep = self.drop_layer(self.linear1(indivPlanetData)) #shape Batchsize x num_labels
        rep = self.drop_layer(F.relu(self.linear2(rep)))
        rep = self.drop_layer(F.relu(self.linear3(rep)))
        
        weights = []
        biases = []

        for layer in self.layers:
            weights.append(list(layer.parameters())[0].data.numpy().T)
            biases.append(list(layer.parameters())[1].data.numpy().T)

        return rep, weights, biases





