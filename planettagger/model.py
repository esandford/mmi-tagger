from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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

    def forward(self, planetContextData, indivPlanetData, is_training=True, softmax_scale=0.0005):
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
                    
                    self.planet_fig, self.planet_cb, self.planet_plottedWeights, self.planet_plottedBiases = plot_net(self.planet_fig,
                                                                        self.planet_cb,
                                                                        self.planet_plottedWeights, 
                                                                        self.planet_plottedBiases, 
                                                                        planet_weights, 
                                                                        planet_biases, 
                                                                        net_name="planet representation",
                                                                        feature_names=self.feature_names,
                                                                        context_rep=False,
                                                                        showplot=True,
                                                                        pause_time=0.01,
                                                                        save=False)
                    
                    
                    self.context_fig, self.context_cb, self.context_plottedWeights, self.context_plottedBiases = plot_net(self.context_fig,
                                                                        self.context_cb,
                                                                        self.context_plottedWeights, 
                                                                        self.context_plottedBiases, 
                                                                        context_weights, 
                                                                        context_biases, 
                                                                        net_name="context representation",
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
                                                                            feature_names=self.feature_names,
                                                                            context_rep=False,
                                                                            pause_time=0.01,
                                                                            showplot=False,
                                                                            save=True,
                                                                            figname="./simulatedPlanets/oneGrammar_distinctRp/fake_grammaticalSystems_allFeatures_uniformP_planetWeights")
                                
                                
                    self.context_fig, self.context_cb, self.context_plottedWeights, self.context_plottedBiases = plot_net(self.context_fig,
                                                                            self.context_cb,
                                                                            self.context_plottedWeights, 
                                                                            self.context_plottedBiases, 
                                                                            context_weights, 
                                                                            context_biases, 
                                                                            net_name="context representation",
                                                                            feature_names=self.feature_names,
                                                                            context_rep=True,
                                                                            context_width=2,
                                                                            pause_time=0.01,
                                                                            showplot=False,
                                                                            save=True,
                                                                            figname="./simulatedPlanets/oneGrammar_distinctRp/fake_grammaticalSystems_allFeatures_uniformP_contextWeights")
            
            self.iteration += 1
            return loss#, planet_weights, planet_biases, context_weights, context_biases

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
        
        self.architecture = [2*width*(num_planet_features+num_stellar_features), 20, 10, num_labels]
        self.linear1 = nn.Linear(self.architecture[0],self.architecture[1])
        self.linear2 = nn.Linear(self.architecture[1],self.architecture[2])
        self.linear3 = nn.Linear(self.architecture[2],self.architecture[3])
        self.drop_layer = nn.Dropout(p=dropout_prob)
        #self.iteration = 0
        #self.linear1weights = np.zeros((2*width*(num_planet_features + num_stellar_features)*20))
        #self.linear1biases = np.zeros((20))
        self.layers = [self.linear1, self.linear2, self.linear3]


    def forward(self, contextPlanetData):
        #contextPlanets will be of the shape: Batchsize x 2width x numPlanetFeatures, e.g. shape (15, 4, 5)
        # self.linear1 wants to operate on something of shape (Batchsize, 2width*numPlanetFeatuers), e.g. (15, 20)
        rep = self.drop_layer(self.linear1(contextPlanetData.view(contextPlanetData.shape[0], -1)))  # returns Batchsize x numLabels
        rep = self.drop_layer(F.relu(self.linear2(rep)))
        rep = self.drop_layer(F.relu(self.linear3(rep)))
        
        """
        if self.iteration % 100 == 0:
            weights = list(self.linear1.parameters())[0].data
            biases = list(self.linear1.parameters())[1].data

            self.linear1biases = np.vstack((self.linear1biases,biases))
            self.linear1weights = np.vstack((self.linear1weights,np.ravel(weights)))
            np.save("./ContextRep_linear1biases.npy",self.linear1biases)
            np.save("./ContextRep_linear1weights.npy",self.linear1weights)

        self.iteration += 1
        """
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

        self.architecture = [num_planet_features, 20, 10, num_labels]
        self.linear1 = nn.Linear(self.architecture[0], self.architecture[1])
        self.linear2 = nn.Linear(self.architecture[1], self.architecture[2])
        self.linear3 = nn.Linear(self.architecture[2], self.architecture[3])
        self.drop_layer = nn.Dropout(p=dropout_prob)
        #self.iteration = 0
        #self.linear1weights = np.zeros((num_planet_features*20))
        #self.linear1biases = np.zeros((20))
        self.layers = [self.linear1, self.linear2, self.linear3]

    def forward(self, indivPlanetData):
        #self.linear wants to operate on something of shape (Batchsize, num_planet_features)
        #rep = self.linear(indivPlanetData.view(indivPlanetData.shape[0],-1))  # B x num_labels
        rep = self.drop_layer(self.linear1(indivPlanetData)) #shape Batchsize x num_labels
        rep = self.drop_layer(F.relu(self.linear2(rep)))
        rep = self.drop_layer(F.relu(self.linear3(rep)))
        """
        if self.iteration % 100 == 0:
            weights = list(self.linear1.parameters())[0].data
            biases = list(self.linear1.parameters())[1].data
            self.linear1biases = np.vstack((self.linear1biases,biases))
            self.linear1weights = np.vstack((self.linear1weights,np.ravel(weights)))

            np.save("./PlanetRep_linear1biases.npy",self.linear1biases)
            np.save("./PlanetRep_linear1weights.npy",self.linear1weights)

        self.iteration += 1
        """
        weights = []
        biases = []

        for layer in self.layers:
            weights.append(list(layer.parameters())[0].data.numpy().T)
            biases.append(list(layer.parameters())[1].data.numpy().T)

        return rep, weights, biases

