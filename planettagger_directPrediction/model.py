import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from viz import plot_net


class MMIModel(nn.Module):

    def __init__(self, num_planet_features, num_stellar_features, num_labels, width, dropout_prob, feature_names, plot, saveplot, plot_path):
        """
        A class inheriting from torch.nn.Module, the base class of pytorch neural network
        modules.
        
        In the MMIModel.forward() step, we run the target and context models forward
        and calculate the loss function.

        Parameters
        ---------
        num_planet_features : int
            The number of training features related to the planet (e.g. 2, if we're
            training on Rp and P).
        num_stellar_features : int
            The number of training features related to the star (e.g. 3, if we're
            training on Teff, logg, Fe/H).
        num_labels : int
            The number of classes to induce.
        width : int
            The number of context planets to look at on each side of the target planet.
        dropout_prob : float
            The probability of dropout in the target and context hidden layers.
        feature_names : list
            The names of the features (organized first planet then star)
        plot : bool
            Whether to live-plot the network training
        saveplot : bool
            Whether to save plots of the network weight strengths after training.
        """
        super(MMIModel, self).__init__()

        # network parameters
        self.num_planet_features = num_planet_features
        self.num_stellar_features = num_stellar_features
        self.feature_names = feature_names
        self.num_total_features = (num_planet_features + num_stellar_features)
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob
        self.width = width
        
        # specify three torch.nn.Module classes. Each of these has an __init__()
        # step, which defines what each layer is, and a forward() step, which defines
        # how the layers are connected.
        # the loss class
        self.loss = Loss()
        # the context planet network class. Given some input data, returns the
        # network output on those data (an array of class probabilities for each target)
        # as well as the network weights and biases at run time.
        self.contextPlanets = ContextPlanetRep(width, num_planet_features, num_stellar_features, num_labels, dropout_prob)
        # the target planet network class. Given some input data, returns the
        # network output on those data (an array of class probabilities for each target)
        # as well as the network weights and biases at run time.
        self.targetPlanet = TargetPlanetRep(num_planet_features, num_labels, dropout_prob)

        # set up plotting framework
        self.plot = plot
        self.saveplot = saveplot
        self.plot_path = plot_path
        self.iteration = 0

        self.planet_plottedWeights = []
        self.planet_plottedBiases = []
        self.planet_fig = None
        self.planet_cb = None

        self.context_plottedWeights = []
        self.context_plottedBiases = []
        self.context_fig = None
        self.context_cb = None

    def forward(self, contextPlanetData, targetPlanetData, is_training=True, batch_size=1, last_epoch=False, softmax_scale=1): #softmax_scale=0.0005):
        """
        Run both the context and target networks forward.

        Parameters
        ---------
        contextPlanetData : obj
            A torch.Tensor object containing the context data
        targetPlanetData : obj
            A torch.Tensor object containing the target data
        is_training : bool
            Whether we're training the network (true) or just evaluating it (false)
        batch_size : int
            How many planets there are in the batch
        last_epoch : bool
            Whether it's the last training epoch
        softmax_scale : float
            A "softening" scale for the softmax function, to avoid numerical errors trying
            to take log(0). The smaller this value, the more extreme the softening. 
            softmax_scale = 1 means no softening.
            [This seemed to only be a problem when I was first getting the
            code to work, no longer necessary.]

        Returns
        ---------
        if is_training: returns loss (float), the value of the loss function on the input data
        if not is_training: 
            returns F.softmax(softmax_scale*target_rep, dim=1), a torch.Tensor obj 
                    containing the softmax'd class probabilities for the target
                    F.log_softmax(softmax_scale*context_rep, dim=1), a torch.Tensor obj
                    containing the log-softmax'd class probabilities for the context
        """

        # Run the context data through the context network.
        # context_rep is a torch.Tensor object.
        # context_weights is a list of numpy arrays, of shapes:
        # [ (numInputFeatures, numLayer1Weights), (numLayer1Weights, numLayer2Weights),
        # ..., (numLayerN-1Weights, numLayerNWeights), (numLayerNWeights, numClasses) ]
        # context_biases is a list of numpy arrays, of shapes:
        # [ (numLayer1Weights), ..., (numLayerNWeights), (numClasses) ]
        context_rep, context_weights, context_biases = self.contextPlanets(contextPlanetData)
        
        # Run the target data through the target network.
        target_rep, target_weights, target_biases = self.targetPlanet(targetPlanetData)

        # if we're training, calculate the loss function and do plotting
        if is_training:
            loss = self.loss(context_rep, target_rep)

            if self.saveplot is True:
                if last_epoch:
                    if self.iteration % batch_size == 0:
                        self.planet_fig, self.planet_cb, self.planet_plottedWeights, self.planet_plottedBiases = plot_net(self.planet_fig,
                                                                                self.planet_cb,
                                                                                self.planet_plottedWeights, 
                                                                                self.planet_plottedBiases, 
                                                                                target_weights, 
                                                                                target_biases, 
                                                                                net_name="planet representation",
                                                                                n_planet_features=self.num_planet_features,
                                                                                n_stellar_features=self.num_stellar_features,
                                                                                feature_names=self.feature_names,
                                                                                context_rep=False,
                                                                                pause_time=0.01,
                                                                                showplot=False,
                                                                                save=True,
                                                                                figname="{0}_targetNetwork".format(self.plot_path))
                                    
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
                                                                                figname="{0}_contextNetwork".format(self.plot_path))
                        plt.close('all')
            if self.plot is True:
                # Only update plots every len(dataset) training iterations.
                if self.iteration % batch_size == 0:
                    self.planet_fig, self.planet_cb, self.planet_plottedWeights, self.planet_plottedBiases = plot_net(self.planet_fig,
                                                                        self.planet_cb,
                                                                        self.planet_plottedWeights, 
                                                                        self.planet_plottedBiases, 
                                                                        target_weights, 
                                                                        target_biases, 
                                                                        net_name="planet representation",
                                                                        n_planet_features=self.num_planet_features,
                                                                        n_stellar_features=self.num_stellar_features,
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
                                                                        n_planet_features=self.num_planet_features,
                                                                        n_stellar_features=self.num_stellar_features,
                                                                        feature_names=self.feature_names,
                                                                        context_rep=True,
                                                                        context_width=2,
                                                                        showplot=True,
                                                                        pause_time=0.01,
                                                                        save=False)
            
                if last_epoch:
                    plt.close('all')
            self.iteration += 1
            return loss

        else:
            # evaluate the target and context networks; return predictions of these networks
            return target_rep, context_rep

class Loss(nn.Module):
    """
    A class inheriting from torch.nn.Module, the base class of pytorch neural network
    modules.
        
    In the Loss.forward() step, we calculate the loss function on the two networks' predictions.
    """
    def __init__(self):
        super(Loss, self).__init__()
        
        # the function that calculates information entropy
        self.entropy = Entropy()

    def forward(self, contextPlanetRep, targetPlanetRep, softmax_scale=1.):# ,softmax_scale=0.0005):
        """
        Run both the context and target networks forward.

        Parameters
        ---------
        contextPlanetRep : obj
            A torch.Tensor object containing the network predictions on the context data, 
            shape (batchsize, num_labels)
        targetPlanetRep : obj
            A torch.Tensor object containing the network predictions on the target data, 
            shape (batchsize, num_labels)
        softmax_scale : float
            A "softening" scale for the softmax function, to avoid numerical errors trying
            to take log(0). The smaller this value, the more extreme the softening. 
            softmax_scale = 1 means no softening.
            [This seemed to only be a problem when I was first getting the
            code to work, no longer necessary.]

        Returns
        ---------
        loss : float
            value of the loss function
        """

        # just use MSE????
        loss = nn.MSELoss()
        output = loss(contextPlanetRep, targetPlanetRep)

        return output

class Entropy(nn.Module):
    """
    A class inheriting from torch.nn.Module, the base class of pytorch neural network
    modules.
        
    In the Entropy.forward() step, we calculate the information entropy of an array of
    class membership probabilities (length = nClasses)
    """

    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, probs):
        """
        Calculates information entropy H(x) = -sum (p(x) * ln(p(x)))

        Parameters
        ---------
        probs : obj
            A torch.Tensor object of shape [nClasses]
        Returns
        ---------
        entropy : float
            The information entropy
        """
        x = probs * torch.log(probs)
        entropy = -1.0 * x.sum()
        return entropy

class ContextPlanetRep(nn.Module):
    """
    A class inheriting from torch.nn.Module, the base class of pytorch neural network
    modules. This network predicts the target planet's class membership based on the
    context data.

    In the __init__() step, we define what each layer is.
    In the forward() step, we define how the layers are connected.

    """

    def __init__(self, width, num_planet_features, num_stellar_features, num_labels, dropout_prob):
        """
        Initialize the context network.

        Parameters
        ---------
        num_planet_features : int
            The number of training features related to the planet (e.g. 2, if we're
            training on Rp and P).
        num_stellar_features : int
            The number of training features related to the star (e.g. 3, if we're
            training on Teff, logg, Fe/H).
        num_labels : int
            The number of classes to induce.
        dropout_prob : float
            The probability of dropout in the target and context hidden layers.
        """

        super(ContextPlanetRep, self).__init__()

        # Number of nodes per layer, from input to output
        self.architecture = [num_stellar_features + (2*width*num_planet_features), 10, 5, num_planet_features]
        
        # Network layers
        self.linear1 = nn.Linear(self.architecture[0],self.architecture[1])
        self.linear2 = nn.Linear(self.architecture[1],self.architecture[2])
        self.linear3 = nn.Linear(self.architecture[2],self.architecture[3])

        # Network layers organized into list
        self.layers = [self.linear1, self.linear2, self.linear3]

        self.drop_layer = nn.Dropout(p=dropout_prob)

    def forward(self, contextPlanetData):
        """
        Run the context planet network forward.

        Parameters
        ---------
        contextPlanetData : obj
            A torch.Tensor object of shape (batchsize, numStellarFeatures + 2width*numPlanetFeatures)

        Returns
        ---------
        rep : obj
            A torch.Tensor object of shape (batchsize, nClasses) containing the network predictions
            on the input context data
        weights : list
            A list of numpy arrays
            The ith entry is a numpy array containing the weights of the ith network layer
        biases : list
            A list of numpy arrays
            The ith entry is a numpy array containing the biases of the ith network layer
        """

        # ReLU activation function between layers; dropout at every layer
        rep = self.drop_layer(self.linear1(contextPlanetData.view(contextPlanetData.shape[0], -1)))
        rep = self.drop_layer(F.relu(self.linear2(rep)))
        rep = self.drop_layer(self.linear3(rep))

        weights = []
        biases = []

        for layer in self.layers:
            weights.append(list(layer.parameters())[0].data.numpy().T)
            biases.append(list(layer.parameters())[1].data.numpy().T)

        return rep, weights, biases

class TargetPlanetRep(nn.Module):
    """
    A class inheriting from torch.nn.Module, the base class of pytorch neural network
    modules. This network predicts the target planet's class membership based on the
    target data.

    In the __init__() step, we define what each layer is.
    In the forward() step, we define how the layers are connected.
    """

    def __init__(self, num_planet_features, num_labels,dropout_prob):
        """
        Initialize the target network.

        Parameters
        ---------
        num_planet_features : int
            The number of training features related to the planet (e.g. 2, if we're
            training on Rp and P).
        num_stellar_features : int
            The number of training features related to the star (e.g. 3, if we're
            training on Teff, logg, Fe/H).
        num_labels : int
            The number of classes to induce.
        dropout_prob : float
            The probability of dropout in the target and context hidden layers.
        """
        super(TargetPlanetRep, self).__init__()

        self.architecture = [num_planet_features]
        self.identity = nn.Identity(self.architecture[0], self.architecture[0])
        
        self.layers = [self.identity]

    def forward(self, targetPlanetData):
        """
        Run the target planet network forward.

        Parameters
        ---------
        targetPlanetData : obj
            A torch.Tensor object of shape (batchsize, numPlanetFeatures)

        Returns
        ---------
        rep : obj
            A torch.Tensor object of shape (batchsize, nClasses) containing the network predictions
            on the input target data
        weights : list
            A list of numpy arrays
            The ith entry is a numpy array containing the weights of the ith network layer
        biases : list
            A list of numpy arrays
            The ith entry is a numpy array containing the biases of the ith network layer
        """

        # ReLU activation function between layers; dropout at every layer
        rep = self.identity(targetPlanetData)

        weights = []
        biases = []

        return rep, weights, biases

