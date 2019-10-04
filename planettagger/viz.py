from __future__ import division, print_function
import numpy as np
import random
import torch
import pickle

import matplotlib.pyplot as plt

import matplotlib.colors as colors
import matplotlib.cm as cmx

"""
def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1

# the function below is for updating both x and y values (great for updating dates on the x-axis)
def live_plotter_xy(x_vec,y1_data,line1,identifier='',pause_time=0.01):
    if line1==[]:
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        line1, = ax.plot(x_vec,y1_data,'r-o',alpha=0.8)
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
        
    line1.set_data(x_vec,y1_data)
    plt.xlim(np.min(x_vec),np.max(x_vec))
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])

    plt.pause(pause_time)
    
    return line1


size = 100
x_vec = np.linspace(0,1,size+1)[0:-1]
y_vec = np.random.randn(len(x_vec))
line1 = []
while True:
    rand_val = np.random.randn(1)
    y_vec[-1] = rand_val
    line1 = live_plotter(x_vec,y_vec,line1)
    y_vec = np.append(y_vec[1:],0.0)

"""


def plot_net(fig, plottedWeights, plottedBiases, weights, biases, net_name, feature_names, context_rep=False, context_width=None, pause_time=0.01):
    """
    weights = a list of numpy arrays. each array contains the weights of 1 layer of the network
    Plot a Torch network, with network connections color-coded by weight.
    """

    #step 1: figure out architecture.
    n_layers = len(weights)
    architecture = []

    for i in range(n_layers):
        architecture.append(np.shape(weights[i])[0])

    #last layer:
    architecture.append(np.shape(weights[-1])[1])
    architecture = np.array((architecture))

    depth = len(architecture)
    width = np.max(architecture) + 1 #including bias node

    # borrowed from corner.py
    factor = 2.         # size of one neuron (?)
    lbdim = 0.3 * factor   # size of left/bottom margin
    trdim = 0.3 * factor   # size of top/right margin
    
    whspace = 0.05         # w/hspace size
    ploth = (factor * depth + factor * (depth - 1.) * whspace) + lbdim + trdim
    plotw = (factor * width + factor * (width - 1.) * whspace) + lbdim + trdim

    if plottedWeights == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(2*plotw/9., 5*ploth/9.),dpi=200)
        
        # Format the figure.
        lb = lbdim / ploth
        tr = 1. - lb

        fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                            wspace=whspace, hspace=whspace)
        

        #step 2: draw the network

        #horizontal positions of layers
        layer_x = np.linspace(0.,1.,(n_layers + 1))
        #print(layer_x)
        neuron_ys = []
        neuronh = 1./width

        for i in range(n_layers+1):
            n_neurons = architecture[i] + 1 # including bias node

            """
            # uniform vertical spacing of neurons across layers
            if n_neurons % 2 == 0: # even number
                neuron0_y = 0.5 - (neuronh/2.) - ((n_neurons/2) - 1)*neuronh
            else:                  # odd number
                neuron0_y = 0.5 - ((n_neurons - 1)/2)*neuronh

            neuron_y = np.arange(neuron0_y, neuron0_y+(n_neurons*neuronh), neuronh)
            neuron_y = neuron_y[0:n_neurons]
            """

            # as much vertical space as possible between neurons
            neuron_y = np.linspace(0.,1.,n_neurons)
            neuron_ys.append(neuron_y)

            # nodes
            ax.plot(layer_x[i]*np.ones_like(neuron_y[0:-1]), neuron_y[0:-1], marker='o', ms=11, mfc='None',mec='k',ls='None')
            
            # bias node
            if i < n_layers:
                ax.plot(layer_x[i]*np.ones_like(neuron_y[-1]), neuron_y[-1], marker='o', ms=11, mfc='#CED3D9',mec='k',ls='None')
            
            # feature names
            if i == 0:
                if context_rep is False: # i.e., if this is a single-planet representation
                    for j in range(len(neuron_y[0:-1])):
                        ax.text(x=-0.07,y=neuron_y[j]-neuronh/3.,s=feature_names[j])
                    ax.text(x=-0.09,y=neuron_y[-1]-neuronh/3.,s='bias')
                    ax.set_xlim(-0.05,1.05)
                    ax.set_ylim(-0.05,1.05)

                elif context_rep is True:
                    nUniqueFeatures = int(len(neuron_y[0:-1])/(2*context_width))
                    
                    contextPlanetLabels = []
                    for l in range(-1*context_width,0):
                        contextPlanetLabels.append(str(l))
                    for l in range(1,context_width+1):
                        contextPlanetLabels.append("+"+str(l))

                    for k in range(nUniqueFeatures):
                        for j in np.arange(k,len(neuron_y[0:-1]),nUniqueFeatures):
                            ax.text(x=-0.55,y=neuron_y[j]-neuronh/3.,s="context planet {0}, {1}".format(contextPlanetLabels[int(np.floor(j/nUniqueFeatures))], feature_names[k]))

                    ax.text(x=-0.15,y=neuron_y[-1]-neuronh/3.,s='bias')

                    ax.set_xlim(-0.55,1.05)
                    ax.set_ylim(-0.05,1.05)
        #ax.axhline(0.5)
        
        #colormap details
        cmin = -5.
        cmax = 5.
        """
        for i in range(n_layers):
            ws = weights[i]
            bs = biases[i]
            if np.min(ws) < cmin:
                cmin = np.min(ws)
            if np.max(ws) > cmax:
                cmax = np.max(ws)

            if np.min(bs) < cmin:
                cmin = np.min(bs)
            if np.max(bs) > cmax:
                cmax = np.max(bs)
        """
        cm = plt.get_cmap('Spectral') 
        cNorm  = colors.Normalize(vmin=cmin, vmax=cmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        #step 3: colormap connections by weights
        plottedWeights = []
        plottedBiases = []

        for i in range(n_layers):
            ws = weights[i]
            bs = biases[i]

            xStart = layer_x[i]
            xEnd = layer_x[i+1]

            neuron_y = neuron_ys[i]
            neuron_yp1 = neuron_ys[i+1]

            plottedWeights_thisLayer = []
            plottedBiases_thisLayer = []
            
            #regular nodes
            #print(np.shape(ws))
            #print(int(np.shape(ws)[0]))
            #print(int(np.shape(ws)[1]))

            for ii in range(0,int(np.shape(ws)[0])):
                plottedWeights_thisLayer_thisNeuron = []
                for jj in range(0,int(np.shape(ws)[1])):
                    w = ws[ii,jj]
                    colorVal = scalarMap.to_rgba(w)

                    yStart = neuron_y[ii]
                    yEnd = neuron_yp1[jj]

                    pN = ax.plot(np.array((xStart,xEnd)),np.array((yStart,yEnd)), color=colorVal, ls='-',marker='None',lw=0.5)
                    plottedWeights_thisLayer_thisNeuron.append(pN[0])
                plottedWeights_thisLayer.append(plottedWeights_thisLayer_thisNeuron)
            
            #bias nodes
            for ii in range(0,len(bs)):
                b = bs[ii]
                colorVal = scalarMap.to_rgba(b)

                yStart = neuron_y[-1]
                yEnd = neuron_yp1[ii]

                pB = ax.plot(np.array((xStart,xEnd)),np.array((yStart,yEnd)), color=colorVal, ls='-',marker='None',lw=0.5)
                plottedBiases_thisLayer.append(pB[0])

            plottedWeights.append(plottedWeights_thisLayer)
            plottedBiases.append(plottedBiases_thisLayer)

        scalarMap.set_array(np.linspace(cmin,cmax,100))
        cb = fig.colorbar(scalarMap, ax=ax, label="weight")
        ax.axis("off")
        ax.set_title(net_name)
        plt.show()

    else:
        cmin = -5
        cmax = 5.
        """
        for i in range(n_layers):
            ws = weights[i]
            bs = biases[i]
            if np.min(ws) < cmin:
                cmin = np.min(ws)
            if np.max(ws) > cmax:
                cmax = np.max(ws)

            if np.min(bs) < cmin:
                cmin = np.min(bs)
            if np.max(bs) > cmax:
                cmax = np.max(bs)
        """
        cm = plt.get_cmap('Spectral') 
        cNorm  = colors.Normalize(vmin=cmin, vmax=cmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        for i in range(n_layers):
            ws = weights[i]
            bs = biases[i]

            #regular nodes
            for ii in range(0,int(np.shape(ws)[0])):
                for jj in range(0,int(np.shape(ws)[1])):
                    w = ws[ii,jj]
                    colorVal = scalarMap.to_rgba(w)
                    plottedWeights[i][ii][jj].set_color(colorVal)
            
            #bias nodes
            for ii in range(0,len(bs)):
                b = bs[ii]
                colorVal = scalarMap.to_rgba(b)
                plottedBiases[i][ii].set_color(colorVal)

        
        scalarMap.set_array(np.linspace(cmin,cmax,100))
        #cb = fig.colorbar(scalarMap, ax=ax, label="weight")
        
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)
    
    return fig, plottedWeights, plottedBiases

