from __future__ import division, print_function
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Computer Modern Roman']

def plot_net(fig, cb, plottedWeights, plottedBiases, weights, biases, net_name, feature_names, context_rep=False, context_width=None, pause_time=0.01):
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
    #print(architecture)
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
        fig, ax = plt.subplots(1, 1, figsize=(2*plotw/9., 7*ploth/9.),dpi=150)
        
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
                    ax.text(x=-0.09,y=neuron_y[-1]-neuronh/3.,s='bias',fontsize=8)
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
                            ax.text(x=-0.55,y=neuron_y[j]-neuronh/3.,s="context planet {0}, {1}".format(contextPlanetLabels[int(np.floor(j/nUniqueFeatures))], feature_names[k]),fontsize=8)

                    ax.text(x=-0.15,y=neuron_y[-1]-neuronh/3.,s='bias',fontsize=8)

                    ax.set_xlim(-0.55,1.05)
                    ax.set_ylim(-0.05,1.05)
        #ax.axhline(0.5)
        
        #colormap details
        cmin = 5.
        cmax = -5.
        
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

                pB = ax.plot(np.array((xStart,xEnd)),np.array((yStart,yEnd)), color=colorVal, ls='-',marker='None',lw=0.5,alpha=1)
                plottedBiases_thisLayer.append(pB[0])

            plottedWeights.append(plottedWeights_thisLayer)
            plottedBiases.append(plottedBiases_thisLayer)

        scalarMap.set_array(np.linspace(cmin,cmax,100))
        cb = fig.colorbar(scalarMap, ax=ax, label="weight")
        ax.axis("off")
        ax.set_title(net_name)
        plt.show()

    else:
        cmin = 5.
        cmax = -5.
        
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
        cb.set_clim(cmin,cmax)
        cb.draw_all()
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)
    
    return fig, cb, plottedWeights, plottedBiases

