from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMIModel(nn.Module):

    def __init__(self, num_labels, width):
        super(MMIModel, self).__init__()
        self.num_labels = num_labels
        self.width = width

        self.loss = Loss()

    def forward(self, past_planets, future_planets, is_training=True):
        if is_training:
            loss = self.loss(past_planets, future_planets)
            return loss

        else:
            future_probs, future_indices = future_planets.max(1)
            return future_probs, future_indices


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.entropy = Entropy()

    def forward(self, past_planets, future_planets):
        pZ_Y = F.softmax(future_planets, dim=1)
        pZ = pZ_Y.mean(0)
        hZ = self.entropy(pZ)

        x = pZ_Y * F.log_softmax(past_planets, dim=1)  # B x m
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
