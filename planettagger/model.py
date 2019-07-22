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

        self.planetContext = ContextHandler(width, num_labels)
        self.indivPlanet = PlanetHandler(num_labels)

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

class ContextHandler(nn.Module):

    def __init__(self, width, num_labels):
        super(ContextHandler, self).__init__()
        self.linear = nn.Linear(2 * width, num_labels) #(in_features, out_features) = (numContextWords, numLabels)

    def forward(self, contextPlanets):
        #contextPlanets will be of the shape: Batchsize x 2width x numPlanetFeatures
        rep = self.linear(contextPlanets.view(contextPlanets.shape[0], -1))  # self.linear(Batchsize x 2width x numPlanetFeatures)
                                                                             # returns Batchsize x numLabels
        return rep


class PlanetHandler(nn.Module):

    def __init__(self, wemb, cemb, num_layers, num_labels):
        super(PlanetHandler, self).__init__()
        self.linear = nn.Linear(wemb.embedding_dim + 2 * cemb.embedding_dim,
                                num_labels)

    def forward(self, words, padded_chars, char_lengths):
        B = len(char_lengths)
        wembs = self.wemb(words)  # B x d_w

        packed = pack_padded_sequence(self.cemb(padded_chars), char_lengths)
        output, (final_h, final_c) = self.lstm(packed)

        final_h = final_h.view(self.lstm.num_layers, 2, B,
                               self.lstm.hidden_size)[-1]         # 2 x B x d_c
        cembs = final_h.transpose(0, 1).contiguous().view(B, -1)  # B x 2d_c

        rep = self.linear(torch.cat([wembs, cembs], 1))  # B x m
        return rep

