from __future__ import division, print_function
import argparse
import random
import os
import sys
import torch

from control import Control
from data import Data
from logger import Logger
from model import MMIModel


def main(args):
    feature_names = args.feature_names
    feature_names = map(str, feature_names.strip('[]').split(','))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')
    data = Data(args.num_planet_features, args.num_stellar_features, args.data, args.truth_known)

    model = MMIModel(args.num_planet_features, args.num_stellar_features, args.num_labels, args.width, args.dropout_prob, feature_names, args.plotting).to(device)
    logger = Logger(args.model + '.log', args.train)
    logger.log('python ' + ' '.join(sys.argv) + '\n')
    logger.log('Random seed: %d' % args.seed)
    control = Control(model, args.model, args.batch_size, device, logger, args.truth_known)

    if args.train:
        if os.path.exists(args.model):
            control.load_model(args.lr)
        print("Resuming...")
        control.train(data, args.data, args.lr, args.epochs)

    elif os.path.exists(args.model):
        control.load_model(args.lr)
        control.classify(args.data, data, args.num_labels)
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Maximal Mutual Information (MMI) Tagger')

    parser.add_argument('model', type=str,
                        help='model path')
    parser.add_argument('data', type=str,
                        help='data path (X.words, assumes X.tags exists)')
    parser.add_argument('--num_labels', type=int, default=45,
                        help='number of labels to induce [%(default)d]')
    parser.add_argument('--num_planet_features', type=int, default=2,
                        help='number of features known about each planet [%(default)d]')
    parser.add_argument('--num_stellar_features', type=int, default=3,
                        help='number of features known about the star/the system overall [%(default)d]')
    parser.add_argument('--feature_names', type=str,
                        help='names of features')
    parser.add_argument('--dropout_prob', type=float, default=0.05,
                        help='dropout probability in network layers [%(default)g]')
    parser.add_argument('--train', action='store_true',
                        help='train?')
    parser.add_argument('--batch_size', type=int, default=10, metavar='B',
                        help='batch size [%(default)d]')
    parser.add_argument('--dim', type=int, default=200,
                        help='dimension of word embeddings [%(default)d]')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='initial learning rate [%(default)g]')
    parser.add_argument('--width', type=int, default=1,
                        help='context width (to each side) [%(default)d]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='max number of epochs [%(default)d]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA?')
    parser.add_argument('--truth_known', action='store_true',
                        help='truth known?')
    parser.add_argument('--plotting', action='store_true',
                        help='plot?')

    args = parser.parse_args()
    main(args)
