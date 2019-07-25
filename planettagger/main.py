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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')
    data = Data(args.num_planet_features, args.data)

    model = MMIModel(args.num_planet_features, args.num_labels, args.width).to(device)
    logger = Logger(args.model + '.log', args.train)
    logger.log('python ' + ' '.join(sys.argv) + '\n')
    logger.log('Random seed: %d' % args.seed)
    control = Control(model, args.model, args.batch_size, device, logger)

    if args.train:
        control.train(data, args.lr, args.epochs)

    elif os.path.exists(args.model):
        control.load_model()
        control.classify(args.data, data)
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Maximal Mutual Information (MMI) Tagger')

    parser.add_argument('model', type=str,
                        help='model path')
    parser.add_argument('data', type=str,
                        help='data path (X.words, assumes X.tags exists)')
    parser.add_argument('--num_labels', type=int, default=45, metavar='m',
                        help='number of labels to induce [%(default)d]')
    parser.add_argument('--num_planet_features', type=int, default=5, metavar='m',
                        help='number of features known about each planet [%(default)d]')
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

    args = parser.parse_args()
    main(args)
