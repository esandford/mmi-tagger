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
    #get arguments as bools
    argsToCheck = [args.train, args.truth_known, args.plot, args.saveplot, args.cross_validate]
    
    if (args.train).lower() == 'true':
        TRAIN = True
    else:
        TRAIN = False

    if (args.truth_known).lower() == 'true':
        TRUTH_KNOWN = True
    else:
        TRUTH_KNOWN = False

    if (args.plot).lower() == 'true':
        PLOT = True
    else:
        PLOT = False

    if (args.saveplot).lower() == 'true':
        SAVEPLOT = True
    else:
        SAVEPLOT = False

    if (args.cross_validate).lower() == 'true':
        CROSS_VALIDATE = True
    else:
        CROSS_VALIDATE = False

    #get arguments as list
    feature_names = args.feature_names
    feature_names = list(map(str, feature_names.strip('[]').split(',')))
    
    #initiate network
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    #get data in correct form, using Data class
    data = Data(args.num_planet_features, args.num_stellar_features, args.data, TRUTH_KNOWN)

    #create model, using MMIModel class
    model = MMIModel(args.num_planet_features, args.num_stellar_features, args.num_labels, args.width, args.dropout_prob, feature_names, PLOT, SAVEPLOT).to(device)
    
    #create logger, using Logger class
    logger = Logger(args.model + '.log', TRAIN)
    logger.log('python ' + ' '.join(sys.argv) + '\n')
    logger.log('Random seed: %d' % args.seed)
    
    #create control, using Control class
    control = Control(model, args.model, args.batch_size, device, logger, TRUTH_KNOWN, args.seed)

    
    if TRAIN is True:
        if os.path.exists(args.model):
            control.load_model(args.lr)
        print("Resuming...")
        control.train(data, args.data, args.lr, args.epochs)

    elif CROSS_VALIDATE is True:
        control.load_model(args.lr)
        CVdata = Data(args.num_planet_features, args.num_stellar_features, args.CVdata, TRUTH_KNOWN)
        control.cross_validate(args.CVdata, CVdata)

    elif os.path.exists(args.model):
        control.load_model(args.lr)
        control.classify(args.data, data, args.num_labels)
       

#main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Maximal Mutual Information (MMI) Plaent Classifier')

    parser.add_argument('model', type=str,
                        help='model path')
    parser.add_argument('data', type=str,
                        help='data path')
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
    parser.add_argument('--train', type=str,
                        help='train?')
    parser.add_argument('--truth_known', type=str,
                        help='truth known?')
    parser.add_argument('--plot', type=str, default="False",
                        help='live plot weights?')
    parser.add_argument('--saveplot', type=str, default="False",
                        help='save plot of final weights?')
    parser.add_argument('--cross_validate', type=str,
                        help='cross-validate the model on a holdout test set?')
    parser.add_argument('--CVdata', type=str,
                        help='holdout test set data path')
    args = parser.parse_args()
    main(args)
