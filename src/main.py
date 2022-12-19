import argparse
import logging
import random
import sys
from datetime import date

import torch
from torch import nn
from torch.optim import Adam, AdamW

import utils
from trainers.rnn_trainer import RNNTrainer, SubspaceRNNTrainer, AttentionRNNTrainer, SubspaceAttentionRNNTrainer

arg_trainer_map = {
    'rnn': RNNTrainer,
    'subspace_rnn': SubspaceRNNTrainer,
    'attention_rnn': AttentionRNNTrainer,
    'subspace_attention_rnn': SubspaceAttentionRNNTrainer
}
arg_optimizer_map = {'adamw': AdamW, 'adam': Adam}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
        'Create and run loss subspace experiments for neural machine translation'
    )

    parser.add_argument('--epochs',
                        default=20,
                        type=int,
                        help='number of epochs to train model')
    parser.add_argument('--device',
                        '-d',
                        default='cuda',
                        type=str,
                        help='cpu or gpu ID to use')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='mini-batch size used to train model')
    parser.add_argument('--dropout_prob',
                        default=0.5,
                        type=float,
                        help='probability for dropout layers')
    parser.add_argument('--save_dir',
                        default='/home/tingchen/learning_subspace_save/',
                        help='path to saved model files')
    parser.add_argument('--data_dir',
                        default='/home/tingchen/data/',
                        help='path to data files')
    parser.add_argument('--optimizer',
                        default='adam',
                        help='type of optimizer to use')
    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float,
                        help='learning rate for optimizer')
    parser.add_argument('--model_type',
                        default='rnn',
                        help='type of model to use')
    parser.add_argument('--num_repeats',
                        default=3,
                        type=int,
                        help='number of times to repeat experiment')
    parser.add_argument('--seed',
                        default=11202022,
                        type=int,
                        help='random seed to be used in numpy and torch')
    parser.add_argument('--embed_size',
                        default=256,
                        type=int,
                        help='dimensionality of token embeddings')
    parser.add_argument('--hidden_size',
                        default=512,
                        type=int,
                        help='dimensionality of hidden layers')
    parser.add_argument('--beta',
                        default=1.0,
                        type=float,
                        help='constant for learning subspaces')
    parser.add_argument('--grad_clip',
                        default=-1.0,
                        type=float,
                        help='max norm of gradients to clip')
    parser.add_argument('--n_layers',
                        default=3,
                        type=int,
                        help='number of lstm layers')
    parser.add_argument(
        '--val_midpoint_only',
        action='store_true',
        help=
        'only collect validation metrics for the midpoint of the line (for speed)'
    )

    parser.add_argument('--debug',
                        action='store_true',
                        help='move stuff to cpu for better tracebacks')

    args = parser.parse_args()
    configs = args.__dict__

    # for repeatability
    torch.manual_seed(configs['seed'])
    random.seed(configs['seed'])

    # set up logging
    filename = f'{configs["model_type"]}-{date.today()}'
    FORMAT = '%(asctime)s;%(levelname)s;%(message)s'
    logging.basicConfig(level=logging.INFO,
                        filename=f'{configs["save_dir"]}logs/{filename}.log',
                        filemode='a',
                        format=FORMAT)
    logging.info(configs)

    # get trainer
    trainer_type = arg_trainer_map[configs['model_type']]
    trainer = trainer_type(
        optimizer_type=arg_optimizer_map[configs['optimizer']],
        criterion=nn.CrossEntropyLoss(ignore_index=utils.PAD_IDX),
        **configs)

    # perform experiment n times
    #for iter in range(configs['num_repeats']):
    # trainer.run_experiment()
    # test_bleu = trainer.calc_test_bleu()
    test_bleu = trainer.ensemble_calc_test_bleu()
    print(f'ensemble test bleu score: {100 * test_bleu:.2f}')
    logging.info(f'ensemble test bleu score: {test_bleu * 100:.2f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())