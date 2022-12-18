import argparse
import logging
import random
import sys
from datetime import date

import torch
from torch import nn
from torch.optim import Adam, AdamW

import utils
from trainers.rnn_trainer import RNNTrainer, SubspaceRNNTrainer

arg_trainer_map = {'rnn': RNNTrainer, 'subspace_rnn': SubspaceRNNTrainer}
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
    trainer.run_experiment()
    test_bleu = trainer.calc_test_bleu()
    print(f'test bleu score: {100 * test_bleu:.2f}')
    logging.info(f'test bleu score: {test_bleu * 100:.2f}')

    # EMB_SIZE = 256
    # NHEAD = 4
    # FFN_HID_DIM = 512
    # BATCH_SIZE = 32
    # NUM_ENCODER_LAYERS = 3
    # NUM_DECODER_LAYERS = 3
    # NUM_EPOCHS = 18

    # if configs['model_type'] == 'rnn_subspace':
    #     model = Seq2SeqLSTMSubspace(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE,
    #                                 EMB_SIZE).to(device)
    #     for m in model.modules():
    #         if hasattr(m, 'initialize'):
    #             m.initialize(self.seed)

    #     criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, amsgrad=True)

    #     # training loop
    #     for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
    #         start_time = timer()
    #         train_loss = train_epoch_rnn_subspace(train_data, model, optimizer,
    #                                               criterion, device,
    #                                               BATCH_SIZE, **configs)
    #         end_time = timer()
    #         val_losses = evaluate_rnn_subspace(val_data, model, criterion,
    #                                            device, BATCH_SIZE)
    #         print(
    #             f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss alpha 0: {val_losses[0]:.3f}, Val loss alpha 0.5: {val_losses[1]:.3f}, Val loss alpha 1: {val_losses[2]:.3f}, "
    #             f"Epoch time = {(end_time - start_time):.3f}s")

    #     # save model
    #     save_path = f'{configs["save_dir"]}subspace_rnn.pt'
    #     torch.save(model.state_dict(), save_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())