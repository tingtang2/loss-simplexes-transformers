import logging
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from models.model import Seq2SeqLSTM
from trainers.base_trainer import BaseTrainer

import utils
from torchtext.data.metrics import bleu_score


class RNNTrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super(RNNTrainer, self).__init__(**kwargs)

        self.model = Seq2SeqLSTM(src_vocab_size=self.src_vocab_size,
                                 tgt_vocab_size=self.tgt_vocab_size,
                                 embed_size=self.embed_size,
                                 hidden_size=self.hidden_size)

        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate)

    def run_experiment(self):
        train_loader, val_loader = self.create_dataloaders()

        # training loop
        for epoch in trange(1, self.epochs + 1):
            start_time = timer()
            train_loss = self.train_epoch(train_loader)
            end_time = timer()

            val_loss, val_bleu = self.evaluate_rnn(val_loader)
            logging.info((
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Val BLEU score: {val_bleu}"
                f"Epoch time = {(end_time - start_time):.3f}s"))

    def train_epoch(self, loader: DataLoader):
        running_loss = 0

        for src, tgt in loader:
            self.optimizer.zero_grad()
            src = src.transpose(-1, -2).to(self.device)
            tgt = tgt.transpose(-1, -2).to(self.device)

            decoder_output, decoder_hidden, decoder_cell = self.model(
                src, tgt[:, :-1])

            loss = self.criterion(
                decoder_output.reshape(-1, decoder_output.size(-1)),
                tgt[:, 1:].reshape(-1))

            loss.backward()

            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / (len(loader) * self.batch_size)

    def eval_epoch(self, loader: DataLoader):
        self.model.eval()

        running_loss = 0

        tgts = []
        pred_tgts = []

        with torch.no_grad():
            for src, tgt in loader:
                src = src.transpose(-1, -2).to(self.device)
                tgt = tgt.transpose(-1, -2).to(self.device)

                decoder_output, decoder_hidden, decoder_cell = self.model(
                    src, tgt[:, :-1])

                loss = self.criterion(
                    decoder_output.reshape(-1, decoder_output.size(-1)),
                    tgt[:, 1:].reshape(-1))

                pred_trg, _ = self.rnn_translate(src)

                running_loss += loss.item()
                pred_tgts.append(pred_trg)
                tgt_words = [
                    " ".join(
                        self.vocab_transform[utils.tgt_lang].lookup_tokens(
                            list(tgt[example].cpu().numpy()))).replace(
                                "<bos>", "").replace("<eos>", "")
                    for example in range(tgt.size(0))
                ]

                tgts.append([tgt_words])

        return running_loss / (len(loader) * self.batch_size), bleu_score(
            pred_tgts, tgts)

    def rnn_translate(self, input_tensor, use_attention=False):
        with torch.no_grad():
            input_length = input_tensor.size()[0]

            encoder_outputs = torch.zeros(utils.MAX_LENGTH,
                                          self.model.encoder.hidden_size,
                                          device=self.device)

            # print(input_tensor.size())
            for ei in range(input_length):
                encoder_output, encoder_hidden, cell = self.encoder(
                    input_tensor[ei].reshape((1, -1)))
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[utils.BOS_IDX]],
                                         device=self.device)  #BOS

            decoder_hidden = encoder_hidden

            decoded_tokens = []
            decoder_attentions = torch.zeros(utils.MAX_LENGTH, utils.MAX_LENTH)

            for di in range(utils.MAX_LENGTH):
                if use_attention:
                    decoder_output, decoder_hidden, cell, decoder_attention = self.model.decoder(
                        decoder_input, decoder_hidden, cell, encoder_outputs)

                    decoder_attentions[di] = decoder_attention.data
                else:
                    decoder_output, decoder_hidden, cell = self.model.decoder(
                        decoder_input.reshape((1, -1)), decoder_hidden, cell)
                topv, topi = decoder_output.data.topk(1)
                decoded_tokens.append(topi.item())
                if topi.item() == utils.EOS_IDX:
                    break

                decoder_input = topi.squeeze().detach()
            decoded_words = " ".join(self.vocab_transform[
                utils.tgt_lang].lookup_tokens(decoded_tokens)).replace(
                    "<bos>", "").replace("<eos>", "")
            print('decoded_words', decoded_words)

            return decoded_words, decoder_attentions[:di + 1]


# subspace functions
##################################################################################################

# def get_weight(m, i):
#     if i == 0:
#         return m.weight
#     return getattr(m, f'weight_{i}')

# def train_epoch_rnn_subspace(dataset, model, optimizer, criterion, device,
#                              batch_size, beta, **kwargs):
#     model.train()
#     running_loss = 0

#     train_dataloader = DataLoader(dataset,
#                                   batch_size=batch_size,
#                                   collate_fn=collate_fn)

#     for src, tgt in train_dataloader:
#         src = src.transpose(-1, -2).to(device)
#         tgt = tgt.transpose(-1, -2).to(device)

#         alpha = torch.rand(1, device=device)
#         for m in model.modules():
#             if isinstance(m, nn.Linear) or isinstance(
#                     m, nn.LSTM) or isinstance(m, nn.Embedding):
#                 setattr(m, f'alpha', alpha)

#         optimizer.zero_grad()
#         decoder_output = model(src, tgt[:, :-1])
#         loss = criterion(decoder_output.reshape(-1, decoder_output.size(-1)),
#                          tgt[:, 1:].reshape(-1))

#         # regularization
#         num = 0.0
#         norm = 0.0
#         norm1 = 0.0
#         for m in model.modules():
#             if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
#                 vi = get_weight(m, 0)
#                 vj = get_weight(m, 1)
#                 num += (vi * vj).sum()
#                 norm += vi.pow(2).sum()
#                 norm1 += vj.pow(2).sum()
#             if isinstance(m, nn.LSTM):
#                 w_ih_i = getattr(m, 'weight_ih_l0')
#                 w_ih_j = getattr(m, 'weight_ih_l0_1')

#                 num += (w_ih_i * w_ih_j).sum()
#                 norm += w_ih_i.pow(2).sum()
#                 norm1 += w_ih_j.pow(2).sum()

#                 w_hh_i = getattr(m, 'weight_hh_l0')
#                 w_hh_j = getattr(m, 'weight_hh_l0_1')

#                 num += (w_hh_i * w_hh_j).sum()
#                 norm += w_hh_i.pow(2).sum()
#                 norm1 += w_hh_j.pow(2).sum()

#         loss += beta * (num.pow(2) / (norm * norm1))

#         loss.backward()

#         optimizer.step()
#         running_loss += loss.item()

#     return running_loss / 29000

# def evaluate_rnn_subspace(dataset, model, criterion, device, batch_size,
#                           **kwargs):
#     model.eval()
#     running_losses = [0, 0, 0]

#     eval_dataloader = DataLoader(dataset,
#                                  batch_size=batch_size,
#                                  collate_fn=collate_fn)
#     alphas = [0.0, 0.5, 1.0]
#     tgts = []
#     pred_tgts = []
#     for i, alpha in enumerate(alphas):
#         for m in model.modules():
#             if isinstance(m, nn.Linear) or isinstance(
#                     m, nn.LSTM) or isinstance(m, nn.Embedding):
#                 setattr(m, f'alpha', alpha)

#         with torch.no_grad():
#             for src, tgt in eval_dataloader:
#                 src = src.transpose(-1, -2).to(device)
#                 tgt = tgt.transpose(-1, -2).to(device)

#                 decoder_output = model(src, tgt[:, :-1])
#                 loss = criterion(
#                     decoder_output.reshape(-1, decoder_output.size(-1)),
#                     tgt[:, 1:].reshape(-1))

#                 running_losses[i] += loss.item()
#                 pred_trg = pred_trg[:-1]

#                 pred_tgts.append(pred_trg)
#                 tgts.append([tgt])

#     return [loss / 1014
#             for loss in running_losses], bleu_score(pred_tgts, tgts)

# def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):

#     trgs = []
#     pred_trgs = []

#     for datum in data:

#         src = vars(datum)['src']
#         trg = vars(datum)['trg']

#         pred_trg, _ = translate_sentence(src, src_field, trg_field, model,
#                                          device, max_len)

#         #cut off  token
#         pred_trg = pred_trg[:-1]

#         pred_trgs.append(pred_trg)
#         trgs.append([trg])

#     return bleu_score(pred_trgs, trgs)
