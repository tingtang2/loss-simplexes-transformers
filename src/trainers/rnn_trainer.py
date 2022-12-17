import logging
import random
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from tqdm import trange

import utils
from models.model import Seq2SeqLSTM
from trainers.base_trainer import BaseTrainer
from torchtext.datasets import Multi30k


class RNNTrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super(RNNTrainer, self).__init__(**kwargs)

        self.model = Seq2SeqLSTM(src_vocab_size=self.src_vocab_size,
                                 tgt_vocab_size=self.tgt_vocab_size,
                                 embed_size=self.embed_size,
                                 hidden_size=self.hidden_size,
                                 dropout_prob=self.dropout_prob,
                                 device=self.device,
                                 n_layers=self.n_layers).to(self.device)

        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate)

        self.name = 'seq2seq_vanilla_lstms'

    def run_experiment(self):
        train_loader, val_loader = self.create_dataloaders()

        # training loop
        for epoch in trange(1, self.epochs + 1):
            start_time = timer()
            train_loss = self.train_epoch(train_loader)
            end_time = timer()

            val_loss, val_bleu = self.eval_epoch(val_loader)
            logging.info((
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Val BLEU score: {val_bleu:.3f} "
                f"Epoch time = {(end_time - start_time):.3f}s"))

            example_sentence_tokenized = utils.text_transform[utils.src_lang](
                self.test_sentence.rstrip("\n"))
            with torch.no_grad():
                output_tokens, _ = self.rnn_translate(
                    example_sentence_tokenized.to(self.device), None)

            translated_sentence = " ".join(output_tokens).replace(
                "<bos>", "").replace("<eos>", "")

            logging.info(
                f'test sentence: {self.test_sentence}, translated sentence: {translated_sentence}'
            )

        self.save_model(self.name)

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss = 0

        for src, tgt in loader:
            self.optimizer.zero_grad()
            src = src.transpose(-1, -2).to(self.device)
            tgt = tgt.transpose(-1, -2).to(self.device)

            decoder_output, decoder_hidden, decoder_cell = self.model(src, tgt)

            output_for_loss = decoder_output[:, 1:].reshape(
                -1, decoder_output.size(-1))
            tgt_for_loss = tgt[:, 1:].reshape(-1)

            loss = self.criterion(output_for_loss, tgt_for_loss)

            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.grad_clip)

            self.optimizer.step()
            running_loss += loss.item()

        return running_loss

    def eval_epoch(self, loader: DataLoader):
        self.model.eval()

        running_loss = 0.0

        random_iteration = random.randint(0, 7)

        with torch.no_grad():
            for i, (src, tgt) in enumerate(loader):
                src = src.transpose(-1, -2).to(self.device)
                tgt = tgt.transpose(-1, -2).to(self.device)

                decoder_output, decoder_hidden, decoder_cell = self.model(
                    src, tgt, teacher_forcing_ratio=0)

                output_for_loss = decoder_output[:, 1:].reshape(
                    -1, decoder_output.size(-1))
                tgt_for_loss = tgt[:, 1:].reshape(-1)
                loss = self.criterion(output_for_loss, tgt_for_loss)

                running_loss += loss.item()

                if i == random_iteration:
                    self.evaluate_randomly(src_tokens=src[0, :],
                                           tgt_tokens=tgt[0, :])

        return running_loss

    def rnn_translate(self, input_tensor, tgt_tensor, use_attention=False):
        self.model.eval()
        with torch.no_grad():
            input_length = input_tensor.size(0)

            encoder_outputs, encoder_hidden, cell = self.model.encoder(
                input_tensor.reshape((1, -1)))

            decoder_input = torch.tensor([[utils.BOS_IDX]],
                                         device=self.device)  #BOS

            decoder_hidden = encoder_hidden

            decoded_tokens = []
            decoder_attentions = torch.zeros(utils.MAX_LENGTH,
                                             utils.MAX_LENGTH)

            for di in range(utils.MAX_LENGTH):
                if use_attention:
                    decoder_output, decoder_hidden, cell, decoder_attention = self.model.decoder(
                        decoder_input, decoder_hidden, cell, encoder_outputs)

                    decoder_attentions[di] = decoder_attention.data
                else:
                    decoder_output, decoder_hidden, cell = self.model.decoder(
                        decoder_input.reshape((1, -1)), decoder_hidden, cell)

                top_pred_token = decoder_output.argmax(-1)
                decoded_tokens.append(top_pred_token.item())
                if top_pred_token.item() == utils.EOS_IDX:
                    break

                decoder_input = top_pred_token.squeeze().detach()

            decoded_words = self.vocab_transform[utils.tgt_lang].lookup_tokens(
                decoded_tokens)

            return decoded_words[:-1], decoder_attentions[:di + 1]

    def evaluate_randomly(self, src_tokens, tgt_tokens):
        src_words = self.vocab_transform[utils.src_lang].lookup_tokens(
            list(src_tokens.detach().cpu().numpy()))
        tgt_words = self.vocab_transform[utils.tgt_lang].lookup_tokens(
            list(tgt_tokens.detach().cpu().numpy()))

        logging.info(f'> {" ".join(src_words)}')
        logging.info(f'= {" ".join(tgt_words)}')
        output_words, attentions = self.rnn_translate(input_tensor=src_tokens,
                                                      tgt_tensor=None)
        output_sentence = ' '.join(output_words)
        logging.info(f'< {output_sentence}')

    def calc_test_bleu(self):
        test_data = Multi30k(split='test',
                             language_pair=(utils.src_lang, utils.tgt_lang))
        self.model.load_state_dict(
            torch.load(f'{self.save_dir}models/{self.name}.pt'))

        loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=self.batch_size,
                                             collate_fn=utils.collate_fn,
                                             shuffle=False)

        self.model.eval()
        tgts = []
        pred_tgts = []
        with torch.no_grad():
            for src, tgt, in loader:
                src = src.transpose(-1, -2).to(self.device)
                tgt = tgt.transpose(-1, -2).to(self.device)

                for sentence in range(src.size(0)):
                    pred_tgt, _ = self.rnn_translate(src[sentence],
                                                     tgt[sentence])
                    pred_tgts.append(pred_tgt)

                # hack to remove <eos> <pad>
                tgt_words = [[[
                    element for element in ' '.join(self.vocab_transform[
                        utils.tgt_lang].lookup_tokens(
                            list(tgt[example, 1:].cpu().numpy()))).replace(
                                '<pad>', '').replace('<eos>', '').split(' ')
                    if element != ''
                ]] for example in range(tgt.size(0))]

                tgts += tgt_words
        return bleu_score(pred_tgts, tgts)


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
