import logging
import random
from timeit import default_timer as timer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from torchtext.datasets import Multi30k
from tqdm import trange

import utils
from models.model import Seq2SeqLSTM, AttentionSeq2SeqLSTM
from models.subspace_models import Seq2SeqLSTMSubspace, AttentionSeq2SeqLSTMSubspace
from trainers.base_trainer import BaseTrainer

from tqdm import tqdm


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

        self.early_stopping_threshold = 10

    def run_experiment(self):
        train_loader, val_loader = self.create_dataloaders()

        # training loop
        cos_sims = []
        l2s = []
        best_val_loss = 1e+5
        early_stopping_counter = 0
        for epoch in trange(1, self.epochs + 1):
            start_time = timer()
            train_loss = self.train_epoch(train_loader)
            end_time = timer()

            if 'subspace' in self.name:
                val_loss, cos_sim, l2 = self.eval_epoch(val_loader)
                l2s.append(l2)
                cos_sims.append(cos_sim)
                logging.info(
                    f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss alpha 0: {val_loss[0]:.3f}, Val loss alpha 0.5: {val_loss[1]:.3f}, Val loss alpha 1: {val_loss[2]:.3f}, "
                    f"Epoch time = {(end_time - start_time):.3f}s, cos sim: {cos_sim}, l2: {l2}"
                )
                if self.val_midpoint_only:
                    val_loss = val_loss[0]
                else:
                    val_loss = val_loss[1]

            else:
                val_loss = self.eval_epoch(val_loader)
                logging.info((
                    f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f} "
                    f"Epoch time = {(end_time - start_time):.3f}s"))

            if val_loss < best_val_loss:
                self.save_model(self.name)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                break

        if 'subspace' in self.name:
            self.save_metrics(cos_sims, name=f'{self.name}_cossims')
            self.save_metrics(l2s, name=f'{self.name}_l2s')

    def run_test_sentence(self):
        example_sentence_tokenized = utils.text_transform[utils.src_lang](
            self.test_sentence.rstrip("\n"))
        with torch.no_grad():
            output_tokens, _ = self.rnn_translate(
                example_sentence_tokenized.to(self.device), None)

        translated_sentence = " ".join(output_tokens).replace("<bos>",
                                                              "").replace(
                                                                  "<eos>", "")

        logging.info(
            f'test sentence: {self.test_sentence}, translated sentence: {translated_sentence}'
        )

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

            if 'subspace' in self.name:
                encoder_outputs, encoder_hidden, cell = self.model.encode(
                    input_tensor.reshape((1, -1)))
            else:
                encoder_outputs, encoder_hidden, cell = self.model.encoder(
                    input_tensor.reshape((1, -1)))

            decoder_input = torch.tensor([[utils.BOS_IDX]],
                                         device=self.device)  #BOS

            decoder_hidden = encoder_hidden

            decoded_tokens = []
            decoder_attentions = torch.zeros(utils.MAX_LENGTH,
                                             utils.MAX_LENGTH)

            for di in range(utils.MAX_LENGTH):
                if 'subspace' in self.name:
                    if use_attention:
                        mask = self.model.create_mask(input_tensor)
                        decoder_output, decoder_hidden, cell, decoder_attention = self.model.decode(
                            decoder_input.reshape((1, -1)), decoder_hidden,
                            cell, encoder_outputs, mask)
                        #TODO: pad this
                        # decoder_attentions[di] = decoder_attention
                    else:
                        decoder_output, decoder_hidden, cell = self.model.decode(
                            decoder_input.reshape((1, -1)), decoder_hidden,
                            cell)
                else:
                    if use_attention:
                        mask = self.model.create_mask(input_tensor)
                        decoder_output, decoder_hidden, cell, decoder_attention = self.model.decoder(
                            decoder_input.reshape((1, -1)), decoder_hidden,
                            cell, encoder_outputs, mask)
                        #TODO: pad this
                        # decoder_attentions[di] = decoder_attention
                    else:
                        decoder_output, decoder_hidden, cell = self.model.decoder(
                            decoder_input.reshape((1, -1)), decoder_hidden,
                            cell)

                top_pred_token = decoder_output.argmax(-1)
                decoded_tokens.append(top_pred_token.item())
                if top_pred_token.item() == utils.EOS_IDX:
                    break

                decoder_input = top_pred_token.squeeze().detach()

            decoded_words = self.vocab_transform[utils.tgt_lang].lookup_tokens(
                decoded_tokens)

            return decoded_words[:-1], decoder_attentions[:di + 1]

    def evaluate_randomly(self, src_tokens, tgt_tokens):
        use_attention = False

        if 'attention' in self.name:
            use_attention = True

        src_words = self.vocab_transform[utils.src_lang].lookup_tokens(
            list(src_tokens.detach().cpu().numpy()))
        tgt_words = self.vocab_transform[utils.tgt_lang].lookup_tokens(
            list(tgt_tokens.detach().cpu().numpy()))

        logging.info(f'> {" ".join(src_words)}')
        logging.info(f'= {" ".join(tgt_words)}')
        output_words, attentions = self.rnn_translate(
            input_tensor=src_tokens,
            tgt_tensor=None,
            use_attention=use_attention)
        output_sentence = ' '.join(output_words)
        logging.info(f'< {output_sentence}')

    def calc_test_bleu(self):
        loader = self.create_test_loader()
        self.model.load_state_dict(
            torch.load(f'{self.save_dir}models/{self.name}.pt'))

        use_attention = False

        if 'attention' in self.name:
            use_attention = True

        self.model.eval()
        tgts = []
        pred_tgts = []
        with torch.no_grad():
            for src, tgt, in loader:
                src = src.transpose(-1, -2).to(self.device)
                tgt = tgt.transpose(-1, -2).to(self.device)

                for sentence in range(src.size(0)):
                    pred_tgt, _ = self.rnn_translate(
                        src[sentence],
                        tgt[sentence],
                        use_attention=use_attention)
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

    def create_test_loader(self):
        test_data = Multi30k(split='test',
                             language_pair=(utils.src_lang, utils.tgt_lang))

        loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=self.batch_size,
                                             collate_fn=utils.collate_fn,
                                             shuffle=False)

        return loader

    def ensemble_calc_test_bleu(self):
        loader = self.create_test_loader()
        self.model.load_state_dict(
            torch.load(
                f'{self.save_dir}models/{self.name}_ensemble_member_0.pt'))

        if 'attention' in self.name:
            use_attention = True
            self.other_model = AttentionSeq2SeqLSTM(
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                dropout_prob=self.dropout_prob,
                device=self.device,
                n_layers=self.n_layers).to(self.device)
        else:
            use_attention = False
            self.other_model = Seq2SeqLSTM(src_vocab_size=self.src_vocab_size,
                                           tgt_vocab_size=self.tgt_vocab_size,
                                           embed_size=self.embed_size,
                                           hidden_size=self.hidden_size,
                                           dropout_prob=self.dropout_prob,
                                           device=self.device,
                                           n_layers=self.n_layers).to(
                                               self.device)

        self.other_model.load_state_dict(
            torch.load(
                f'{self.save_dir}models/{self.name}_ensemble_member_1.pt'))

        self.model.eval()
        self.other_model.eval()
        tgts = []
        pred_tgts = []
        with torch.no_grad():
            for src, tgt, in loader:
                src = src.transpose(-1, -2).to(self.device)
                tgt = tgt.transpose(-1, -2).to(self.device)

                for sentence in range(src.size(0)):
                    pred_tgt, _ = self.ensemble_rnn_translate(
                        src[sentence],
                        tgt[sentence],
                        use_attention=use_attention)
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

    def ensemble_rnn_translate(self,
                               input_tensor,
                               tgt_tensor,
                               use_attention=False):
        self.model.eval()
        self.other_model.eval()

        with torch.no_grad():
            input_length = input_tensor.size(0)

            # first model
            encoder_outputs, encoder_hidden, cell = self.model.encoder(
                input_tensor.reshape((1, -1)))

            decoder_input = torch.tensor([[utils.BOS_IDX]],
                                         device=self.device)  #BOS
            decoder_hidden = encoder_hidden

            # second model
            encoder_outputs_other, encoder_hidden_other, cell_other = self.other_model.encoder(
                input_tensor.reshape((1, -1)))

            decoder_input_other = torch.tensor([[utils.BOS_IDX]],
                                               device=self.device)  #BOS
            decoder_hidden_other = encoder_hidden_other

            decoded_tokens = []
            decoder_attentions = torch.zeros(utils.MAX_LENGTH,
                                             utils.MAX_LENGTH)

            for di in range(utils.MAX_LENGTH):
                # first model
                if use_attention:
                    mask = self.model.create_mask(input_tensor)
                    decoder_output, decoder_hidden, cell, decoder_attention = self.model.decoder(
                        decoder_input.reshape((1, -1)), decoder_hidden, cell,
                        encoder_outputs, mask)
                    #TODO: pad this
                    # decoder_attentions[di] = decoder_attention
                else:
                    decoder_output, decoder_hidden, cell = self.model.decoder(
                        decoder_input.reshape((1, -1)), decoder_hidden, cell)

                # second model
                if use_attention:
                    mask_other = self.other_model.create_mask(input_tensor)
                    decoder_output_other, decoder_hidden_other, cell_other, decoder_attention = self.other_model.decoder(
                        decoder_input_other.reshape(
                            (1, -1)), decoder_hidden_other, cell_other,
                        encoder_outputs_other, mask_other)
                    #TODO: pad this
                    # decoder_attentions[di] = decoder_attention
                else:
                    decoder_output_other, decoder_hidden_other, cell_other = self.other_model.decoder(
                        decoder_input_other.reshape((1, -1)),
                        decoder_hidden_other, cell_other)

                combined_decoder_output = (decoder_output +
                                           decoder_output_other) / 2

                top_pred_token = combined_decoder_output.argmax(-1)
                decoded_tokens.append(top_pred_token.item())
                if top_pred_token.item() == utils.EOS_IDX:
                    break

                decoder_input = top_pred_token.squeeze().detach()
                decoder_input_other = top_pred_token.squeeze().detach()

            decoded_words = self.vocab_transform[utils.tgt_lang].lookup_tokens(
                decoded_tokens)

            return decoded_words[:-1], decoder_attentions[:di + 1]


class SubspaceRNNTrainer(RNNTrainer):

    def __init__(self, **kwargs) -> None:
        super(SubspaceRNNTrainer, self).__init__(**kwargs)

        self.model = Seq2SeqLSTMSubspace(src_vocab_size=self.src_vocab_size,
                                         tgt_vocab_size=self.tgt_vocab_size,
                                         embed_size=self.embed_size,
                                         hidden_size=self.hidden_size,
                                         dropout_prob=self.dropout_prob,
                                         device=self.device,
                                         n_layers=self.n_layers,
                                         seed=self.seed).to(self.device)

        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate)

        self.name = 'seq2seq_vanilla_lstms_subspace'

    def get_weight(self, m, i):
        if i == 0:
            return m.weight
        return getattr(m, f'weight_{i}')

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss = 0

        for src, tgt in loader:
            src = src.transpose(-1, -2).to(self.device)
            tgt = tgt.transpose(-1, -2).to(self.device)

            alpha = torch.rand(1, device=self.device)
            for m in self.model.modules():
                if isinstance(m, nn.Linear) or isinstance(
                        m, nn.LSTM) or isinstance(m, nn.Embedding):
                    setattr(m, f'alpha', alpha)

            self.optimizer.zero_grad()
            decoder_output, decoder_hidden, decoder_cell = self.model(src, tgt)

            output_for_loss = decoder_output[:, 1:].reshape(
                -1, decoder_output.size(-1))
            tgt_for_loss = tgt[:, 1:].reshape(-1)

            loss = self.criterion(output_for_loss, tgt_for_loss)

            # regularization
            num = 0.0
            norm = 0.0
            norm1 = 0.0
            for m in self.model.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                    vi = self.get_weight(m, 0)
                    vj = self.get_weight(m, 1)

                    num += (vi * vj).sum()
                    norm += vi.pow(2).sum()
                    norm1 += vj.pow(2).sum()
                if isinstance(m, nn.LSTM):
                    for layer_num in range(self.n_layers):
                        w_ih_i = getattr(m, f'weight_ih_l{layer_num}')
                        w_ih_j = getattr(m, f'weight_ih_l{layer_num}_1')

                        num += (w_ih_i * w_ih_j).sum()
                        norm += w_ih_i.pow(2).sum()
                        norm1 += w_ih_j.pow(2).sum()

                        w_hh_i = getattr(m, f'weight_hh_l{layer_num}')
                        w_hh_j = getattr(m, f'weight_hh_l{layer_num}_1')

                        num += (w_hh_i * w_hh_j).sum()
                        norm += w_hh_i.pow(2).sum()
                        norm1 += w_hh_j.pow(2).sum()

            loss += self.beta * (num.pow(2) / (norm * norm1))
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.grad_clip)

            self.optimizer.step()
            running_loss += loss.item()

        return running_loss

    def eval_epoch(self, loader: DataLoader):
        self.model.eval()

        running_losses = [0.0, 0.0, 0.0]
        random_iteration = random.randint(0, 7)
        alphas = [0.0, 0.5, 1.0]

        if self.val_midpoint_only:
            alphas = [0.5]

        for i, alpha in enumerate(alphas):
            for m in self.model.modules():
                if isinstance(m, nn.Linear) or isinstance(
                        m, nn.LSTM) or isinstance(m, nn.Embedding):
                    setattr(m, f'alpha', alpha)

            # compute losses
            with torch.no_grad():
                for j, (src, tgt) in enumerate(loader):
                    src = src.transpose(-1, -2).to(self.device)
                    tgt = tgt.transpose(-1, -2).to(self.device)

                    decoder_output, decoder_hidden, decoder_cell = self.model(
                        src, tgt, teacher_forcing_ratio=0)
                    output_for_loss = decoder_output[:, 1:].reshape(
                        -1, decoder_output.size(-1))
                    tgt_for_loss = tgt[:, 1:].reshape(-1)
                    loss = self.criterion(output_for_loss, tgt_for_loss)

                    running_losses[i] += loss.item()
                    if j == random_iteration and i == 0:
                        self.evaluate_randomly(src_tokens=src[0, :],
                                               tgt_tokens=tgt[0, :])

        # compute l2 and cos sim
        num = 0.0
        norm = 0.0
        norm1 = 0.0

        total_l2 = 0.0

        for m in self.model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                vi = self.get_weight(m, 0)
                vj = self.get_weight(m, 1)

                num += (vi * vj).sum()
                norm += vi.pow(2).sum()
                norm1 += vj.pow(2).sum()

                total_l2 += (vi - vj).pow(2).sum()
            if isinstance(m, nn.LSTM):
                for layer_num in range(self.n_layers):
                    w_ih_i = getattr(m, f'weight_ih_l{layer_num}')
                    w_ih_j = getattr(m, f'weight_ih_l{layer_num}_1')

                    num += (w_ih_i * w_ih_j).sum()
                    norm += w_ih_i.pow(2).sum()
                    norm1 += w_ih_j.pow(2).sum()

                    total_l2 += (w_ih_i - w_ih_j).pow(2).sum()

                    w_hh_i = getattr(m, f'weight_hh_l{layer_num}')
                    w_hh_j = getattr(m, f'weight_hh_l{layer_num}_1')

                    total_l2 += (w_hh_i - w_hh_j).pow(2).sum()

                    num += (w_hh_i * w_hh_j).sum()
                    norm += w_hh_i.pow(2).sum()
                    norm1 += w_hh_j.pow(2).sum()

        total_cosim = num.pow(2) / (norm * norm1)
        total_l2 = total_l2.sqrt()

        return running_losses, total_cosim.item(), total_l2.item()

    def calc_test_bleu(self):
        loader = self.create_test_loader()

        self.model.load_state_dict(
            torch.load(f'{self.save_dir}models/{self.name}.pt'))
        self.model.eval()

        use_attention = False

        if 'attention' in self.name:
            use_attention = True

        alphas = [i / 10 for i in range(0, 11)]
        bleu_scores = []

        for i, alpha in enumerate(tqdm(alphas)):
            for m in self.model.modules():
                if isinstance(m, nn.Linear) or isinstance(
                        m, nn.LSTM) or isinstance(m, nn.Embedding):
                    setattr(m, f'alpha', alpha)

            tgts = []
            pred_tgts = []
            with torch.no_grad():
                for src, tgt, in loader:
                    src = src.transpose(-1, -2).to(self.device)
                    tgt = tgt.transpose(-1, -2).to(self.device)

                    for sentence in range(src.size(0)):
                        pred_tgt, _ = self.rnn_translate(
                            src[sentence],
                            tgt[sentence],
                            use_attention=use_attention)
                        pred_tgts.append(pred_tgt)

                    # hack to remove <eos> <pad>
                    tgt_words = [[[
                        element for element in ' '.join(self.vocab_transform[
                            utils.tgt_lang].lookup_tokens(
                                list(tgt[example, 1:].cpu().numpy()))).
                        replace('<pad>', '').replace('<eos>', '').split(' ')
                        if element != ''
                    ]] for example in range(tgt.size(0))]

                    tgts += tgt_words
                bleu_scores.append(bleu_score(pred_tgts, tgts))

        self.save_metrics(bleu_scores, name=f'{self.name}_test_bleu_scores')

        return max(bleu_scores)


class AttentionRNNTrainer(RNNTrainer):

    def __init__(self, **kwargs) -> None:
        super(AttentionRNNTrainer, self).__init__(**kwargs)

        self.model = AttentionSeq2SeqLSTM(src_vocab_size=self.src_vocab_size,
                                          tgt_vocab_size=self.tgt_vocab_size,
                                          embed_size=self.embed_size,
                                          hidden_size=self.hidden_size,
                                          dropout_prob=self.dropout_prob,
                                          device=self.device,
                                          n_layers=self.n_layers).to(
                                              self.device)

        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate)

        self.name = 'seq2seq_attention_lstms'


class SubspaceAttentionRNNTrainer(SubspaceRNNTrainer):

    def __init__(self, **kwargs) -> None:
        super(SubspaceAttentionRNNTrainer, self).__init__(**kwargs)

        self.model = AttentionSeq2SeqLSTMSubspace(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            dropout_prob=self.dropout_prob,
            device=self.device,
            n_layers=self.n_layers,
            seed=self.seed).to(self.device)

        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate)

        self.name = 'seq2seq_attention_lstms_subspace'