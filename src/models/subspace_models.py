import random

import torch
from torch import nn
from torch.nn import functional as F

import utils
from models.subspace_layers import LinesEmbedding, LinesLinear, LinesLSTM


class Seq2SeqLSTMSubspace(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size,
                 dropout_prob, n_layers, device, seed):
        super(Seq2SeqLSTMSubspace, self).__init__()

        self.src_embedding = LinesEmbedding(src_vocab_size, embed_size)
        self.src_embedding.initialize(seed)
        self.encoder = LinesLSTM(input_size=embed_size,
                                 hidden_size=hidden_size,
                                 batch_first=True,
                                 num_layers=n_layers,
                                 bias=False)
        self.encoder.initialize(seed)

        self.dropout = nn.Dropout(dropout_prob)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # self.encoder_2_decoder = LinesLinear(2 * hidden_size, hidden_size)

        self.tgt_embedding = LinesEmbedding(tgt_vocab_size, embed_size)
        self.tgt_embedding.initialize(seed)
        self.decoder = LinesLSTM(input_size=embed_size,
                                 hidden_size=hidden_size,
                                 num_layers=n_layers,
                                 batch_first=True,
                                 bias=False)
        self.decoder.initialize(seed)
        self.output = LinesLinear(hidden_size, tgt_vocab_size)
        self.output.initialize(seed)
        self.device = device

    def encode(self, src_tokens):
        batch_size, seq_len = src_tokens.size()
        src_embed = self.src_embedding(src_tokens).view(
            batch_size, seq_len, -1)

        src_embed = self.dropout(src_embed)

        output, (hidden, cell) = self.encoder(src_embed)

        return output, hidden, cell

    def decode(self, tgt_tokens, hidden, cell):
        batch_size, tgt_seq_len = tgt_tokens.size()
        tgt_embed = self.tgt_embedding(tgt_tokens).view(
            batch_size, tgt_seq_len, -1)
        tgt_embed = F.relu(self.dropout(tgt_embed))

        output, (hidden, cell) = self.decoder(tgt_embed, (hidden, cell))
        return self.output(output), hidden, cell

    def forward(self, src_tokens, tgt_tokens, teacher_forcing_ratio=0.5):
        batch_size, seq_len = src_tokens.size()
        tgt_len = tgt_tokens.size(1)

        # tensor to store decoder outputs
        decoder_outputs = torch.zeros(batch_size, tgt_len,
                                      self.tgt_vocab_size).to(self.device)
        decoder_outputs[:, 0] = utils.BOS_IDX  # set first output to BOS

        encoder_output, encoder_hidden, encoder_cell = self.encode(
            src_tokens=src_tokens)

        decoder_input = tgt_tokens[:, 0].reshape(batch_size, 1)  # BOS token
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        for i in range(1, tgt_len):
            decoder_output, decoder_hidden, decoder_cell = self.decode(
                decoder_input, decoder_hidden, decoder_cell)

            decoder_outputs[:, i] = decoder_output.squeeze()

            teacher_force = random.random() < teacher_forcing_ratio
            top_pred_token = decoder_output.argmax(-1).reshape(batch_size, 1)

            decoder_input = tgt_tokens[:, i].reshape(
                batch_size, 1) if teacher_force else top_pred_token.detach()

        return decoder_outputs, decoder_hidden, decoder_cell


class AttentionSeq2SeqLSTMSubspace(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size,
                 dropout_prob, device, n_layers, seed):
        super(AttentionSeq2SeqLSTMSubspace, self).__init__()

        # encoder stuff
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_embedding = LinesEmbedding(src_vocab_size, embed_size)
        self.src_embedding.initialize(seed)
        self.encoder = LinesLSTM(input_size=embed_size,
                                 hidden_size=hidden_size,
                                 batch_first=True,
                                 num_layers=n_layers,
                                 bias=False)
        self.encoder.initialize(seed)

        self.dropout = nn.Dropout(dropout_prob)

        # attention stuff
        self.attn = LinesLinear(2 * hidden_size, hidden_size)
        self.v = LinesLinear(hidden_size, 1, bias=False)

        self.coalesce_layers = LinesLinear(n_layers, 1)

        # decoder stuff
        self.tgt_embedding = LinesEmbedding(tgt_vocab_size, embed_size)
        self.tgt_embedding.initialize(seed)
        self.decoder = LinesLSTM(input_size=hidden_size + embed_size,
                                 hidden_size=hidden_size,
                                 num_layers=n_layers,
                                 batch_first=True,
                                 bias=False)
        self.decoder.initialize(seed)
        self.output = LinesLinear(2 * hidden_size + embed_size, tgt_vocab_size)
        self.output.initialize(seed)

        self.device = device

    def create_mask(self, src_tokens):
        return src_tokens != utils.PAD_IDX

    def encode(self, src_tokens):
        batch_size, seq_len = src_tokens.size()
        src_embed = self.src_embedding(src_tokens).view(
            batch_size, seq_len, -1)

        src_embed = self.dropout(src_embed)

        output, (hidden, cell) = self.encoder(src_embed)

        return output, hidden, cell

    def decode(self, tgt_tokens, hidden, cell, encoder_outputs, mask):
        batch_size, tgt_seq_len = tgt_tokens.size()
        tgt_embed = self.tgt_embedding(tgt_tokens).view(
            batch_size, tgt_seq_len, -1)
        tgt_embed = F.relu(self.dropout(tgt_embed))

        # compute attention
        batch_size = encoder_outputs.shape[0]
        hid_dim = encoder_outputs.shape[-1]
        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        new_hidden = F.relu(
            self.coalesce_layers(hidden.reshape(batch_size, hid_dim, -1)))

        new_hidden = new_hidden.transpose(-1, -2)
        new_hidden = new_hidden.repeat(1, src_len, 1)

        energy = torch.tanh(
            self.attn(torch.cat((new_hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)

        attention_vals = F.softmax(attention, dim=1)
        attention_vals = attention_vals.unsqueeze(1)

        weighted_encoder_outputs = attention_vals @ encoder_outputs
        rnn_input = torch.cat((tgt_embed, weighted_encoder_outputs), dim=-1)

        output, (hidden, cell) = self.decoder(rnn_input, (hidden, cell))
        output = self.output(
            torch.cat((output, weighted_encoder_outputs, tgt_embed), dim=-1))
        return output, hidden, cell, attention_vals.squeeze()

    def forward(self, src_tokens, tgt_tokens, teacher_forcing_ratio=0.5):
        batch_size, src_len = src_tokens.size()
        tgt_len = tgt_tokens.size(1)

        # tensor to store decoder outputs
        decoder_outputs = torch.zeros(batch_size, tgt_len,
                                      self.tgt_vocab_size).to(self.device)
        decoder_outputs[:, 0] = utils.BOS_IDX  # set first output to BOS

        encoder_outputs, encoder_hidden, encoder_cell = self.encode(src_tokens)

        decoder_input = tgt_tokens[:, 0].reshape(batch_size, 1)  # BOS token
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # create mask for attention
        mask = self.create_mask(src_tokens)

        for i in range(1, tgt_len):
            decoder_output, decoder_hidden, decoder_cell, attention = self.decode(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs,
                mask)

            decoder_outputs[:, i] = decoder_output.squeeze()

            teacher_force = random.random() < teacher_forcing_ratio
            top_pred_token = decoder_output.argmax(-1).reshape(batch_size, 1)

            decoder_input = tgt_tokens[:, i].reshape(
                batch_size, 1) if teacher_force else top_pred_token.detach()

        return decoder_outputs, decoder_hidden, decoder_cell