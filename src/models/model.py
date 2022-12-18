# code adapted from https://pytorch.org/tutorials/beginner/translation_transformer.html and
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb

import torch
from torch import nn, Tensor
from torch.nn import Transformer
import torch.nn.functional as F

import math
import utils

import random


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) /
                        emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size,
                                                      dropout=dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


# RNN based networks
class EncoderRNN(nn.Module):

    def __init__(self, src_vocab_size, embed_size, hidden_size, dropout_prob,
                 n_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.src_tok_emb = TokenEmbedding(src_vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True)
        # bidirectional=True)

        self.dropout = nn.Dropout(dropout_prob)

        # self.encoder_2_decoder = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, inputs):
        batch_size, seq_len = inputs.size()
        src_emb = self.src_tok_emb(inputs).view(batch_size, seq_len, -1)
        src_emb = self.dropout(src_emb)

        output, (hidden, cell) = self.lstm(src_emb)
        return output, hidden, cell


class DecoderRNN(nn.Module):

    def __init__(self, tgt_vocab_size, embed_size, hidden_size, output_size,
                 dropout_prob, n_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs, hidden, cell):
        batch_size, seq_len = inputs.size()
        inputs_embed = self.tgt_tok_emb(inputs).view(batch_size, seq_len, -1)
        inputs_embed = F.relu(self.dropout(inputs_embed))

        output, (hidden, cell) = self.lstm(inputs_embed, (hidden, cell))
        output = self.out(output)
        return output, hidden, cell


class Seq2SeqLSTM(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size,
                 dropout_prob, device, n_layers):
        super(Seq2SeqLSTM, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = EncoderRNN(src_vocab_size=src_vocab_size,
                                  embed_size=embed_size,
                                  hidden_size=hidden_size,
                                  dropout_prob=dropout_prob,
                                  n_layers=n_layers)

        self.decoder = DecoderRNN(tgt_vocab_size=tgt_vocab_size,
                                  embed_size=embed_size,
                                  hidden_size=hidden_size,
                                  output_size=tgt_vocab_size,
                                  dropout_prob=dropout_prob,
                                  n_layers=n_layers)

        self.device = device

    def forward(self, src_tokens, tgt_tokens, teacher_forcing_ratio=0.5):
        batch_size, src_len = src_tokens.size()
        tgt_len = tgt_tokens.size(1)

        # tensor to store decoder outputs
        decoder_outputs = torch.zeros(batch_size, tgt_len,
                                      self.tgt_vocab_size).to(self.device)
        decoder_outputs[:, 0] = utils.BOS_IDX  # set first output to BOS

        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(
            src_tokens)

        decoder_input = tgt_tokens[:, 0].reshape(batch_size, 1)  # BOS token
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        for i in range(1, tgt_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell)

            decoder_outputs[:, i] = decoder_output.squeeze()

            teacher_force = random.random() < teacher_forcing_ratio
            top_pred_token = decoder_output.argmax(-1).reshape(batch_size, 1)

            decoder_input = tgt_tokens[:, i].reshape(
                batch_size, 1) if teacher_force else top_pred_token.detach()

        return decoder_outputs, decoder_hidden, decoder_cell


## with attention


class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim, n_layers):
        super(Attention, self).__init__()

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

        self.coalesce_layers = nn.Linear(n_layers, 1)

    def forward(self, hidden, encoder_outputs, mask):

        #hidden = [n layers, batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim]

        batch_size = encoder_outputs.shape[0]
        hid_dim = encoder_outputs.shape[-1]
        src_len = encoder_outputs.shape[1]

        #repeat decoder hidden state src_len times
        hidden = F.relu(
            self.coalesce_layers(hidden.reshape(batch_size, hid_dim, -1)))

        hidden = hidden.transpose(-1, -2)
        hidden = hidden.repeat(1, src_len, 1)

        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        #attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class AttentionDecoderRNN(nn.Module):

    def __init__(self, tgt_vocab_size, embed_size, hidden_size, output_size,
                 dropout_prob, n_layers):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=hidden_size + embed_size,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=n_layers)
        self.out = nn.Linear(hidden_size + hidden_size + embed_size,
                             output_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.attention = Attention(enc_hid_dim=hidden_size,
                                   dec_hid_dim=hidden_size,
                                   n_layers=n_layers)

    def forward(self, inputs, hidden, cell, encoder_outputs, mask):
        batch_size, seq_len = inputs.size()

        # compute inputs embeds
        inputs_embed = self.tgt_tok_emb(inputs).view(batch_size, seq_len, -1)
        inputs_embed = F.relu(self.dropout(inputs_embed))

        # compute attention
        attention_vals = self.attention(hidden, encoder_outputs, mask)

        #a = [batch size, 1, src len]
        attention_vals = attention_vals.unsqueeze(1)

        weighted_encoder_outputs = attention_vals @ encoder_outputs

        rnn_input = torch.cat((inputs_embed, weighted_encoder_outputs), dim=-1)

        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        # print('output size', output.size())
        # print('weighted size', weighted_encoder_outputs.size())
        # print('inputs size', inputs_embed.size())
        output = self.out(
            torch.cat((output, weighted_encoder_outputs, inputs_embed),
                      dim=-1))
        return output, hidden, cell, attention_vals.squeeze()


class AttentionSeq2SeqLSTM(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size,
                 dropout_prob, device, n_layers):
        super(AttentionSeq2SeqLSTM, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = EncoderRNN(src_vocab_size=src_vocab_size,
                                  embed_size=embed_size,
                                  hidden_size=hidden_size,
                                  dropout_prob=dropout_prob,
                                  n_layers=n_layers)

        self.decoder = AttentionDecoderRNN(tgt_vocab_size=tgt_vocab_size,
                                           embed_size=embed_size,
                                           hidden_size=hidden_size,
                                           output_size=tgt_vocab_size,
                                           dropout_prob=dropout_prob,
                                           n_layers=n_layers)

        self.device = device

    def create_mask(self, src_tokens):
        return src_tokens != utils.PAD_IDX

    def forward(self, src_tokens, tgt_tokens, teacher_forcing_ratio=0.5):
        batch_size, src_len = src_tokens.size()
        tgt_len = tgt_tokens.size(1)

        # tensor to store decoder outputs
        decoder_outputs = torch.zeros(batch_size, tgt_len,
                                      self.tgt_vocab_size).to(self.device)
        decoder_outputs[:, 0] = utils.BOS_IDX  # set first output to BOS

        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(
            src_tokens)

        decoder_input = tgt_tokens[:, 0].reshape(batch_size, 1)  # BOS token
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # create mask for attention
        mask = self.create_mask(src_tokens)

        for i in range(1, tgt_len):
            decoder_output, decoder_hidden, decoder_cell, attention = self.decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs,
                mask)

            decoder_outputs[:, i] = decoder_output.squeeze()

            teacher_force = random.random() < teacher_forcing_ratio
            top_pred_token = decoder_output.argmax(-1).reshape(batch_size, 1)

            decoder_input = tgt_tokens[:, i].reshape(
                batch_size, 1) if teacher_force else top_pred_token.detach()

        return decoder_outputs, decoder_hidden, decoder_cell