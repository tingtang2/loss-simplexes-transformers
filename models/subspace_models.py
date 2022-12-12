from torch import nn
from torch.nn import functional as F
from models.subspace_layers import LinesLSTM, LinesLinear, LinesEmbedding


class Seq2SeqLSTMSubspace(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size,
                 hidden_size):
        super(Seq2SeqLSTMSubspace, self).__init__()

        self.src_embedding = LinesEmbedding(src_vocab_size, embed_size)
        self.encoder = LinesLSTM(input_size=embed_size,
                                 hidden_size=hidden_size,
                                 batch_first=True,
                                 bias=False)

        self.encoder_2_decoder = LinesLinear(2 * hidden_size, hidden_size)

        self.tgt_embedding = LinesEmbedding(tgt_vocab_size, embed_size)
        self.decoder = LinesLSTM(input_size=embed_size,
                                 hidden_size=hidden_size,
                                 batch_first=True,
                                 bias=False)
        self.output = LinesLinear(hidden_size, tgt_vocab_size)

    def forward(self, src_tokens, tgt_tokens):
        batch_size, seq_len = src_tokens.size()
        src_embed = self.src_embedding(src_tokens).view(
            batch_size, seq_len, -1)

        output, (hidden, cell) = self.encoder(src_embed)

        batch_size, tgt_seq_len = tgt_tokens.size()
        tgt_embed = self.tgt_embedding(tgt_tokens).view(
            batch_size, tgt_seq_len, -1)
        tgt_embed = F.relu(tgt_embed)

        output, (hidden, cell) = self.decoder(tgt_embed, (hidden, cell))
        return self.output(output)