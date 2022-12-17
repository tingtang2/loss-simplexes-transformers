from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer

from data import get_text_transforms, get_vocabs


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


src_lang = 'de'
tgt_lang = 'en'
token_transform = {}

token_transform[src_lang] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[tgt_lang] = get_tokenizer('spacy', language='en_core_web_sm')

text_transform = get_text_transforms(src_lang, tgt_lang, token_transform,
                                     get_vocabs(src_lang, tgt_lang),
                                     tensor_transform)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
MAX_LENGTH = 1000


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    #for pair in batch:
    for src_sample, tgt_sample in batch:
        # src_sample = pair[src_lang]
        # tgt_sample = pair[tgt_lang]
        src_batch.append(text_transform[src_lang](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[tgt_lang](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch