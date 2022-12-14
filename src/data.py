from datasets import load_dataset
import torch

from typing import List, Dict

# leftovers for WMT
# dataset = get_dataset()
# dataset.set_format(type='torch')
# train_data = dataset['train'][:configs['subsample_size']]['translation']
# val_data = dataset['validation'][:]['translation']


def get_dataset():
    dataset = load_dataset('wmt14', 'de-en')
    print(dataset)

    return dataset


def get_vocabs(src_lang: str,
               tgt_lang: str,
               path='/home/tingchen/loss-simplexes-transformers/data/',
               dataset='multi30k') -> Dict:
    vocab_transforms = {}

    for ln in [src_lang, tgt_lang]:
        vocab_transforms[ln] = torch.load(path + f'{ln}_vocab_{dataset}.vocab')

    return vocab_transforms


# helper function to club together sequential operations
def sequential_transforms(*transforms):

    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# src and tgt language text transforms to convert raw strings into tensors indices
def get_text_transforms(src_lang, tgt_lang, token_transform, vocab_transform,
                        tensor_transform):
    text_transform = {}
    for ln in [src_lang, tgt_lang]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln],  #Tokenization
            vocab_transform[ln],  #Numericalization
            tensor_transform)  # Add BOS/EOS and create tensor

    return text_transform
