# base class for experiments
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from torchtext.datasets import Multi30k
from tqdm import tqdm

import utils

from data import get_vocabs

from torch.utils.data import DataLoader


class BaseTrainer(ABC):

    def __init__(self,
                 optimizer_type,
                 criterion,
                 device: str,
                 save_dir: Union[str, Path],
                 batch_size: int,
                 dropout_prob: float,
                 learning_rate: float,
                 save_plots: bool = True,
                 seed: int = 11202022,
                 debug: bool = False,
                 **kwargs) -> None:
        super().__init__()

        # basic configs every trainer needs
        self.optimizer_type = optimizer_type
        self.criterion = criterion
        self.device = torch.device(device)

        if debug:
            self.device = torch.device('cpu')

        self.save_plots = save_plots
        self.save_dir = save_dir
        self.seed = seed
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate

        # extra configs in form of kwargs
        for key, item in kwargs.items():
            setattr(self, key, item)

    def create_dataloaders(self):
        train_data = Multi30k(split='train',
                              language_pair=(utils.src_lang, utils.tgt_lang))
        val_data = Multi30k(split='valid',
                            language_pair=(utils.src_lang, utils.tgt_lang))

        self.vocab_transform = get_vocabs(utils.src_lang, utils.tgt_lang)

        self.src_vocab_size = len(self.vocab_transform[utils.src_lang])
        self.tgt_vocab_size = len(self.vocab_transform[utils.tgt_lang])

        train_dataloader = DataLoader(train_data,
                                      batch_size=self.batch_size,
                                      collate_fn=utils.collate_fn)
        val_dataloader = DataLoader(val_data,
                                    batch_size=self.batch_size,
                                    collate_fn=utils.collate_fn,
                                    shuffle=False)

        return train_dataloader, val_dataloader

    @abstractmethod
    def run_experiment(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def eval_epoch(self):
        pass

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), f'{self.save_dir}models/{name}.pt')

    def save_metrics(self, metrics: List[float], name: str, phase: str):
        save_name = f'{name}_{phase}.json'
        with open(Path(self.save_dir, 'metrics', save_name), 'w') as f:
            json.dump(metrics, f)