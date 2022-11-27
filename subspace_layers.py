# implementation from https://github.com/vaseline555/SuPerFed/blob/main/src/models/layers.py

from torch import nn
from torch.nn import functional as F
import torch

import math

# for compatibility
StandardLinear = nn.Linear
StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d
StandardLSTM = nn.LSTM
StandardEmbedding = nn.Embedding

# Linear layer implementation
class SubspaceLinear(nn.Linear):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        x = F.linear(input=x, weight=w, bias=self.bias)
        return x

class TwoParamLinear(SubspaceLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, seed):
        if seed == -1: # SCAFFOLD
            torch.nn.init.zeros_(self.weight_local)
        else:
            torch.manual_seed(seed)
            torch.nn.init.xavier_normal_(self.weight_local)

class LinesLinear(TwoParamLinear):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight_local
        return w


# LSTM layer
# https://discuss.pytorch.org/t/defining-weight-manually-for-lstm/102360/2
class SubspaceLSTM(nn.LSTM):
    def forward(self, x):
        w = self.get_weight()
        h = (
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device), 
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        )
        with torch.no_grad():
            torch._cudnn_rnn_flatten_weight(
                weight_arr=w, 
                weight_stride0=(4 if self.bias else 2),
                input_size=self.input_size,
                mode=torch.backends.cudnn.rnn.get_cudnn_mode('LSTM'),
                hidden_size=self.hidden_size,
                proj_size=0,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False
            )
        result = torch._VF.lstm(x, h, w, self.bias, self.num_layers, 0.0, self.training, self.bidirectional, self.batch_first) 
        return result[0], result[1:]
    
class TwoParamLSTM(SubspaceLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for l in range(self.num_layers):
            setattr(self, f'weight_hh_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'weight_hh_l{l}'))))
            setattr(self, f'weight_ih_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'weight_ih_l{l}'))))
            if self.bias:
                setattr(self, f'bias_hh_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'bias_hh_l{l}'))))
                setattr(self, f'bias_ih_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'bias_ih_l{l}'))))
        
    def initialize(self, seed):
        if seed == -1: # SCAFFOLD
            for l in range(self.num_layers):
                torch.nn.init.zeros_(getattr(self, f'weight_hh_l{l}_local'))
                torch.nn.init.zeros_(getattr(self, f'weight_ih_l{l}_local'))
        else:
            for l in range(self.num_layers):
                torch.manual_seed(seed)
                torch.nn.init.uniform_(getattr(self, f'weight_hh_l{l}_local'), a=math.sqrt(1 / self.hidden_size) * -1, b=math.sqrt(1 / self.hidden_size))
                torch.nn.init.uniform_(getattr(self, f'weight_ih_l{l}_local'), a=math.sqrt(1 / self.hidden_size) * -1, b=math.sqrt(1 / self.hidden_size))
            
class LinesLSTM(TwoParamLSTM):
    def get_weight(self):
        weight_list = []
        for l in range(self.num_layers):
            weight_list.append((1 - self.alpha) * getattr(self, f'weight_ih_l{l}') + self.alpha * getattr(self, f'weight_ih_l{l}_local'))
            weight_list.append((1 - self.alpha) * getattr(self, f'weight_hh_l{l}') + self.alpha * getattr(self, f'weight_hh_l{l}_local'))
        return weight_list

    
    
# Embedding layer
class SubspaceEmbedding(nn.Embedding):
    def forward(self, x):
        w = self.get_weight()
        x = F.embedding(input=x, weight=w)
        return x

class TwoParamEmbedding(SubspaceEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))
    
    def initialize(self, seed):
        if seed == -1: # SCAFFOLD
            torch.nn.init.zeros_(self.weight_local)
        else:
            torch.manual_seed(seed)
            torch.nn.init.normal_(self.weight_local)
        
class LinesEmbedding(TwoParamEmbedding):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight_local
        return w