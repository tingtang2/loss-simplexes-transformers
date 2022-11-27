import sys
import torch
from data import get_dataset, get_text_transforms, get_vocabs
from torch.nn.utils.rnn import pad_sequence
from model import Seq2SeqTransformer, EncoderRNN, DecoderRNN
from subspace_models import Seq2SeqLSTMSubspace
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from timeit import default_timer as timer
from typing import List
from tqdm import tqdm
from pathlib import Path

SEED = 111920222
torch.manual_seed(SEED)

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

src_lang = 'de'
tgt_lang = 'en'
token_transform = {}

token_transform[src_lang] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[tgt_lang] = get_tokenizer('spacy', language='en_core_web_sm')

text_transform = get_text_transforms(src_lang, tgt_lang, token_transform,  get_vocabs(src_lang, tgt_lang), tensor_transform)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
MAX_LENGTH = 1000


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for pair in batch:
        src_sample = pair[src_lang]
        tgt_sample = pair[tgt_lang]
        src_batch.append(text_transform[src_lang](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[tgt_lang](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


# utils for decoding
####################################################################
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)

        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str, vocab_transform):
    model.eval()
    src = text_transform[src_lang](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[tgt_lang].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

# train and eval functions for transformer
##################################################################################
def train_epoch(model, optimizer, dataset, criterion, batch_size, device):
    model.train()
    losses = 0
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device, PAD_IDX)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model, data, criterion, batch_size, device):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device, PAD_IDX)
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

# train and eval functions for rnn
##################################################################################################
def train_epoch_rnn(dataset, 
                    encoder, 
                    decoder, 
                    encoder_optimizer, 
                    decoder_optimizer, 
                    criterion, 
                    device, 
                    batch_size):
    encoder.train()
    decoder.train()
    running_loss = 0
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        src = src.transpose(-1, -2).to(device)
        tgt = tgt.transpose(-1, -2).to(device)
        
        encoder_outputs, encoder_hidden, encoder_cell = encoder(src)

        decoder_output, decoder_hidden = decoder(tgt[:, :-1], encoder_hidden, encoder_cell)
        loss = criterion(decoder_output.reshape(-1, decoder_output.size(-1)), tgt[:, 1:].reshape(-1))

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataset)

def evaluate_rnn(dataset, 
                 encoder, 
                 decoder, 
                 criterion, 
                 device, 
                 batch_size):
    encoder.eval()
    decoder.eval()
    running_loss = 0
    
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    with torch.no_grad():
        for src, tgt in eval_dataloader:
            src = src.transpose(-1, -2).to(device)
            tgt = tgt.transpose(-1, -2).to(device)
            
            encoder_outputs, encoder_hidden, encoder_cell = encoder(src)

            decoder_output, decoder_hidden = decoder(tgt[:, :-1], encoder_hidden, encoder_cell)
            loss = criterion(decoder_output.reshape(-1, decoder_output.size(-1)), tgt[:, 1:].reshape(-1))

            running_loss += loss.item()

    return running_loss / len(eval_dataloader)

def get_weight(m, i):
    if i == 0:
        return m.weight
    return getattr(m, f'weight_{i}')

def train_epoch_rnn_subspace(dataset, 
                             model, 
                             optimizer, 
                             criterion, 
                             device, 
                             batch_size,
                             beta,
                             **kwargs):
    model.train()
    running_loss = 0
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.transpose(-1, -2).to(device)
        tgt = tgt.transpose(-1, -2).to(device)
        
        alpha = torch.rand(1, device=device)
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM) or isinstance(m, nn.Embedding):
                setattr(m, f'alpha', alpha)

        optimizer.zero_grad()
        decoder_output = model(src, tgt[:, :-1])
        loss = criterion(decoder_output.reshape(-1, decoder_output.size(-1)), tgt[:, 1:].reshape(-1))

        # regularization
        num = 0.0
        norm = 0.0
        norm1 = 0.0
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                vi = get_weight(m, 0)
                vj = get_weight(m, 1)
                num += (vi * vj).sum()
                norm += vi.pow(2).sum()
                norm1 += vj.pow(2).sum()
            if isinstance(m, nn.LSTM):
                w_ih_i = getattr(m, 'weight_ih_l0')
                w_ih_j = getattr(m, 'weight_ih_l0_1')

                num += (w_ih_i * w_ih_j).sum()
                norm += w_ih_i.pow(2).sum()
                norm1 += w_ih_j.pow(2).sum()

                w_hh_i = getattr(m, 'weight_hh_l0')
                w_hh_j = getattr(m, 'weight_hh_l0_1')

                num += (w_hh_i * w_hh_j).sum()
                norm += w_hh_i.pow(2).sum()
                norm1 += w_hh_j.pow(2).sum()

        loss += beta * (num.pow(2) / (norm * norm1))

        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataset)

def evaluate_rnn_subspace(dataset, 
                          model, 
                          criterion, 
                          device, 
                          batch_size,
                          **kwargs):
    model.eval()
    running_losses = [0, 0, 0]
    
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    alphas = [0.0, 0.5, 1.0]
    for i, alpha in enumerate(alphas):
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM) or isinstance(m, nn.Embedding):
                setattr(m, f'alpha', alpha)
        
        with torch.no_grad():
            for src, tgt in eval_dataloader:
                src = src.transpose(-1, -2).to(device)
                tgt = tgt.transpose(-1, -2).to(device)

                decoder_output = model(src, tgt[:, :-1])
                loss = criterion(decoder_output.reshape(-1, decoder_output.size(-1)), tgt[:, 1:].reshape(-1))

                running_losses[i] += loss.item()

    return [loss / len(dataset) for loss in running_losses]

def main() -> int:
    dataset = get_dataset()
    dataset.set_format(type='torch')
    
    # configs
    configs = {'subsample_size': 10000,
               'model_type': 'rnn_subspace',
               'beta': 1.0,
               'save_dir': 'subspace_saved_metrics_models/',
               'debug': False}

    train_data = dataset['train'][:configs['subsample_size']]['translation']
    val_data = dataset['validation'][:]['translation']

    if configs['debug']:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_transform = get_vocabs(src_lang, tgt_lang)

    SRC_VOCAB_SIZE = len(vocab_transform[src_lang])
    TGT_VOCAB_SIZE = len(vocab_transform[tgt_lang])

    EMB_SIZE = 256
    NHEAD = 4
    FFN_HID_DIM = 512
    BATCH_SIZE = 32
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    NUM_EPOCHS = 18

    if configs['model_type'] == 'transformer':
        transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        transformer = transformer.to(device)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        # training loop
        for epoch in tqdm(range(1, NUM_EPOCHS+1)):
            start_time = timer()
            train_loss = train_epoch(transformer, optimizer, train_data, criterion, BATCH_SIZE, device)
            end_time = timer()
            val_loss = evaluate(transformer, val_data, criterion, BATCH_SIZE, device)
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    
    elif configs['model_type'] == 'rnn':
        encoder = EncoderRNN(SRC_VOCAB_SIZE, EMB_SIZE, EMB_SIZE, device).to(device)
        decoder = DecoderRNN(TGT_VOCAB_SIZE, EMB_SIZE, EMB_SIZE, TGT_VOCAB_SIZE, device).to(device)
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-5, amsgrad=True)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-5, amsgrad=True)

        # training loop
        for epoch in tqdm(range(1, NUM_EPOCHS+1)):
            start_time = timer()
            train_loss = train_epoch_rnn(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, BATCH_SIZE)
            end_time = timer()
            val_loss = evaluate_rnn(val_data, encoder, decoder, criterion, device, BATCH_SIZE)
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    
    elif configs['model_type'] == 'rnn_subspace':
        model = Seq2SeqLSTMSubspace(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE, EMB_SIZE).to(device)
        for m in model.modules():
            if hasattr(m, 'initialize'):
                m.initialize(SEED)
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, amsgrad=True)

        # training loop
        for epoch in tqdm(range(1, NUM_EPOCHS+1)):
            start_time = timer()
            train_loss = train_epoch_rnn_subspace(train_data, model, optimizer, criterion, device, BATCH_SIZE, **configs)
            end_time = timer()
            val_losses = evaluate_rnn_subspace(val_data, model, criterion, device, BATCH_SIZE)
            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss alpha 0: {val_losses[0]:.3f}, Val loss alpha 0.5: {val_losses[1]:.3f}, Val loss alpha 1: {val_losses[2]:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")

        # save model
        save_path = f'~/{configs["save_dir"]}/subspace_rnn.pt'
        torch.save(model.state_dict(), save_path)
    return 0

if __name__ == '__main__':
    sys.exit(main())