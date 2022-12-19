# TODO: Restructure this

import torch

SRC_VOCAB_SIZE = len(vocab_transform[utils.src_lang])
TGT_VOCAB_SIZE = len(vocab_transform[utils.tgt_lang])

EMB_SIZE = 256
NHEAD = 4
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 18

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, NHEAD, SRC_VOCAB_SIZE,
                                 TGT_VOCAB_SIZE, FFN_HID_DIM)
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

# training loop
for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, train_data, criterion,
                             BATCH_SIZE, device)
    end_time = timer()
    val_loss = evaluate(transformer, val_data, criterion, BATCH_SIZE, device)
    print((
        f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
        f"Epoch time = {(end_time - start_time):.3f}s"))


# utils for decoding
####################################################################
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz),
                                  device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(
            torch.bool)).to(device)

        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str, vocab_transform):
    model.eval()
    src = text_transform[src_lang](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,
                               src,
                               src_mask,
                               max_len=num_tokens + 5,
                               start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[tgt_lang].lookup_tokens(
        list(tgt_tokens.cpu().numpy()))).replace("<bos>",
                                                 "").replace("<eos>", "")


# train and eval functions for transformer
##################################################################################
def train_epoch(model, optimizer, dataset, criterion, batch_size, device):
    model.train()
    losses = 0
    train_dataloader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, device, PAD_IDX)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]),
                         tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model, data, criterion, batch_size, device):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(data,
                                batch_size=batch_size,
                                collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, device, PAD_IDX)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                       tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]),
                         tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)
