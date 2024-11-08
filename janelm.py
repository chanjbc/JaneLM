import torch
import torch.nn as nn
from torch.nn import functional as F

from model import DecoderTransformer, TransformerConfig


# training hyperparameters
max_iters: int = 1_000 # num epochs
eval_interval: int = 250 # how frequently the model is evaluated
eval_iters: int = 250 # how many runs to average over when evaluating
learning_rate: int = 1e-4
loss_tolerance = 0.075 # if the validaiton loss increases by this amount, stop training


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = TransformerConfig(
    batch_size=1,
    context_size=1,
    n_embed=4,
    head_size=4,
    n_head=1,
    n_block=1,
    dropout=0
)


# get file and read
with open('./data/janeausten.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# # optional dataset statistics
# print(f"Length: {len(text):,}")
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# print(f"Vocab Size: {vocab_size}")
# print(f"Characters: {''.join(chars)}")

# # NO TIKTOKEN: create encoders/decoders
# stoi = { ch:i for i, ch in enumerate(chars) }
# itos = { i:ch for i, ch in enumerate(chars) }
# encode = lambda s: [ stoi[c] for c in s ]
# decode = lambda l: ''.join([ itos[c] for c in l ])
# data = torch.tensor(encode(text), dtype=torch.long)

# TIKTOKEN
import tiktoken
enc = tiktoken.get_encoding('gpt2')
TransformerConfig.vocab_size = enc.n_vocab
data = torch.tensor(enc.encode(text), dtype=torch.long)


# train/test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, config):
    """Get batches of length context size and their labels."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.context_size, (config.batch_size,))
    x = torch.stack([ data[i:i+config.context_size] for i in ix ])
    # corresponding label is the next character
    y = torch.stack([ data[i+1:i+1+config.context_size] for i in ix ])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(config):
    """Get training and validation loss. """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, config)
            # only need loss, not logits
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out




model = DecoderTransformer(config)
m = model.to(device)
print(config)
print(f"Number of Parameters: {sum(p.numel() for p in m.parameters())/1e6} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best = float('inf')
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(config)
        print(f"STEP {iter}: Train Loss: {losses['train']:.4f}; Val Loss: {losses['val']:.4f}")
        if losses['val'] < best:
            torch.save(model.state_dict(), './model.pth')
            best = min(best, losses['val'])
        # stop if model begins to overfit significantly: may need to tweak value
        if losses['val'] > best + loss_tolerance:
            break

    xb, yb = get_batch('train', config)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()






model_state = torch.load('./model.pth', map_location=device)
best_model = DecoderTransformer(config).to(device)
best_model.load_state_dict(model_state)

# create mini batch with 0 (start token)
idx = torch.zeros((1, 1), dtype=torch.long, device=device)

# # NO TIKTOKEN: evaluate using characters
# print(decode(best_model.generate(idx, max_new_tokens=500)[0].tolist()))
# open('sample.txt', 'w').write(decode(best_model.generate(idx, max_new_tokens=1_000)[0].tolist()))

# TIKTOKEN:
# print(enc.decode(best_model.generate(idx, max_new_tokens=500)[0].tolist()))
# write results of generation to sample file (may want to change errors= setting for undefined characters)
open('sample.txt', 'w').write(enc.decode(best_model.generate(idx, max_new_tokens=10_000)[0].tolist(), errors='ignore'))

