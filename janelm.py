import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters - note that the two constraints are n_embed % head_size == 0 and n_embed / head_size = n_head
batch_size: int = 32         # "B" dimension - num examples to process in parallel
context_size: int = 96      # "T" dimension - max sequence length
n_embed: int = 512           # "C" dimension - embedding dimension
head_size: int = 128         # "Dk"
n_head: int = 4
n_block: int = 6             # num of decoder transformer blocks

max_iters: int = 15_000       # num epochs
eval_interval: int = 250     # how frequently the model is evaluated
eval_iters: int = 250        # how many runs to average over when evaluating
learning_rate: int = 1e-4

dropout = 0.4
loss_tolerance = 0.075 # if the validaiton loss increases by this amount, stop training

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Model Parameters: B: {batch_size}\tn_h: {n_head}\tT: {context_size}\tn_embed: {n_embed}\thead_size: {head_size}\tDropout: {dropout}")



# get file and read
with open('austen.txt', 'r', encoding='utf-8') as f:
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
vocab_size = enc.n_vocab
data = torch.tensor(enc.encode(text), dtype=torch.long)


# train/test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Get batches of length context size and their labels."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([ data[i:i+context_size] for i in ix ])
    # corresponding label is the next character
    y = torch.stack([ data[i+1:i+1+context_size] for i in ix ])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """Get training and validation loss. """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """Create a single causal self attention block."""
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # only need the sequence length dimension. needed because input may not yet be context_size (e.g., beginning generation from scratch) 
        _, T, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # QK^T / sqrt(Dk) (B x T x T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        # mask future tokens (causal self attention)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # row-wise softmax
        weights = F.softmax(weights, dim=-1)

        # regularize and finish by multiplying by V (self_att = softmax(QK^T/sqrt(dk))V)
        weights = self.dropout(weights)
        weights = weights @ v
        return weights

class MultiHead(nn.Module):
    """Create multi-headed attention block."""
    def __init__(self, n_head, head_size):
        super().__init__()
        assert n_embed % head_size == 0
        assert n_embed / head_size == n_head

        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ensemble multiple heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Create MLP layer (using GELU as per the GPT-2 implementation)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.GELU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Create decoder transformer block."""
    def __init__(self, n_head, head_size):
        super().__init__()
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # residual connections (x + ...) to help gradients flow
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramGPT(nn.Module):
    """Create entire model by stacking blocks."""
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_head, head_size) for _ in range(n_block)])
        self.ln_f = nn.LayerNorm(n_embed) # layer norm final layer
        self.unembed = nn.Linear(n_embed, vocab_size)

        # apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        # we need to grab T here because the input may not yet be context_size (e.g., beginning generation from scratch)
        B, T = x.shape

        # retrieve embeddings
        tok_embed = self.token_embedding_table(x)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embed + pos_embed

        # run through transformer layers and final layer norm
        x = self.blocks(x)
        x = self.ln_f(x)

        # unembed to get logits
        logits = self.unembed(x)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, vocab_size)
            # same as targets = targets.view(B*T)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # clip tokens to context size
            x_context = x[:, -context_size:]

            # calling forward without using loss
            logits, _ = self(x_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x



model = BigramGPT()
m = model.to(device)
print(f"Number of Parameters: {sum(p.numel() for p in m.parameters())/1e6} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best = float('inf')
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"STEP {iter}: Train Loss: {losses['train']:.4f}; Val Loss: {losses['val']:.4f}")
        if losses['val'] < best:
            torch.save(model.state_dict(), './model.pth')
            best = min(best, losses['val'])
        # stop if model begins to overfit significantly: may need to tweak value
        if losses['val'] > best + loss_tolerance:
            break

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()






model_state = torch.load('./model.pth', map_location=device)
best_model = BigramGPT().to(device)
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

