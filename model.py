import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass

@dataclass
class TransformerConfig:
    """ Transformer hyperparameters """
    batch_size: int = 32 # B dimension - num examples to process in parallel
    context_size: int = 96 # T dimension - max sequence length
    # note that we require n_embed % head_size == 0 and n_embed / head_size = n_head
    n_embed: int = 512 # C dimension - embedding dimension
    head_size: int = 128 # Dk - head dimension
    n_head: int = 4 # num of attention heads
    n_block: int = 6 # num of decoder transformer blocks
    dropout: float = 0.4 # dropout rate

class Head(nn.Module):
    """ Single causal self attention head """
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.context_size, config.context_size)))
        self.dropout = nn.Dropout(config.dropout)

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
    """ Multi-headed attention block """
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.head_size == 0
        assert config.n_embed / config.head_size == config.n_head

        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # ensemble multiple heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ MLP layer (using GELU as per the GPT-2 implementation) """
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4*config.n_embed),
            nn.GELU(),
            nn.Linear(4*config.n_embed, config.n_embed),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Decoder transformer block (combine transformer, MLP, layernorm, residual connection) """
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHead(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        # residual connections (x + ...) to help gradients flow
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderTransformer(nn.Module):
    """ Assemble entire model by stacking blocks and final layer norm """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.context_size, config.n_embed)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_block)])
        self.ln_f = nn.LayerNorm(config.n_embed) # layer norm final layer
        self.unembed = nn.Linear(config.n_embed, config.vocab_size)

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
        pos_embed = self.position_embedding_table(torch.arange(T, device=x.device))
        x = tok_embed + pos_embed

        # run through transformer layers and final layer norm
        x = self.blocks(x)
        x = self.ln_f(x)

        # unembed to get logits
        logits = self.unembed(x)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, self.config.vocab_size)
            # same as targets = targets.view(B*T)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # clip tokens to context size
            x_context = x[:, -self.config.context_size:]

            # calling forward without using loss
            logits, _ = self(x_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x