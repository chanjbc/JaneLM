import torch
import torch.nn as nn
from torch.nn import functional as F

from pydantic import BaseModel, Field, model_validator



class ModelConfig(BaseModel):
    """
    Sets transformer hyperparameters.
    """
    T: int = Field(default=96, gt=0)
    d: int = Field(default=512, gt=0)
    d_k: int = Field(default=128, gt=0)
    d_v: int = Field(default=128, gt=0)
    n_h: int = Field(default=4, gt=0)
    n_h: int = Field(default=1, gt=0)
    n_block: int = Field(default=6, gt=0)
    dropout: float = Field(default=0.4, ge=0, lt=1)
    n_vocab: int = Field(default=0, ge=0)
    tokenization: str = Field(default="tiktoken", choices=["tiktoken", "character"])

    @model_validator(mode="after")
    def validate_head_size(self) -> "ModelConfig":
        if self.d / self.d_k != self.n_h:
            raise ValueError(f"Constraint failed: {self.d} / {self.d_k} != {self.n_h}")
        return self



class Head(nn.Module):
    """Defines a single causal self attention head."""
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.d, config.d_k, bias=False)
        self.key = nn.Linear(config.d, config.d_k, bias=False)
        self.value = nn.Linear(config.d, config.d_k, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.T, config.T)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Only need the sequence length dimension. Needed because input may not yet be context_size (e.g., beginning generation from scratch) 
        _, T, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # QK^T / sqrt(d_k) (B x T x T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

        # Mask future tokens (causal self-attention)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Row-wise softmax
        weights = F.softmax(weights, dim=-1)

        # Regularize and finish by multiplying by V (self_att = softmax(QK^T/sqrt(dk))V)
        weights = self.dropout(weights)
        weights = weights @ v
        return weights



class MultiHead(nn.Module):
    """Defines a multi-headed attention block."""
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_h)])
        self.proj = nn.Linear(config.d, config.d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Ensemble multiple heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class FeedForward(nn.Module):
    """Defines an MLP layer (using GELU as per the GPT-2 implementation)."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d, 4*config.d),
            nn.GELU(),
            nn.Linear(4*config.d, config.d),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    """Defines a decoder transformer block (combine transformer, MLP, layernorm, residual connection)."""
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHead(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.d)
        self.ln2 = nn.LayerNorm(config.d)

    def forward(self, x):
        # Residual connections to prevent vanishing gradients
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class JaneLM(nn.Module):
    """Assemble entire model by stacking blocks and final layer norm."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.n_vocab, config.d)
        self.position_embedding_table = nn.Embedding(config.T, config.d)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_block)])
        self.ln_f = nn.LayerNorm(config.d) # layer norm final layer
        self.unembed = nn.Linear(config.d, config.n_vocab)

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Add asymmetric weight initialization
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        # Need T here because the input may not yet be context_size (e.g., beginning generation from scratch)
        B, T = x.shape

        # Retrieve embeddings
        tok_embed = self.token_embedding_table(x)
        pos_embed = self.position_embedding_table(torch.arange(T, device=x.device))
        x = tok_embed + pos_embed

        # Run through transformer layers and final layer norm
        x = self.blocks(x)
        x = self.ln_f(x)

        # Unembed to get logits
        logits = self.unembed(x)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, self.config.n_vocab)
            # same as targets = targets.view(B*T)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # Clip tokens to context size
            x_context = x[:, -self.config.T:]

            # Calling forward without using loss
            logits, _ = self(x_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x