import torch
import torch.nn as nn
from torch.nn import functional as F

from pydantic import BaseModel, Field, model_validator



class ModelConfig(BaseModel):
    """
    Sets transformer hyperparameters.
    """
    T: int = Field(default=128, gt=0)
    d: int = Field(default=2_048, gt=0)
    d_k: int = Field(default=256, gt=0)
    n_h: int = Field(default=8, gt=0)
    n_block: int = Field(default=6, gt=0)
    dropout: float = Field(default=0.4, ge=0, lt=1)
    n_vocab: int = Field(default=2_500, gt=0) # Only used with custom-bpe tokenization (with character/tiktoken, computed automatically)
    tokenization: str = Field(default="custom-bpe", choices=["character", "custom-bpe", "tiktoken"])
    n_kv: int = Field(default=4, gt=0) # Number of key/value heads for GQA (1 means MQA)

    @model_validator(mode="after")
    def validate_gqa(self) -> "ModelConfig":
        if self.n_h % self.n_kv != 0:
            raise ValueError(f"Constraint failed: n_h({self.n_h}) % n_kv({self.n_kv}) != 0")
        return self

    @model_validator(mode="after")
    def validate_head_size(self) -> "ModelConfig":
        if self.d / self.d_k != self.n_h:
            raise ValueError(f"Constraint failed: d({self.d}) / d_k({self.d_k}) != n_h({self.n_h})")
        return self



def precompute_freq(d_k: int, T: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequencies."""
    theta_i = base ** (-2 * torch.arange(0, d_k, 2) / d_k)
    m = torch.arange(T)
    freq = torch.outer(m, theta_i)
    return freq



def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K."""

    # Extract even/odd portions (Assumes q, k shape (B, T, n_h, d_k))
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]

    cos = freqs[None, :, None, :].cos()
    sin = freqs[None, :, None, :].sin()

    q_out_even = q_even * cos - q_odd * sin
    q_out_odd = q_even * sin + q_odd * cos
    k_out_even = k_even * cos - k_odd * sin
    k_out_odd = k_even * sin + k_odd * cos

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_out[..., 0::2] = q_out_even
    q_out[..., 1::2] = q_out_odd
    k_out[..., 0::2] = k_out_even
    k_out[..., 1::2] = k_out_odd

    return q_out, k_out



class MultiHead(nn.Module):
    """Defines a multi-headed attention block with Grouped-Query Attention (GQA)."""
    def __init__(self, config):
        super().__init__()
        self.d = config.d
        self.n_h = config.n_h
        self.n_kv = config.n_kv
        self.d_k = config.d_k
        self.T = config.T
        
        # TODO 3: Initialize batched linear layers for Q, K, and V instead of individual Heads.
        # Note: Q needs to output for `n_h` heads, while K and V output for `n_kv` heads.
        self.query = nn.Linear(self.d, self.d, bias=False)
        self.key = nn.Linear(self.d, self.n_kv * self.d_k, bias=False)
        self.value = nn.Linear(self.d, self.n_kv * self.d_k, bias=False)
        
        self.proj = nn.Linear(config.d, config.d)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(config.T, config.T)))
        self.register_buffer("freq", precompute_freq(config.d_k, config.T))

    def forward(self, x):
        # Note: need the sequence length here bc input may not yet be context_size (e.g., beginning generation from scratch)
        B, T, C = x.shape
        
        q = self.query(x).reshape(B, T, self.n_h, self.d_k)
        k = self.key(x).reshape(B, T, self.n_kv, self.d_k)
        v = self.value(x).reshape(B, T, self.n_kv, self.d_k)
        
        # TODO 5: Precompute frequencies and apply RoPE to Q and K
        # e.g., freqs_cis = precompute_freqs_cis(self.d_k, self.T).to(x.device)
        # q, k = apply_rotary_emb(q, k, freqs_cis[:T])]

        # Compute frequencies and apply RoPE
        q, k = apply_rope(q, k, self.freq[:T])
        
        q = q.transpose(1, 2)
        k = k.repeat_interleave(self.n_h // self.n_kv, dim=2).transpose(1, 2)
        v = v.repeat_interleave(self.n_h // self.n_kv, dim=2).transpose(1, 2)
        
        # TODO 7: Compute scaled dot-product attention
        # - Transpose Q, K, V to (B, n_h, T, d_k)
        # - Compute Q @ K^T / sqrt(d_k)
        # - Apply causal mask using self.tril
        # - Softmax and dropout
        # - Multiply by V

        # sa = softmax(m * (Q @ K^T / sqrt(d_k))) @ V
        out = q @ k.transpose(-2, -1) * self.d_k ** -0.5
        out = out.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        out = F.softmax(out, dim=-1)

        out = self.dropout(out)
        out = out @ v
        
        # Reshape back to (B, T, C) before projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.dropout(self.proj(out))



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
        # self.position_embedding_table removed for RoPE
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

        x = self.token_embedding_table(x)

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

    def generate(self, x, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # Clip tokens to context size
            x_context = x[:, -self.config.T:]

            # Calling forward without using loss
            logits, _ = self(x_context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x