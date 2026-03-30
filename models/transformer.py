# models/transformer.py
# Transformer Encoder built fully from scratch using raw PyTorch tensor ops.
# NO nn.Linear, NO nn.LayerNorm, NO nn.Embedding, NO nn.Dropout, NO nn.MultiheadAttention.
# Every parameter is a manually initialized nn.Parameter.
# Every operation is an explicit tensor computation.

import math
import torch
import torch.nn as nn


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation — built from scratch using raw torch ops.
    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax from scratch."""
    x_max = x.max(dim=dim, keepdim=True).values
    e_x   = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)


def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Inverted dropout from scratch (no nn.Dropout)."""
    if not training or p == 0.0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1.0 - p)


# ── Manual Linear layer ────────────────────────────────────────────────────────

class Linear(nn.Module):
    """
    Fully-connected layer: y = x @ W.T + b
    Built from scratch — no nn.Linear.
    Weights initialized with Kaiming uniform (He initialization).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Kaiming uniform init: std = sqrt(2 / in_features)
        std = math.sqrt(2.0 / in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) → (..., out_features)
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


# ── Manual Embedding layer ─────────────────────────────────────────────────────

class Embedding(nn.Module):
    """
    Token embedding table: maps integer ids → dense vectors.
    Built from scratch — no nn.Embedding.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.padding_idx   = padding_idx
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)
        # Zero out the padding row
        with torch.no_grad():
            self.weight[padding_idx] = 0.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T) → (B, T, D) via index selection
        return self.weight[input_ids]


# ── Manual Layer Normalization ─────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """
    Layer normalization: normalize across last dimension.
    Formula: (x - mean) / sqrt(var + eps) * gamma + beta
    Built from scratch — no nn.LayerNorm.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))   # scale
        self.beta  = nn.Parameter(torch.zeros(normalized_shape))  # shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ── Positional Encoding ────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (not learned).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        pe       = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a parameter — not updated by optimizer)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → add positional signal
        return x + self.pe[:, :x.size(1), :]


# ── Multi-Head Self-Attention ──────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention built from scratch.

    Steps:
      1. Project input into Q, K, V using manual Linear layers.
      2. Split into H heads, reshape to (B, H, T, head_dim).
      3. Compute scaled dot-product: scores = Q @ K.T / sqrt(head_dim)
      4. Apply optional mask (set -inf on padding positions).
      5. Softmax → attention weights.
      6. Weighted sum of V.
      7. Concatenate heads, project with output Linear.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.dropout_p = dropout_p

        # Separate Q, K, V projection matrices (from scratch)
        self.W_q = Linear(embed_dim, embed_dim)
        self.W_k = Linear(embed_dim, embed_dim)
        self.W_v = Linear(embed_dim, embed_dim)
        self.W_o = Linear(embed_dim, embed_dim)  # output projection

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        H       = self.num_heads
        hd      = self.head_dim

        # 1. Linear projections
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Reshape into heads: (B, T, D) → (B, H, T, head_dim)
        Q = Q.view(B, T, H, hd).transpose(1, 2)
        K = K.view(B, T, H, hd).transpose(1, 2)
        V = V.view(B, T, H, hd).transpose(1, 2)

        # 3. Scaled dot-product attention scores: (B, H, T, T)
        scores = (Q @ K.transpose(-2, -1)) / self.scale

        # 4. Apply padding mask (mask=0 means "ignore this position")
        if mask is not None:
            # mask shape: (B, T) → broadcast to (B, 1, 1, T)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5. Softmax over key dimension → attention weights
        attn_weights = softmax(scores, dim=-1)
        attn_weights = dropout(attn_weights, self.dropout_p, self.training)

        # 6. Weighted sum of values: (B, H, T, head_dim)
        context = attn_weights @ V

        # 7. Concatenate heads: (B, H, T, hd) → (B, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, D)

        # 8. Output projection
        return self.W_o(context)


# ── Feed-Forward Network ───────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network:
        FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    Built from scratch using manual Linear layers and our custom GELU.
    """
    def __init__(self, embed_dim: int, ff_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.W1 = Linear(embed_dim, ff_dim)
        self.W2 = Linear(ff_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = gelu(self.W1(x))
        x = dropout(x, self.dropout_p, self.training)
        x = self.W2(x)
        return x


# ── Single Transformer Encoder Layer ──────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    """
    One encoder block:
        x = LayerNorm(x + MultiHeadSelfAttention(x))
        x = LayerNorm(x + FeedForward(x))

    Uses Pre-LN (norm before sublayer) for training stability.
    All sublayers built from scratch.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.attn   = MultiHeadSelfAttention(embed_dim, num_heads, dropout_p)
        self.ff     = FeedForward(embed_dim, ff_dim, dropout_p)
        self.norm1  = LayerNorm(embed_dim)
        self.norm2  = LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + dropout(self.attn(self.norm1(x), mask), self.dropout_p, self.training)
        # Pre-norm FFN with residual
        x = x + dropout(self.ff(self.norm2(x)), self.dropout_p, self.training)
        return x


# ── Full Medical Transformer Encoder ──────────────────────────────────────────

class MedicalTransformerEncoder(nn.Module):
    """
    Full transformer encoder for medical embeddings.

    Architecture:
        Input IDs → Embedding + Positional Encoding
                  → N × TransformerEncoderLayer
                  → [CLS] token → MLM head + Type head

    Every submodule is built from scratch:
        - Embedding (manual lookup table)
        - PositionalEncoding (sinusoidal, manual)
        - LayerNorm (manual mean/var normalization)
        - MultiHeadSelfAttention (manual Q/K/V projections + scaled dot-product)
        - FeedForward (manual matmuls + GELU)
        - Output heads (manual Linear projections)

    Outputs:
        token_out   : (B, T, vocab_size)  — logits for MLM
        cls_embed   : (B, embed_dim)      — [CLS] entity representation
        type_logits : (B, num_types)      — entity type prediction
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 256,
        num_layers:  int = 6,
        num_heads:   int = 8,
        ff_dim:      int = 1024,
        max_len:     int = 128,
        num_types:   int = 4,
        dropout_p:   float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Token embedding (manual lookup table — no nn.Embedding)
        self.token_embed = Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional encoding (fixed sinusoidal)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)

        # Stack of N encoder layers (all built from scratch)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_p)
            for _ in range(num_layers)
        ])

        # Final layer norm (manual)
        self.final_norm = LayerNorm(embed_dim)

        # MLM head: project token representations back to vocab
        # Two-layer projection for richer representation
        self.mlm_dense = Linear(embed_dim, embed_dim)
        self.mlm_norm  = LayerNorm(embed_dim)
        self.mlm_head  = Linear(embed_dim, vocab_size)

        # Type classification head: 3-layer MLP from [CLS] → num_types
        self.type_fc1  = Linear(embed_dim, embed_dim // 2)
        self.type_fc2  = Linear(embed_dim // 2, num_types)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            input_ids : (B, T) — token ids
            mask      : (B, T) — 1=attend, 0=ignore (optional)

        Returns:
            token_out   : (B, T, vocab_size)
            cls_embed   : (B, embed_dim)
            type_logits : (B, num_types)
        """
        # 1. Token embeddings + positional encoding
        x = self.token_embed(input_ids)    # (B, T, D)
        x = self.pos_enc(x)                # add positional signal
        x = dropout(x, 0.1, self.training)

        # 2. Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # 3. Final normalization
        x = self.final_norm(x)

        # 4. [CLS] token is always at position 0
        cls_embed = x[:, 0, :]             # (B, D)

        # 5. MLM head: dense → gelu → norm → project to vocab
        mlm_out   = gelu(self.mlm_dense(x))
        mlm_out   = self.mlm_norm(mlm_out)
        token_out = self.mlm_head(mlm_out) # (B, T, vocab_size)

        # 6. Type classification head: gelu → linear
        type_h      = gelu(self.type_fc1(cls_embed))
        type_logits = self.type_fc2(type_h)  # (B, num_types)

        return token_out, cls_embed, type_logits


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MedicalTransformerEncoder(
        vocab_size=500, embed_dim=64, num_layers=2, num_heads=4, ff_dim=128
    )
    dummy = torch.randint(0, 500, (2, 16))  # batch=2, seq=16
    tok_out, cls_emb, type_log = model(dummy)
    print("token_out shape :", tok_out.shape)    # (2, 16, 500)
    print("cls_embed shape  :", cls_emb.shape)   # (2, 64)
    print("type_logits shape:", type_log.shape)  # (2, 4)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total:,}")
    print("All from scratch — no nn.Linear, nn.LayerNorm, nn.Embedding used.")