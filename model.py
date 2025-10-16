"""
Character-level GPT model for proof tactics generation.

Based on the architecture in TRAINING_PLAN.md.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration."""

    vocab_size: int = 256
    context_length: int = 512
    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 10
    d_mlp: int = 1280
    dropout: float = 0.1
    use_bias: bool = False
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS norm: x / sqrt(mean(x^2) + eps) * weight
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        # Q, K, V projections
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.use_bias)

        # Output projection
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Calculate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # (B, nh, T, hd)

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.proj(y))

        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_mlp, bias=config.use_bias)
        self.fc2 = nn.Linear(config.d_mlp, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """Character-level GPT model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm
        self.norm = RMSNorm(config.d_model)

        # Language modeling head
        if config.tie_embeddings:
            # Tie weights with token embedding
            self.lm_head = lambda x: F.linear(x, self.token_embedding.weight)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params:,}")

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T) for loss calculation

        Returns:
            logits (B, T, vocab_size) if targets is None
            (logits, loss) if targets is provided
        """
        B, T = idx.size()
        assert (
            T <= self.config.context_length
        ), f"Sequence length {T} exceeds context length {self.config.context_length}"

        # Token embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)

        # Position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos)  # (T, d_model)

        # Combine embeddings
        x = self.dropout(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Language modeling head
        if isinstance(self.lm_head, nn.Linear):
            logits = self.lm_head(x)
        else:
            logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return (logits, loss) if loss is not None else logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate new tokens.

        Args:
            idx: Starting indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Generated indices (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = (
                idx
                if idx.size(1) <= self.config.context_length
                else idx[:, -self.config.context_length :]
            )

            # Forward pass
            logits = self(idx_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Apply top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = False

                # Scatter to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
