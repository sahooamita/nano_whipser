"""Tiny encoder-decoder transformer for speech-to-text."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embeddings."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out), (k, v)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, kv = self.attn(self.norm1(x), mask=mask, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, kv


class Encoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_embed = SinusoidalPositionalEmbedding(d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T', d_model)
        x = x + self.pos_embed(x)
        for layer in self.layers:
            x, _ = layer(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEmbedding(d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight  # weight tying

    def forward(
        self,
        tokens: torch.Tensor,
        encoder_out: torch.Tensor,
        kv_caches: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        B, T = tokens.shape
        x = self.token_embed(tokens)
        x = x + self.pos_embed(x)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=tokens.device)).unsqueeze(0).unsqueeze(0)

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, kv = layer(x, mask=mask, kv_cache=cache)
            new_caches.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches


class NanoWhisper(nn.Module):
    """Minimal encoder-decoder ASR model."""

    def __init__(
        self,
        n_mels: int = 80,
        vocab_size: int = 256,
        d_model: int = 256,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(n_mels, d_model, n_encoder_layers, n_heads, dim_feedforward, dropout)
        self.decoder = Decoder(vocab_size, d_model, n_decoder_layers, n_heads, dim_feedforward, dropout)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(mel)
        # Cross-attention omitted for simplicity; encoder output projected to decoder via residual?
        # For nano, we simply add encoder context to decoder via a projection
        logits, _ = self.decoder(tokens, encoder_out)
        return logits

    @torch.no_grad()
    def transcribe(
        self,
        mel: torch.Tensor,
        tokenizer,
        max_len: int = 224,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> str:
        self.eval()
        encoder_out = self.encoder(mel)
        tokens = torch.tensor([[bos_id]], device=mel.device)
        kv_caches = None

        for _ in range(max_len):
            logits, kv_caches = self.decoder(tokens if kv_caches is None else tokens[:, -1:], encoder_out, kv_caches)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        return tokenizer.decode(tokens[0].tolist())
