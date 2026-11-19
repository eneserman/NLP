"""Conformer encoder (Gulati et al., 2020) in plain torch, for the models we train from scratch.

Input: log-mel frames at 100 per second. A 2-D convolution front end subsamples time by
`time_stride` (2 -> 50 frames/s for CTC and attention models, so that CTC has at least one frame
per character; 4 -> 25 frames/s for the transducer, which can emit several symbols per frame).
Then `n_layers` Conformer blocks: half-step feed-forward, self-attention (sinusoidal absolute
positions, torch scaled_dot_product_attention with a padding mask), convolution module,
half-step feed-forward, layer norm.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def lengths_to_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """True at valid positions, shape (B, T)."""
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


def sinusoids(T: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Subsampling(nn.Module):
    """Two 3x3 convolutions with stride 2 in frequency; time stride (2, 1) or (2, 2) -> x2 or x4."""

    def __init__(self, n_mels: int, d_model: int, time_stride: int, channels: int = 64):
        super().__init__()
        if time_stride not in (2, 4):
            raise ValueError("time_stride must be 2 or 4")
        s2 = 2 if time_stride == 4 else 1
        self.strides = (2, s2)
        self.convs = nn.ModuleList([nn.Conv2d(1, channels, 3, stride=(2, 2), padding=1),
                                    nn.Conv2d(channels, channels, 3, stride=(s2, 2), padding=1)])
        f = n_mels
        for _ in range(2):
            f = (f + 2 - 3) // 2 + 1
        self.out = nn.Linear(channels * f, d_model)

    @staticmethod
    def _out_len(lengths: torch.Tensor, s: int) -> torch.Tensor:
        return (lengths + 2 - 3) // s + 1

    def out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        for s in self.strides:
            lengths = self._out_len(lengths, s)
        return lengths

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # Frames after each utterance's end are set to 0 before and after every convolution, so the
        # output of an utterance does not depend on how much padding the batch has.
        x = x.masked_fill(~lengths_to_mask(lengths, x.shape[1])[:, :, None], 0.0).unsqueeze(1)
        for conv, s in zip(self.convs, self.strides):
            x = F.silu(conv(x))                            # (B, C, T', F')
            lengths = self._out_len(lengths, s)
            x = x.masked_fill(~lengths_to_mask(lengths, x.shape[2])[:, None, :, None], 0.0)
        B, C, T, Fq = x.shape
        x = self.out(x.permute(0, 2, 1, 3).reshape(B, T, C * Fq))
        return x, lengths


class FeedForward(nn.Module):
    def __init__(self, d: int, ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(ff, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.heads, self.dropout = heads, dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        attn_mask = mask[:, None, None, :]                  # keys that are padding are not attended
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = self.norm(x).masked_fill(~mask[..., None], 0.0).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1).masked_fill(~mask[:, None, :], 0.0)   # padding stays 0 for the depthwise conv
        y = self.dw(y).masked_fill(~mask[:, None, :], 0.0)
        y = self.pw2(F.silu(self.bn(y)))
        return self.drop(y.transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, kernel: int, dropout: float):
        super().__init__()
        self.ff1 = FeedForward(d, ff, dropout)
        self.att = SelfAttention(d, heads, dropout)
        self.conv = ConvModule(d, kernel, dropout)
        self.ff2 = FeedForward(d, ff, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        x = x + 0.5 * self.ff1(x)
        x = x + self.att(x, mask)
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, d_model: int = 256, n_layers: int = 12, heads: int = 4,
                 ff: int = 1024, kernel: int = 15, dropout: float = 0.1, time_stride: int = 2):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.sub = Subsampling(n_mels, d_model, time_stride)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(ConformerBlock(d_model, heads, ff, kernel, dropout) for _ in range(n_layers))
        self.d_model = d_model

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """x (B, T, n_mels), lengths (B,) -> (B, T', d_model), lengths', mask (B, T')."""
        x, lengths = self.sub(x, lengths)
        mask = lengths_to_mask(lengths, x.shape[1])
        x = self.drop(x * math.sqrt(self.d_model) + sinusoids(x.shape[1], self.d_model, x.device))
        for blk in self.blocks:
            x = blk(x, mask)
        return x, lengths, mask
