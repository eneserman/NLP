"""Acoustic features in plain torch (no extra audio library), for the models we train ourselves.

* log_mel: 25 ms Hann window, 10 ms hop, 80 mel bands (HTK mel scale), log(power + 1e-6).
* mfcc: 13 cepstra (DCT-II of 40 log-mel bands) + deltas + delta-deltas = 39 dimensions,
  the classic GMM-HMM front end.
* cmvn: mean and variance normalization per utterance.
* spec_augment: time and frequency masks (training only).
"""
from __future__ import annotations

import math

import torch

SR = 16000
N_FFT = 400
HOP = 160


def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int = N_FFT, sr: int = SR, fmin: float = 20.0, fmax: float | None = None) -> torch.Tensor:
    """Triangular filters, shape (n_mels, n_fft // 2 + 1)."""
    fmax = fmax or sr / 2
    mels = torch.linspace(float(hz_to_mel(torch.tensor(fmin))), float(hz_to_mel(torch.tensor(fmax))), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    lo, ce, hi = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    up = (bins[None] - lo) / (ce - lo)
    down = (hi - bins[None]) / (hi - ce)
    return torch.clamp(torch.minimum(up, down), min=0.0)


def power_spectrum(wave: torch.Tensor) -> torch.Tensor:
    """wave (..., samples) -> (..., frames, n_fft // 2 + 1)."""
    win = torch.hann_window(N_FFT, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(wave, N_FFT, HOP, N_FFT, win, center=True, return_complex=True)
    return spec.abs().pow(2).transpose(-1, -2)


def log_mel(wave: torch.Tensor, n_mels: int = 80) -> torch.Tensor:
    fb = mel_filterbank(n_mels).to(wave.device, wave.dtype)
    return torch.log(power_spectrum(wave) @ fb.T + 1e-6)


def dct_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """Orthonormal DCT-II, shape (n_in, n_out)."""
    n = torch.arange(n_in, dtype=torch.float64)
    k = torch.arange(n_out, dtype=torch.float64)
    m = torch.cos(math.pi / n_in * (n[:, None] + 0.5) * k[None])
    m[:, 0] *= 1.0 / math.sqrt(2.0)
    return (m * math.sqrt(2.0 / n_in)).float()


def deltas(x: torch.Tensor, n: int = 2) -> torch.Tensor:
    """Regression deltas over the time axis (-2), edges repeated."""
    pad = torch.cat([x[..., :1, :].expand(*x.shape[:-2], n, x.shape[-1]), x,
                     x[..., -1:, :].expand(*x.shape[:-2], n, x.shape[-1])], dim=-2)
    T = x.shape[-2]
    num = sum(k * (pad[..., n + k:n + k + T, :] - pad[..., n - k:n - k + T, :]) for k in range(1, n + 1))
    return num / (2 * sum(k * k for k in range(1, n + 1)))


def mfcc(wave: torch.Tensor, n_ceps: int = 13, n_mels: int = 40) -> torch.Tensor:
    c = log_mel(wave, n_mels) @ dct_matrix(n_mels, n_ceps).to(wave.device)
    d = deltas(c)
    return torch.cat([c, d, deltas(d)], dim=-1)


def cmvn(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-utterance mean/variance normalization over time; x (frames, dims)."""
    return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + eps)


def num_frames(samples: int) -> int:
    return samples // HOP + 1  # torch.stft with center=True


def spec_augment(x: torch.Tensor, lengths: torch.Tensor, n_freq: int = 2, f: int = 27,
                 n_time: int = 5, t_frac: float = 0.05) -> torch.Tensor:
    """Masks on a batch (B, frames, dims); masked values are set to 0 (the mean after CMVN).
    Time masks are at most t_frac of each utterance's own length. Uses the global torch RNG."""
    B, T, D = x.shape
    dev = x.device
    keep = torch.ones(B, T, D, dtype=torch.bool, device=dev)
    d_idx = torch.arange(D, device=dev)[None, :]
    t_idx = torch.arange(T, device=dev)[None, :]
    for _ in range(n_freq):
        w = torch.randint(0, f + 1, (B, 1), device=dev)
        s = (torch.rand(B, 1, device=dev) * (D - w).clamp(min=1)).long()
        keep &= ~((d_idx >= s) & (d_idx < s + w))[:, None, :]
    tmax = (lengths.to(dev).float() * t_frac).long().clamp(min=1)[:, None]
    for _ in range(n_time):
        w = (torch.rand(B, 1, device=dev) * (tmax + 1)).long()
        s = (torch.rand(B, 1, device=dev) * (lengths.to(dev)[:, None] - w).clamp(min=1)).long()
        keep &= ~((t_idx >= s) & (t_idx < s + w))[:, :, None]
    return x * keep
