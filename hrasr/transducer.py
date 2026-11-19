"""RNN-T (Graves, 2012) loss and greedy decoding in plain torch.

Loss: the forward variable alpha(t, u) over the T x (U+1) lattice,
    alpha(t, u) = logaddexp( alpha(t-1, u) + blank(t-1, u),  alpha(t, u-1) + emit(t, u-1) ),
    loss = -( alpha(T-1, U) + blank(T-1, U) ).
Within one frame t the recursion over u is a running log-sum, so each frame is computed at once:
with a(k) = alpha(t-1, k) + blank(t-1, k) and c(u) = sum_{j<u} emit(t, j),
    alpha(t, u) = c(u) + logcumsumexp_k<=u( a(k) - c(k) ).
This is exact; autograd gives the gradient. tests/test_transducer.py checks it against a sum over
all alignments.
"""
from __future__ import annotations

import torch


def rnnt_loss(log_probs: torch.Tensor, targets: torch.Tensor, t_len: torch.Tensor, u_len: torch.Tensor,
              blank: int = 0, reduction: str = "mean") -> torch.Tensor:
    """log_probs (B, T, U+1, V) log-softmax outputs of the joint network; targets (B, U) int64.
    Returns the negative log-likelihood per utterance (reduction "none"), or its mean / sum."""
    B, T, U1, V = log_probs.shape
    lp = log_probs.float()
    blank_lp = lp[..., blank]                                                   # (B, T, U+1)
    if U1 > 1:
        emit_lp = torch.gather(lp[:, :, :-1, :], 3, targets[:, None, :U1 - 1, None].expand(B, T, U1 - 1, 1)).squeeze(3)
    else:
        emit_lp = lp.new_zeros(B, T, 0)
    neg_inf = torch.finfo(lp.dtype).min / 4
    zeros = lp.new_zeros(B, 1)
    rows = []
    a = torch.cat([zeros, lp.new_full((B, U1 - 1), neg_inf)], dim=1)           # alpha(-1) + "blank" -> start at (0, 0)
    for t in range(T):
        c = torch.cat([zeros, torch.cumsum(emit_lp[:, t], dim=1)], dim=1)       # (B, U+1)
        alpha_t = c + torch.logcumsumexp(a - c, dim=1)
        rows.append(alpha_t)
        a = alpha_t + blank_lp[:, t]
    alpha = torch.stack(rows, dim=1)                                            # (B, T, U+1)
    b = torch.arange(B, device=lp.device)
    ll = alpha[b, t_len - 1, u_len] + blank_lp[b, t_len - 1, u_len]
    nll = -ll
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    return nll.mean()


@torch.no_grad()
def greedy_decode(enc: torch.Tensor, enc_len: torch.Tensor, predictor, joint, blank: int = 0,
                  max_symbols: int = 10) -> list[list[int]]:
    """Batched greedy transducer decoding.

    predictor.step(tokens (B,), state) -> (out (B, H), state); predictor.initial(B) -> (out, state).
    joint(enc_frame (B, H_enc), pred_out (B, H)) -> logits (B, V).
    At each frame, emit the best non-blank symbol while it beats blank (at most max_symbols)."""
    B, T, _ = enc.shape
    hyps: list[list[int]] = [[] for _ in range(B)]
    pred_out, state = predictor.initial(B, enc.device)
    for t in range(T):
        active = enc_len > t
        for _ in range(max_symbols):
            logits = joint(enc[:, t], pred_out)
            best = logits.argmax(-1)
            emit = active & (best != blank)
            if not bool(emit.any()):
                break
            new_out, new_state = predictor.step(best.clamp(min=0), state)
            m = emit[:, None]
            pred_out = torch.where(m, new_out, pred_out)
            state = tuple(torch.where(emit[None, :, None], ns, s) for ns, s in zip(new_state, state))
            for i in torch.nonzero(emit).flatten().tolist():
                hyps[i].append(int(best[i]))
    return hyps
