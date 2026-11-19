"""The neural course baselines that we train from scratch (configs/train_scratch.yaml).

All three share the Conformer encoder (hrasr/conformer.py) and character targets (hrasr/vocab.py).

* CTCModel:   encoder -> linear -> CTC. Greedy decoding (best path, repeats merged, blanks removed).
* AEDModel:   encoder -> Transformer decoder (attention encoder-decoder) with an extra CTC head on the
              encoder. Training loss: ctc_weight * CTC + (1 - ctc_weight) * cross-entropy (label
              smoothing). Decoding: attention beam search, then the n-best list is rescored with
              (1 - lam) * attention log-prob + lam * CTC log-prob, divided by the length (lam is chosen
              on dev).
* RNNTModel:  encoder -> joint network with an LSTM prediction network (transducer). Greedy decoding.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from hrasr.conformer import ConformerEncoder, sinusoids
from hrasr.transducer import greedy_decode, rnnt_loss


def ctc_greedy(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int = 0) -> list[list[int]]:
    best = log_probs.argmax(-1).cpu()
    out = []
    for seq, n in zip(best, lengths.cpu().tolist()):
        seq = seq[:n].tolist()
        out.append([s for i, s in enumerate(seq) if s != blank and (i == 0 or s != seq[i - 1])])
    return out


class CTCModel(nn.Module):
    def __init__(self, vocab_size: int, enc: dict):
        super().__init__()
        self.encoder = ConformerEncoder(**enc)
        self.head = nn.Linear(self.encoder.d_model, vocab_size)

    def forward(self, x, lengths):
        h, hl, _ = self.encoder(x, lengths)
        return self.head(h).float().log_softmax(-1), hl

    def loss(self, x, lengths, targets, t_len):
        lp, hl = self(x, lengths)
        return F.ctc_loss(lp.transpose(0, 1), targets, hl, t_len, blank=0, reduction="mean", zero_infinity=True)

    @torch.no_grad()
    def decode(self, x, lengths, **_):
        lp, hl = self(x, lengths)
        return ctc_greedy(lp, hl)


class AEDModel(nn.Module):
    def __init__(self, vocab_size: int, enc: dict, dec_layers: int = 6, dec_ff: int = 1024, dec_heads: int = 4,
                 dropout: float = 0.1, ctc_weight: float = 0.3, label_smoothing: float = 0.1):
        super().__init__()
        self.encoder = ConformerEncoder(**enc)
        d = self.encoder.d_model
        self.V = vocab_size                       # characters, blank (0) and unk (1)
        self.sos, self.eos = vocab_size, vocab_size + 1
        self.n_out = vocab_size + 2
        self.ctc_head = nn.Linear(d, vocab_size)
        self.embed = nn.Embedding(self.n_out, d)
        layer = nn.TransformerDecoderLayer(d, dec_heads, dec_ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.n_out)
        self.d = d
        self.ctc_weight, self.label_smoothing = ctc_weight, label_smoothing

    def encode(self, x, lengths):
        h, hl, mask = self.encoder(x, lengths)
        return h, hl, mask

    def decode_logits(self, tokens, h, mask):
        L = tokens.shape[1]
        y = self.embed(tokens) * math.sqrt(self.d) + sinusoids(L, self.d, tokens.device)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)
        y = self.decoder(y, h, tgt_mask=causal, memory_key_padding_mask=~mask)
        return self.out(self.norm(y)).float()

    def loss(self, x, lengths, targets, t_len):
        h, hl, mask = self.encode(x, lengths)
        lp = self.ctc_head(h).float().log_softmax(-1)
        ctc = F.ctc_loss(lp.transpose(0, 1), targets, hl, t_len, blank=0, reduction="mean", zero_infinity=True)
        B, U = targets.shape
        pos = torch.arange(U + 1, device=targets.device)[None, :]
        dec_in = torch.cat([torch.full((B, 1), self.sos, device=targets.device, dtype=torch.long), targets], 1)
        gold = torch.cat([targets, torch.zeros(B, 1, device=targets.device, dtype=torch.long)], 1)
        gold = torch.where(pos == t_len[:, None], torch.full_like(gold, self.eos), gold)
        gold = torch.where(pos > t_len[:, None], torch.full_like(gold, -100), gold)
        logits = self.decode_logits(dec_in, h, mask)
        ce = F.cross_entropy(logits.reshape(-1, self.n_out), gold.reshape(-1), ignore_index=-100,
                             label_smoothing=self.label_smoothing)
        return self.ctc_weight * ctc + (1 - self.ctc_weight) * ce

    @torch.no_grad()
    def beam_search(self, h, hl, mask, beam: int, max_len: torch.Tensor) -> list[list[tuple[list[int], float]]]:
        """-> per utterance, up to `beam` finished hypotheses (tokens, summed attention log-prob)."""
        B, T, D = h.shape
        dev = h.device
        K = beam
        hk = h.repeat_interleave(K, 0)
        mk = mask.repeat_interleave(K, 0)
        tokens = torch.full((B * K, 1), self.sos, dtype=torch.long, device=dev)
        scores = torch.full((B, K), float("-inf"), device=dev)
        scores[:, 0] = 0.0
        finished: list[list[tuple[list[int], float]]] = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for step in range(int(max_len.max()) + 1):
            lp = self.decode_logits(tokens, hk, mk)[:, -1].log_softmax(-1)          # (B*K, n_out)
            lp[:, 0] = float("-inf")                                                # never blank or unk
            lp[:, 1] = float("-inf")
            lp[:, self.sos] = float("-inf")
            cand = (scores[:, :, None] + lp.view(B, K, -1)).view(B, -1)
            top, idx = cand.topk(2 * K, dim=1)
            new_tokens, new_scores = [], torch.full((B, K), float("-inf"), device=dev)
            src_rows = torch.zeros(B, K, dtype=torch.long, device=dev)
            next_sym = torch.zeros(B, K, dtype=torch.long, device=dev)
            top, idx = top.cpu(), idx.cpu()
            for b in range(B):
                if bool(done[b]):
                    continue
                kept = 0
                at_max = step == int(max_len[b])
                for rank, (s, i) in enumerate(zip(top[b].tolist(), idx[b].tolist())):
                    if s == float("-inf") or kept == K:
                        break
                    k, sym = divmod(i, self.n_out)
                    if sym == self.eos or at_max:
                        if rank < K or at_max:          # an ending counts only if it is in the top K
                            finished[b].append((tokens[b * K + k, 1:].tolist(), s))
                        continue
                    src_rows[b, kept] = b * K + k
                    next_sym[b, kept] = sym
                    new_scores[b, kept] = s
                    kept += 1
                best_done = max((f[1] for f in finished[b]), default=float("-inf"))
                best_alive = float(new_scores[b].max())
                # raw scores only decrease, so with K endings no alive hypothesis can beat the best one
                if at_max or kept == 0 or (len(finished[b]) >= K and best_alive < best_done):
                    done[b] = True
            if bool(done.all()):
                break
            tokens = torch.cat([tokens[src_rows.view(-1)], next_sym.view(-1, 1)], dim=1)
            scores = new_scores.masked_fill(done[:, None], float("-inf"))
        return finished

    @torch.no_grad()
    def nbest(self, x, lengths, beam: int = 1, max_len_ratio: float = 0.6) -> list[list[tuple[list[int], float, float]]]:
        """-> per utterance: (tokens, attention log-prob, CTC log-prob) for each finished hypothesis."""
        h, hl, mask = self.encode(x, lengths)
        max_len = (hl.float() * max_len_ratio).long() + 5
        found = self.beam_search(h, hl, mask, beam, max_len)
        lp = self.ctc_head(h).float().log_softmax(-1)
        out = []
        for b, nb in enumerate(found):
            if not nb:
                out.append([])
                continue
            seqs = [s for s, _ in nb]
            n = len(seqs)
            tgt = torch.zeros(n, max(1, max(len(s) for s in seqs)), dtype=torch.long, device=h.device)
            for i, s in enumerate(seqs):
                if s:
                    tgt[i, :len(s)] = torch.tensor(s, device=h.device)
            tl = torch.tensor([len(s) for s in seqs], device=h.device)
            ctc_ll = -F.ctc_loss(lp[b:b + 1].expand(n, -1, -1).transpose(0, 1), tgt, hl[b:b + 1].expand(n), tl,
                                 blank=0, reduction="none", zero_infinity=False)
            out.append([(s, a, float(c)) for (s, a), c in zip(nb, ctc_ll.tolist())])
        return out

    @staticmethod
    def choose(nb: list[tuple[list[int], float, float]], ctc_lambda: float) -> list[int]:
        """Best hypothesis by ((1 - lam) * attention + lam * CTC) / (length + 1). lam = 0: attention only."""
        if not nb:
            return []
        def total(c):
            s, a, ctc = c
            return ((1 - ctc_lambda) * a + (ctc_lambda * ctc if ctc_lambda else 0.0)) / (len(s) + 1)
        return max(nb, key=total)[0]

    @torch.no_grad()
    def decode(self, x, lengths, beam: int = 1, ctc_lambda: float = 0.0, max_len_ratio: float = 0.6, **_):
        return [self.choose(nb, ctc_lambda) for nb in self.nbest(x, lengths, beam, max_len_ratio)]


class Predictor(nn.Module):
    """LSTM prediction network; the blank index (0) is the start symbol."""

    def __init__(self, vocab_size: int, embed: int, hidden: int, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.layers, self.hidden = layers, hidden
        self.drop = nn.Dropout(dropout)

    def forward(self, targets):
        """targets (B, U) -> outputs for the prefixes of length 0..U, shape (B, U+1, H)."""
        start = torch.zeros(targets.shape[0], 1, dtype=torch.long, device=targets.device)
        y, _ = self.lstm(self.drop(self.embed(torch.cat([start, targets], 1))))
        return y

    def initial(self, B: int, device):
        state = (torch.zeros(self.layers, B, self.hidden, device=device),
                 torch.zeros(self.layers, B, self.hidden, device=device))
        return self.step(torch.zeros(B, dtype=torch.long, device=device), state)

    def step(self, tokens, state):
        y, state = self.lstm(self.embed(tokens)[:, None, :], state)
        return y[:, 0], state


class Joint(nn.Module):
    def __init__(self, enc_dim: int, pred_dim: int, hidden: int, vocab_size: int):
        super().__init__()
        self.enc = nn.Linear(enc_dim, hidden)
        self.pred = nn.Linear(pred_dim, hidden)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, enc, pred):
        """Full lattice: enc (B, T, E), pred (B, U+1, P) -> (B, T, U+1, V). One frame: (B, E), (B, P) -> (B, V)."""
        if enc.dim() == 3:
            return self.out(torch.tanh(self.enc(enc)[:, :, None, :] + self.pred(pred)[:, None, :, :]))
        return self.out(torch.tanh(self.enc(enc) + self.pred(pred)))


class RNNTModel(nn.Module):
    def __init__(self, vocab_size: int, enc: dict, pred_embed: int = 256, pred_hidden: int = 320,
                 pred_layers: int = 1, joint_hidden: int = 320, dropout: float = 0.1):
        super().__init__()
        self.encoder = ConformerEncoder(**enc)
        self.predictor = Predictor(vocab_size, pred_embed, pred_hidden, pred_layers, dropout)
        self.joint = Joint(self.encoder.d_model, pred_hidden, joint_hidden, vocab_size)

    def loss(self, x, lengths, targets, t_len):
        h, hl, _ = self.encoder(x, lengths)
        p = self.predictor(targets)
        lp = self.joint(h, p).float().log_softmax(-1)
        return rnnt_loss(lp, targets, hl, t_len, blank=0, reduction="mean")

    @torch.no_grad()
    def decode(self, x, lengths, max_symbols: int = 10, **_):
        h, hl, _ = self.encoder(x, lengths)
        return greedy_decode(h, hl, self.predictor, self.joint, blank=0, max_symbols=max_symbols)


def build(kind: str, vocab_size: int, cfg: dict) -> nn.Module:
    if kind == "ctc":
        return CTCModel(vocab_size, cfg["encoder"])
    if kind == "aed":
        return AEDModel(vocab_size, cfg["encoder"], **cfg.get("decoder", {}))
    if kind == "rnnt":
        return RNNTModel(vocab_size, cfg["encoder"], **cfg.get("transducer", {}))
    raise ValueError(kind)
