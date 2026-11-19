"""Classical baseline: MFCC + GMM-HMM, character (grapheme) units, Viterbi training.

Model
* One 3-state left-to-right HMM per character (Croatian spelling is close to phonemic) and one
  for silence ("sil"). Each state has a diagonal-covariance Gaussian mixture over 39-dim MFCC
  (13 + deltas + delta-deltas, per-utterance CMVN; hrasr/features.py).
* Training: flat start (each utterance's frames split evenly over its states), then iterations of
  Viterbi alignment and re-estimation; the number of Gaussians per state is doubled on a schedule
  (1 -> 2 -> ... -> max). Silence is optional at the start, the end and between words.
* Decoding: Viterbi search over a loop of all character HMMs and silence with a character bigram
  language model (tokens: characters, word boundary " ", sentence start and end, add-k smoothing).
  A word boundary is either the silence model or a transition without frames (no pause).
  LM weight and insertion penalty are chosen on dev.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

SIL = "sil"
N_STATES = 3
LOG_2PI = math.log(2 * math.pi)


class CharBigramLM:
    """log P(next | previous) over characters, " " (word boundary), "<s>" and "</s>"; add-k smoothing."""

    def __init__(self, texts, chars: list[str], k: float = 0.1):
        self.tokens = list(chars) + [" ", "</s>"]
        self.ctx = list(chars) + [" ", "<s>"]
        counts = Counter()
        for t in texts:
            seq = ["<s>"] + list(t) + ["</s>"]
            counts.update(zip(seq[:-1], seq[1:]))
        self.logp = {}
        for a in self.ctx:
            total = sum(counts[(a, b)] for b in self.tokens) + k * len(self.tokens)
            for b in self.tokens:
                self.logp[(a, b)] = math.log((counts[(a, b)] + k) / total)

    def __call__(self, a: str, b: str) -> float:
        return self.logp[(a, b)]


class GMMHMM:
    def __init__(self, chars: list[str], dim: int = 39, var_floor: float = 1e-3):
        self.units = [c for c in chars if c != " "] + [SIL]
        self.uidx = {u: i for i, u in enumerate(self.units)}
        self.S = len(self.units) * N_STATES
        self.dim, self.var_floor = dim, var_floor
        self.means = np.zeros((self.S, 1, dim))
        self.vars = np.ones((self.S, 1, dim))
        self.logw = np.zeros((self.S, 1))
        self.log_self = np.full(self.S, math.log(0.6))
        self.log_next = np.full(self.S, math.log(0.4))

    # ------------------------------------------------------------ likelihoods
    def frame_loglik(self, X: np.ndarray) -> np.ndarray:
        """X (T, D) -> (T, S): log sum_k w_k N(x; mu_k, diag var_k)."""
        X = np.asarray(X, dtype=np.float64)
        S, K, D = self.means.shape
        inv = 1.0 / self.vars                                                    # (S, K, D)
        const = self.logw - 0.5 * (D * LOG_2PI + np.log(self.vars).sum(-1) + (self.means ** 2 * inv).sum(-1))
        quad = -0.5 * (X ** 2) @ inv.reshape(S * K, D).T + X @ (self.means * inv).reshape(S * K, D).T
        ll = quad.reshape(len(X), S, K) + const[None]
        m = ll.max(-1, keepdims=True)
        return (m + np.log(np.exp(ll - m).sum(-1, keepdims=True)))[..., 0]

    # ------------------------------------------------------------ training graph and alignment
    def utterance_units(self, text: str) -> tuple[list[int], list[bool]]:
        """Unit ids with optional flags: sil? word sil? word ... sil? (sil optional everywhere)."""
        units, optional = [self.uidx[SIL]], [True]
        for w in text.split():
            for ch in w:
                if ch in self.uidx:
                    units.append(self.uidx[ch])
                    optional.append(False)
            units.append(self.uidx[SIL])
            optional.append(True)
        return units, optional

    def align(self, ll: np.ndarray, units: list[int], optional: list[bool]) -> np.ndarray | None:
        """Viterbi alignment of frames to the utterance's state chain -> global state id per frame,
        or None if the utterance has fewer frames than its required states."""
        T = ll.shape[0]
        states = np.array([u * N_STATES + s for u in units for s in range(N_STATES)])
        J = len(states)
        first = np.zeros(J, dtype=bool)
        first[::N_STATES] = True
        # skip edge into the first state of unit i from the last state of unit i-2 when unit i-1 is optional
        skip = np.zeros(J, dtype=bool)
        for i in range(2, len(units)):
            if optional[i - 1]:
                skip[i * N_STATES] = True
        neg = -1e30
        delta = np.full(J, neg)
        delta[0] = ll[0, states[0]]
        if optional[0] and len(units) > 1:
            delta[N_STATES] = ll[0, states[N_STATES]]
        bp = np.zeros((T, J), dtype=np.int8)
        ls, ln = self.log_self[states], self.log_next[states]
        for t in range(1, T):
            stay = delta + ls
            prev = np.full(J, neg)
            prev[1:] = delta[:-1] + ln[:-1]
            sk = np.full(J, neg)
            idx = np.nonzero(skip)[0]
            sk[idx] = delta[idx - N_STATES - 1] + ln[idx - N_STATES - 1]
            cand = np.stack([stay, prev, sk])
            choice = cand.argmax(0)
            delta = cand[choice, np.arange(J)] + ll[t, states]
            bp[t] = choice
        ends = [J - 1] + ([J - 1 - N_STATES] if optional[-1] and len(units) > 1 else [])
        j = max(ends, key=lambda e: delta[e])
        if delta[j] < neg / 2:
            return None
        path = np.empty(T, dtype=np.int64)
        for t in range(T - 1, -1, -1):
            path[t] = states[j]
            c = bp[t, j]
            j = j if c == 0 else (j - 1 if c == 1 else j - N_STATES - 1)
        return path

    def flat_start_alignment(self, T: int, units: list[int], optional: list[bool]) -> np.ndarray | None:
        """Even split of the frames over the states of the utterance: the edge silences and the
        characters (as in a Kaldi flat start, the silences between words are left out)."""
        keep = [i for i, o in enumerate(optional) if not o or i in (0, len(units) - 1)]
        states = [units[i] * N_STATES + s for i in keep for s in range(N_STATES)]
        if not states or T < len(states):
            return None
        bounds = np.linspace(0, T, len(states) + 1).astype(int)
        path = np.empty(T, dtype=np.int64)
        for k, s in enumerate(states):
            path[bounds[k]:bounds[k + 1]] = s
        return path

    # ------------------------------------------------------------ re-estimation
    def reestimate(self, feats: list[np.ndarray], paths: list[np.ndarray]) -> None:
        """M-step from hard alignments: each frame belongs to its state; within the state, to its best
        Gaussian (k-means style). Transition probabilities from state occupancy."""
        S, K, D = self.means.shape
        n = np.zeros((S, K))
        s1 = np.zeros((S, K, D))
        s2 = np.zeros((S, K, D))
        stay, leave = np.zeros(S), np.zeros(S)
        for X, p in zip(feats, paths):
            X = np.asarray(X, dtype=np.float64)
            if K > 1:
                comp = self._best_component(X, p)
            else:
                comp = np.zeros(len(X), dtype=np.int64)
            np.add.at(n, (p, comp), 1.0)
            np.add.at(s1, (p, comp), X)
            np.add.at(s2, (p, comp), X ** 2)
            same = p[1:] == p[:-1]
            np.add.at(stay, p[:-1][same], 1.0)
            np.add.at(leave, p[:-1][~same], 1.0)
            leave[p[-1]] += 1.0
        seen = n > 0
        mu = np.where(seen[..., None], s1 / np.maximum(n, 1)[..., None], self.means)
        var = np.where(seen[..., None], s2 / np.maximum(n, 1)[..., None] - mu ** 2, self.vars)
        self.means = mu
        self.vars = np.maximum(var, self.var_floor)
        w = n + 1e-3
        self.logw = np.log(w / w.sum(1, keepdims=True))
        p_stay = (stay + 1.0) / (stay + leave + 2.0)
        self.log_self = np.log(p_stay)
        self.log_next = np.log(1.0 - p_stay)

    def _best_component(self, X: np.ndarray, p: np.ndarray) -> np.ndarray:
        S, K, D = self.means.shape
        mu, var, lw = self.means[p], self.vars[p], self.logw[p]                    # (T, K, D)
        ll = lw - 0.5 * (np.log(var).sum(-1) + ((X[:, None, :] - mu) ** 2 / var).sum(-1))
        return ll.argmax(1)

    def split(self, eps: float = 0.2) -> None:
        """Double the Gaussians: each mean moves by +-eps standard deviations."""
        sd = np.sqrt(self.vars)
        self.means = np.concatenate([self.means + eps * sd, self.means - eps * sd], 1)
        self.vars = np.concatenate([self.vars, self.vars], 1)
        self.logw = np.concatenate([self.logw, self.logw], 1) - math.log(2.0)

    # ------------------------------------------------------------ decoding
    def decode(self, X: np.ndarray, lm: CharBigramLM, lm_weight: float, ins_penalty: float) -> str:
        """Viterbi over the character loop with the bigram LM -> text."""
        ll = self.frame_loglik(X)
        T = ll.shape[0]
        U = len(self.units)
        sil = self.uidx[SIL]
        names = self.units
        tok = [n if n != SIL else " " for n in names]
        # cross-unit transition (exit of u at t-1 -> first state of v at t) and whether a word boundary is crossed
        trans = np.full((U, U), -1e30)
        space = np.zeros((U, U), dtype=bool)
        for u in range(U):
            for v in range(U):
                if u == sil and v == sil:
                    continue
                if u == sil:
                    trans[u, v] = lm_weight * lm(" ", tok[v]) + ins_penalty
                elif v == sil:
                    trans[u, v] = lm_weight * lm(tok[u], " ") + ins_penalty
                else:
                    direct = lm(tok[u], tok[v])
                    via = lm(tok[u], " ") + lm(" ", tok[v])
                    space[u, v] = via > direct
                    trans[u, v] = lm_weight * max(direct, via) + ins_penalty
        start = np.array([lm_weight * lm("<s>", tok[v]) if v != sil else 0.0 for v in range(U)])
        end = np.array([lm_weight * lm(tok[u], "</s>") if u != sil else lm_weight * lm(" ", "</s>") for u in range(U)])
        ls = self.log_self.reshape(U, N_STATES)
        ln = self.log_next.reshape(U, N_STATES)
        llu = ll.reshape(T, U, N_STATES)
        neg = -1e30
        delta = np.full((U, N_STATES), neg)
        delta[:, 0] = start + llu[0, :, 0]
        bp_entry = np.full((T, U), -1, dtype=np.int32)      # source unit when entering state 0, -1 = stayed
        bp_move = np.zeros((T, U, N_STATES), dtype=bool)    # states 1, 2: True = came from previous state
        for t in range(1, T):
            exit_ = delta[:, 2] + ln[:, 2]
            entry = exit_[:, None] + trans                  # (U src, U dst)
            src = entry.argmax(0)
            entry_best = entry[src, np.arange(U)]
            new = np.empty_like(delta)
            stay0 = delta[:, 0] + ls[:, 0]
            take = entry_best > stay0
            new[:, 0] = np.where(take, entry_best, stay0)
            bp_entry[t] = np.where(take, src, -1)
            for s in (1, 2):
                st = delta[:, s] + ls[:, s]
                mv = delta[:, s - 1] + ln[:, s - 1]
                bp_move[t, :, s] = mv > st
                new[:, s] = np.maximum(st, mv)
            delta = new + llu[t]
        final = delta[:, 2] + end
        u = int(final.argmax())
        s = 2
        out: list[str] = []
        for t in range(T - 1, 0, -1):
            if s > 0:
                if bp_move[t, u, s]:
                    s -= 1
                continue
            src = bp_entry[t, u]
            if src >= 0:
                out.append(tok[u])
                if u != sil and src != sil and space[src, u]:
                    out.append(" ")
                u, s = int(src), 2
        out.append(tok[u])
        return " ".join("".join(reversed(out)).split())

    # ------------------------------------------------------------ io
    def state_dict(self) -> dict:
        return {"units": np.array(self.units), "means": self.means, "vars": self.vars, "logw": self.logw,
                "log_self": self.log_self, "log_next": self.log_next}

    @classmethod
    def from_state_dict(cls, d: dict) -> "GMMHMM":
        units = [str(u) for u in d["units"]]
        m = cls([u for u in units if u != SIL] + [" "], dim=d["means"].shape[-1])
        m.means, m.vars, m.logw = d["means"], d["vars"], d["logw"]
        m.log_self, m.log_next = d["log_self"], d["log_next"]
        return m
