"""Arc Sound reference implementation (numpy only). The C library (src/arcsound.c) follows this file step for step.

Pipeline: 16 kHz mono PCM -> log-mel (64 bins, 100 frames/s) -> causal conv stem (10 frames/s) -> streaming encoder
(recurrent blocks with carried state) -> causal local attention -> per-category presence / onset / offset
probabilities at 10 frames/s -> event decoder (start, end, category, score).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ weights
def load_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n]); base = 8 + n; out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        assert v["dtype"] == "F32", (k, v["dtype"])
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(raw[base + a:base + b], dtype="<f4").reshape(v["shape"]).copy()
    return out


# ------------------------------------------------------------------ math helpers
def _erf(x):
    # Abramowitz & Stegun 7.1.26 is too coarse for parity; use the float64 series/continued-fraction via np.vectorize(math.erf)
    import math
    return np.vectorize(math.erf, otypes=[np.float64])(x)


def gelu(x):
    return (0.5 * x * (1.0 + _erf(x.astype(np.float64) / np.sqrt(2.0)))).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w


def layernorm(x, w, b, eps):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def lin(x, W, b=None):
    y = x @ W.T
    return y + b if b is not None else y


# ------------------------------------------------------------------ front end
def logmel(pcm, W, meta):
    """pcm float32 in [-1, 1] at 16 kHz -> (frames, n_mels) natural-log mel power; frame t is centred on sample t*hop."""
    hop, nfft = meta["hop"], meta["n_fft"]
    y = np.pad(np.asarray(pcm, np.float32), (nfft // 2, nfft // 2))
    n = 1 + len(pcm) // hop
    idx = np.arange(nfft)[None] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * W["fe.window"], axis=1)) ** 2
    m = np.log(spec.astype(np.float32) @ W["fe.mel"].T + meta["log_eps"])
    return m.astype(np.float16).astype(np.float32)        # features are rounded to half precision (part of the model)


def silence(pcm, db):
    """per 0.1 s frame: True where the audio is digitally silent (muted / cut)."""
    hop = 1600; n = len(pcm) // hop
    r = np.sqrt(np.mean(np.asarray(pcm[: n * hop], np.float64).reshape(n, hop) ** 2, 1))
    return 20 * np.log10(r + 1e-9) < db


def conv2d(x, w, b, st):
    """x (Cin, T, M) already padded; w (Cout, Cin, kt, km); stride st=(st_t, st_m) -> (Cout, T', M')."""
    Ci, T, M = x.shape; Co, _, kt, km = w.shape
    To, Mo = (T - kt) // st[0] + 1, (M - km) // st[1] + 1
    s = x.strides
    win = np.lib.stride_tricks.as_strided(x, (Ci, To, Mo, kt, km), (s[0], s[1] * st[0], s[2] * st[1], s[1], s[2]))
    return np.einsum("ctmab,ocab->otm", win, w, optimize=True) + b[:, None, None]


# ------------------------------------------------------------------ model
class ArcSound:
    def __init__(self, model_dir):
        d = Path(model_dir)
        self.W = load_safetensors(d / "arcsound.safetensors")
        self.meta = json.load(open(d / "arcsound_meta.json"))
        self.cats = self.meta["categories"]

    def probs(self, mel):
        """(T100, n_mels) log-mel -> P, O, F (T10, C) presence / onset / offset probabilities."""
        W, mt = self.W, self.meta; eps, D = mt["rms_eps"], mt["d_model"]
        x = ((mel - W["fe.mean"]) / W["fe.std"]).astype(np.float32)
        x = np.pad(x, ((2, 0), (1, 1)))[None]                                   # time: past only
        x = gelu(conv2d(x, W["stem.c1.w"], W["stem.c1.b"], (2, 2)))
        x = gelu(conv2d(np.pad(x, ((0, 0), (4, 0), (1, 1))), W["stem.c2.w"], W["stem.c2.b"], (5, 2)))
        Ch, T, M = x.shape
        h = lin(x.transpose(1, 0, 2).reshape(T, Ch * M), W["stem.proj.w"], W["stem.proj.b"])
        h = lin(layernorm(h, W["enc.in_norm.w"], W["enc.in_norm.b"], mt["ln_eps"]), W["enc.in.w"], W["enc.in.b"])
        H = D // mt["head_dim"]; N, Pd = mt["d_state"], mt["head_dim"]
        for i in range(mt["n_layers"]):
            q = f"enc.{i}."; g = lambda n: W[q + n]
            h = h * g("scale") + g("shift")
            nu = rms(h, g("seq.norm.w"), eps)
            dt = softplus(lin(nu, g("seq.step.w"), g("seq.step.b")))            # (T, H)
            alpha = np.exp(-dt * g("seq.decay")); lam = sigmoid(lin(nu, g("seq.mix.w"), g("seq.mix.b")))
            gamma = lam * dt; beta = (1 - lam) * dt * alpha
            Bm = rms(lin(nu, g("seq.wb.w"), g("seq.wb.b")).reshape(T, H, N), g("seq.nb.w"), eps) + g("seq.bb")
            Cm = rms(lin(nu, g("seq.wc.w"), g("seq.wc.b")).reshape(T, H, N), g("seq.nc.w"), eps) + g("seq.bc")
            xv = lin(nu, g("seq.wx.w"), g("seq.wx.b")).reshape(T, H, Pd)
            th = lin(nu, g("seq.rot.w"), g("seq.rot.b")).reshape(T, H, N // 2) * dt[..., None]
            cs, sn = np.cos(th), np.sin(th)
            st = np.zeros((H, N, Pd), np.float32); prev = np.zeros((H, N, Pd), np.float32); y = np.empty((T, H, Pd), np.float32)
            for t in range(T):
                S = Bm[t][:, :, None] * xv[t][:, None, :]
                pre = alpha[t][:, None, None] * st + beta[t][:, None, None] * prev
                a, b = pre[:, 0::2], pre[:, 1::2]; c, s = cs[t][:, :, None], sn[t][:, :, None]
                st = np.empty_like(pre); st[:, 0::2] = c * a - s * b; st[:, 1::2] = s * a + c * b
                st = st + gamma[t][:, None, None] * S; prev = S
                y[t] = np.einsum("hn,hnp->hp", Cm[t], st)
            gate = lin(nu, g("seq.gate.w"), g("seq.gate.b")); gate = gate * sigmoid(gate)
            h = h + lin(y.reshape(T, D) * gate, g("seq.out.w"), g("seq.out.b"))
            f = rms(h, g("ff.norm.w"), eps)
            h = h + lin(gelu(lin(f, g("ff.up.w"), g("ff.up.b"))), g("ff.down.w"), g("ff.down.b"))
        h = h + lin(gelu(lin(h, W["enc.adapt.down"])), W["enc.adapt.up"])
        h = rms(h, W["enc.out_norm.w"], eps)
        nh, win = mt["n_heads"], mt["attn_window"]; dh = D // nh
        for j in range(mt["attn_layers"]):
            q = f"att.{j}."
            qkv = lin(rms(h, W[q + "norm.w"], eps), W[q + "qkv.w"], W[q + "qkv.b"]).reshape(T, 3, nh, dh)
            o = np.empty((T, nh, dh), np.float32)
            for t in range(T):
                k0 = max(0, t - win + 1)
                sc = np.einsum("hd,khd->hk", qkv[t, 0], qkv[k0:t + 1, 1]) / np.sqrt(dh)
                sc = np.exp(sc - sc.max(1, keepdims=True)); sc /= sc.sum(1, keepdims=True)
                o[t] = np.einsum("hk,khd->hd", sc, qkv[k0:t + 1, 2])
            h = h + lin(o.reshape(T, D), W[q + "o.w"], W[q + "o.b"])
            f = rms(h, W[q + "ff.norm.w"], eps)
            h = h + lin(gelu(lin(f, W[q + "ff.up.w"], W[q + "ff.up.b"])), W[q + "ff.down.w"], W[q + "ff.down.b"])
        z = sigmoid(lin(h, W["head.w"], W["head.b"])); C = len(self.cats)
        return z[:, :C], z[:, C:2 * C], z[:, 2 * C:]

    def events(self, pcm):
        mel = logmel(pcm, self.W, self.meta)
        P, O, F = self.probs(mel)
        return decode(P, O, F, self.meta, silence(pcm, self.meta["decode"]["silence_db"]))


def decode(P, O, F, meta, sil=None):
    """Hysteresis per category on 3-frame-smoothed presence, gap merge, minimum duration (decode.per_cat overrides gap /
    min_dur per category), then onset / offset
    refinement from the boundary heads for impulsive categories. Times in seconds (10 frames/s)."""
    dc = meta["decode"]; cats = meta["categories"]; imp = set(meta["impulsive"]); T = len(P)
    if sil is not None:
        P = P.copy(); m = sil[:T]; P[: len(m)][m] = 0
    ev = []
    for j, c in enumerate(cats):
        on = dc["on"][c]; off = dc["off_ratio"] * on
        pc = (dc.get("per_cat") or {}).get(c) or {}
        gap, min_dur = pc.get("gap", dc["gap"]), pc.get("min_dur", dc["min_dur"])
        p = np.convolve(P[:, j], np.ones(3) / 3, "same")
        segs, act, t0 = [], False, 0
        for t in range(T + 1):
            v = p[t] if t < T else 0
            if not act and v >= on:
                act, t0 = True, t
            elif act and v < off:
                act = False; segs.append([t0, t])
        merged = []
        for s in segs:
            if merged and (s[0] - merged[-1][1]) / 10 <= gap:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        for a, b in merged:
            if (b - a) / 10 < min_dur:
                continue
            s, e = a / 10, b / 10
            if c in imp:
                lo, hi = max(a - 10, 0), min(a + 10, T)
                if O[lo:hi, j].max() > 0.3:
                    s = (lo + int(O[lo:hi, j].argmax())) / 10
                lo, hi = max(b - 15, 0), min(b + 15, T)
                if F[lo:hi, j].max() > 0.3:
                    e = max((lo + int(F[lo:hi, j].argmax())) / 10, s + 0.3)
            ev.append(dict(start=round(s, 2), end=round(e, 2), category=c, score=round(float(P[a:b, j].max()), 3)))
    return sorted(ev, key=lambda e: (e["start"], e["category"]))
