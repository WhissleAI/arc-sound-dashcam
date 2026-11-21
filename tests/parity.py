"""C library vs the Python reference on one audio file: frame probabilities and events.

  python tests/parity.py AUDIO.wav [--model model] [--cli build/arcsound_cli]
AUDIO: 16 kHz mono 16-bit WAV. Prints the largest probability difference and compares the event lists.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from arcsound.model import ArcSound, decode, logmel, silence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("audio"); ap.add_argument("--model", default="model")
    ap.add_argument("--cli", default="build/arcsound_cli"); a = ap.parse_args()
    y, sr = sf.read(a.audio, dtype="int16"); assert sr == 16000 and y.ndim == 1
    pcm = y.astype(np.float32) / 32768.0
    r = ArcSound(a.model); C = len(r.cats)
    P, O, F = r.probs(logmel(pcm, r.W, r.meta))
    ref = decode(P, O, F, r.meta, silence(pcm, r.meta["decode"]["silence_db"]))
    with tempfile.TemporaryDirectory() as tmp:
        pf = os.path.join(tmp, "p.f32")
        out = subprocess.run([a.cli, "-m", a.model, "--json", "--probs", pf, a.audio], capture_output=True, text=True, check=True)
        cp = np.fromfile(pf, np.float32).reshape(-1, 3 * C)
    got = [json.loads(l) for l in out.stdout.splitlines()]
    print(out.stderr.strip())
    print(f"frames: python {len(P)}, C {len(cp)}")
    d = np.abs(np.concatenate([P, O, F], 1) - cp)
    print(f"probabilities: max |diff| {d.max():.2e}, mean {d.mean():.2e}")
    key = lambda e: (e["category"], round(e["start"], 1), round(e["end"], 1))
    a_, b_ = sorted(map(key, ref)), sorted(map(key, got))
    same = sum(1 for x in a_ if x in set(b_))
    print(f"events: python {len(ref)}, C {len(got)}, identical (category, start, end) {same}")
    ok = len(P) == len(cp) and d.max() < 1e-3 and same >= 0.98 * max(len(ref), 1) and abs(len(ref) - len(got)) <= max(1, len(ref) // 50)
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
