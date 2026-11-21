"""python -m arcsound [-m MODEL_DIR] MEDIA [--json OUT] [--srt OUT]   (any audio / video file ffmpeg can read)"""
import argparse
import json
import subprocess

import numpy as np

from .model import ArcSound


def ts(x):
    h, r = divmod(x, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("media"); ap.add_argument("-m", "--model", default="model")
    ap.add_argument("--json"); ap.add_argument("--srt"); a = ap.parse_args()
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", a.media, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    pcm = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    ev = ArcSound(a.model).events(pcm)
    for e in ev:
        print(f"{e['start']:8.2f} {e['end']:8.2f}  {e['category']:13s} {e['score']:.2f}")
    if a.json:
        json.dump(ev, open(a.json, "w"), indent=1)
    if a.srt:
        with open(a.srt, "w") as f:
            for i, e in enumerate(ev, 1):
                f.write(f"{i}\n{ts(e['start'])} --> {ts(max(e['end'], e['start'] + 0.8))}\n[{e['category']}] {e['score']:.2f}\n\n")


if __name__ == "__main__":
    main()
