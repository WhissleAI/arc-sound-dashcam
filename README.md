# arc-sound-dashcam: Arc Sound for dashcams and car cabins

**Arc Sound** marks when sounds start and end in dashcam and in-cabin audio, and what they are: crashes, horns, sirens,
skids, glass, shouts and more, as a live timeline. This repository has the inference code:

- **`libarcsound`**: a native C library (C99 + libm, no other dependencies) with a streaming API. You push PCM in any
  chunk size and get events back.
- **`arcsound_cli`**: a command-line tool for WAV files, ffmpeg pipes and live microphones.
- **`python/arcsound`**: a small numpy reference implementation that the C library is tested against.

The model is small (1.9 M parameters, 7.6 MB in float32) and causal: it never looks ahead. It is built to run on
small boards such as the Raspberry Pi Zero.

## Categories

| group | categories |
|---|---|
| incident | `crash_impact`, `glass_break`, `tire_skid`, `debris`, `incident` (the whole crash sequence) |
| alerts | `horn`, `siren`, `alarm_beep`, `train` |
| people and animals | `scream_shout`, `dog` |
| vehicle | `door`, `click_mech`, `engine`, `vehicle_pass` |
| scene | `traffic`, `thunder`, `water`, `wind_rain`, `music`, `hum` |

Each event has a start and end in seconds, a category, and a score: the peak presence probability inside the event.
Sounds can overlap, so a horn during traffic noise is reported as two events.

## Quickstart

```bash
git clone git@github.com:WhissleAI/arc-sound-dashcam.git && cd arc-sound-dashcam
./scripts/fetch_model.sh                  # -> ./model (arcsound.safetensors, arcsound_meta.json), SHA-256 checked
                                          #    model files are private: needs read access (see below)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure

# a video or any audio file, through ffmpeg
ffmpeg -loglevel error -i drive.mp4 -f s16le -ac 1 -ar 16000 - | ./build/arcsound_cli -
# a 16 kHz mono 16-bit WAV file, one JSON object per event
./build/arcsound_cli --json clip.wav
# live from a microphone (Linux / Raspberry Pi): on/off changes as they happen, events as they finish
arecord -q -f S16_LE -r 16000 -c 1 -t raw | ./build/arcsound_cli --live -
```

Example output on the first 60 s of clip `tv1_000` of the public test set
[WhissleAI/arc-sound-dashcam-test-v1](https://huggingface.co/datasets/WhissleAI/arc-sound-dashcam-test-v1)
(`start end category score`; music and traffic rows left out):

```
    9.60    10.90  crash_impact  0.59
    9.60    12.40  incident      0.82
    9.80    13.00  debris        0.54
   11.50    12.30  scream_shout  0.34
   12.90    13.20  horn          0.38
   12.90    13.20  alarm_beep    0.37
   14.20    15.10  scream_shout  0.31
```

The model files are distributed separately from this code, as release assets of a private repository. Ask Whissle Inc.
(ksingla@whissle.ai) for read access, then log in with `gh auth login` (or set `GITHUB_TOKEN`) before running
`scripts/fetch_model.sh`.

Options:

- `-m DIR`: the model directory (default `model`).
- `--threshold CAT=P`: change a category's presence threshold (repeatable). Lower finds more, higher is stricter.
- `--probs F`: write every frame's probabilities as float32 `frames x 3C`.

Python (numpy, plus ffmpeg on the PATH):

```bash
PYTHONPATH=python python -m arcsound drive.mp4 --json drive.events.json --srt drive.srt
```

The `.srt` file can be loaded as subtitles on top of the video to review the timeline.

## C API

```c
#include "arcsound.h"

char err[512];
arcsound_model *m = arcsound_load("model", err, sizeof err);   /* one model, many streams */
arcsound_stream *s = arcsound_stream_new(m);

while (read_pcm(buf, &n)) {                        /* 16 kHz mono; float [-1, 1] or int16 */
    arcsound_push_s16(s, buf, n);

    int act[ARCSOUND_MAX_CATS];                    /* what is sounding right now (0.1-0.2 s latency) */
    int na = arcsound_active(s, act, ARCSOUND_MAX_CATS);

    arcsound_event ev[64];                         /* finished events, with refined start / end */
    int k = arcsound_poll(s, ev, 64);
    for (int i = 0; i < k; i++)
        printf("%.2f-%.2f %s %.2f\n", ev[i].start, ev[i].end, arcsound_category_name(m, ev[i].category), ev[i].score);
}
arcsound_flush(s);                                  /* end of input: finish the last events, then poll again */
arcsound_stream_free(s);
arcsound_free(m);
```

See [include/arcsound.h](include/arcsound.h) for the full API: thresholds, per-frame probabilities, reset.

**Timing.** The model makes a decision every 0.1 s from the audio so far. `arcsound_active()` follows presence with
0.1-0.2 s of delay. A finished event (`arcsound_poll`) is reported once later audio can no longer change it, about 1.5 s
after it ends. That is when its start and end have been refined from the boundary outputs. Streaming results do not
depend on chunk size: pushing one sample at a time or a whole file gives bit-identical output (`tests/test_stream.c`).

**Threads.** A loaded model is read-only and can be shared. Each stream must be used by one thread at a time.

## How it works

1. 16 kHz mono audio becomes 64-band log-mel features, 100 frames per second.
2. A small causal convolution stem reduces these to 10 frames per second.
3. A streaming encoder carries its state forward through the whole recording, so earlier context (a skid, a scene of
   heavy traffic) informs later decisions. A causal local attention layer looks back over the last 6.4 s.
4. Three outputs per category every 0.1 s: presence, onset and offset probabilities.
5. The event decoder:
   - smooths presence over 3 frames;
   - applies per-category on / off thresholds (off = 0.6 x on);
   - merges gaps of 0.5 s or less and drops events shorter than 0.3 s (long sounds such as trains and water have
     their own gap and minimum length in `arcsound_meta.json`, `decode.per_cat`);
   - for impulsive sounds, moves the start and end to the peaks of the onset and offset outputs.

   Stretches of digital silence (muted or cut audio) never produce events.

## Performance

- **x86-64:** one core runs about 45x faster than real time (plain C, `-O3`, no SIMD).
- **Raspberry Pi Zero:**
  - Per 0.1 s frame the model does about 3 M multiply-adds, plus a 512-point FFT every 10 ms.
  - A Pi Zero 2 W should run it in real time with room to spare. The original Pi Zero (ARM11, no SIMD) is probably
    too slow for this model.
  - Neither board has been measured yet; measurements will be added here.
- **Memory:** the weights take 7.6 MB, and each stream keeps about 0.5 MB plus 2.5 KB per second of audio
  (the per-frame probabilities; about 9 MB per hour).

## Tests

- `ctest`: streaming invariance (`tests/test_stream.c`, synthetic audio). The same audio pushed in chunks of 1 to 16000
  samples must give bit-identical probabilities and events.
- `python tests/parity.py AUDIO.wav`: the C library against the Python reference. Requires numpy and soundfile.
  On a 10.9-minute dashcam recording:
  - all 6,535 frames, maximum probability difference 2e-4;
  - all 111 events identical.

## Licence

Code and model weights: CC BY-NC 4.0 plus the Whissle terms in [LICENSE](LICENSE). Non-commercial use only. For
commercial licences contact Whissle Inc. (ksingla@whissle.ai). Third-party components are listed in [NOTICE](NOTICE).
