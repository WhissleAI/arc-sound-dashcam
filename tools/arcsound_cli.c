/* arcsound_cli: sound-event timeline for a WAV file or a live 16 kHz s16le stream on stdin.
 *
 *   arcsound_cli [-m MODEL_DIR] [--json] [--live] [--probs OUT.f32] [--threshold CAT=P ...] INPUT
 *     INPUT  a 16 kHz mono 16-bit WAV file, or - for raw s16le 16 kHz mono on stdin, e.g.
 *            ffmpeg -loglevel error -i drive.mp4 -f s16le -ac 1 -ar 16000 - | arcsound_cli -
 *            arecord -q -f S16_LE -r 16000 -c 1 -t raw | arcsound_cli --live -
 *     --json     one JSON object per event (start, end, category, score) instead of a table
 *     --live     print events as they finish and category on/off changes as they happen
 *     --probs F  write all frame probabilities (float32, frames x 3C: presence, onset, offset)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arcsound.h"

static int read_wav_header(FILE *f) {   /* positions f at the PCM data; 0 if ok */
    unsigned char h[12];
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) return -1;
    int fmt_ok = 0;
    for (;;) {
        unsigned char c[8];
        if (fread(c, 1, 8, f) != 8) return -1;
        unsigned n = c[4] | c[5] << 8 | c[6] << 16 | (unsigned)c[7] << 24;
        if (!memcmp(c, "fmt ", 4)) {
            unsigned char b[64];
            if (n < 16 || n > sizeof b || fread(b, 1, n, f) != n) return -1;
            int fmt = b[0] | b[1] << 8, ch = b[2] | b[3] << 8, bits = b[14] | b[15] << 8;
            unsigned sr = b[4] | b[5] << 8 | b[6] << 16 | (unsigned)b[7] << 24;
            if (fmt != 1 || ch != 1 || sr != 16000 || bits != 16) {
                fprintf(stderr, "need 16 kHz mono 16-bit PCM WAV (got fmt %d, %d ch, %u Hz, %d bit); convert with\n"
                                "  ffmpeg -i in -ac 1 -ar 16000 -c:a pcm_s16le out.wav\n", fmt, ch, sr, bits);
                return -2;
            }
            fmt_ok = 1;
        } else if (!memcmp(c, "data", 4)) {
            return fmt_ok ? 0 : -1;
        } else if (fseek(f, n + (n & 1), SEEK_CUR)) {
            return -1;
        }
    }
}

static int cmp_ev(const void *a, const void *b) {
    const arcsound_event *x = a, *y = b;
    if (x->start != y->start) return x->start < y->start ? -1 : 1;
    return x->category - y->category;
}

int main(int argc, char **argv) {
    const char *model = "model", *in = NULL, *probs = NULL;
    int json = 0, live = 0;
    const char *thr[64]; int nthr = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--json")) json = 1;
        else if (!strcmp(argv[i], "--live")) live = 1;
        else if (!strcmp(argv[i], "--probs") && i + 1 < argc) probs = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc && nthr < 64) thr[nthr++] = argv[++i];
        else if (argv[i][0] == '-' && argv[i][1]) { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
        else in = argv[i];
    }
    if (!in) { fprintf(stderr, "usage: arcsound_cli [-m MODEL_DIR] [--json] [--live] [--probs F] [--threshold CAT=P] INPUT.wav|-\n"); return 2; }
    char err[512];
    arcsound_model *m = arcsound_load(model, err, sizeof err);
    if (!m) { fprintf(stderr, "%s\n", err); return 1; }
    for (int i = 0; i < nthr; i++) {
        char name[64]; float p;
        if (sscanf(thr[i], "%63[^=]=%f", name, &p) != 2 || arcsound_category_index(m, name) < 0) { fprintf(stderr, "bad --threshold %s\n", thr[i]); return 2; }
        arcsound_set_threshold(m, arcsound_category_index(m, name), p);
    }
    FILE *f = strcmp(in, "-") ? fopen(in, "rb") : stdin;
    if (!f) { fprintf(stderr, "cannot open %s\n", in); return 1; }
    if (f != stdin) { int r = read_wav_header(f); if (r) { if (r == -1) fprintf(stderr, "%s: not a WAV file\n", in); return 1; } }
    arcsound_stream *s = arcsound_stream_new(m);
    int C = arcsound_num_categories(m);
    arcsound_event *all = NULL; int nall = 0, cap = 0;
    int prev_on[ARCSOUND_MAX_CATS] = {0};
    int16_t buf[1600]; size_t n; long long samples = 0;
    clock_t t0 = clock();
    int done = 0;
    while (!done) {
        n = fread(buf, sizeof(int16_t), 1600, f);
        if (n > 0) { arcsound_push_s16(s, buf, n); samples += (long long)n; }
        else { arcsound_flush(s); done = 1; }
        arcsound_event ev[64]; int k;
        while ((k = arcsound_poll(s, ev, 64)) > 0)
            for (int i = 0; i < k; i++) {
                if (nall == cap) { cap = cap ? 2 * cap : 256; all = realloc(all, sizeof *all * (size_t)cap); }
                all[nall++] = ev[i];
                if (live) printf("[%8.2f] event  %-13s %7.2f - %7.2f  score %.2f\n", samples / 16000.0, arcsound_category_name(m, ev[i].category), ev[i].start, ev[i].end, ev[i].score);
            }
        if (live) {
            int act[ARCSOUND_MAX_CATS], na = arcsound_active(s, act, ARCSOUND_MAX_CATS), on[ARCSOUND_MAX_CATS] = {0};
            for (int i = 0; i < na; i++) on[act[i]] = 1;
            for (int c = 0; c < C; c++) if (on[c] != prev_on[c]) {
                printf("[%8.2f] %s %s\n", samples / 16000.0, on[c] ? "ON " : "off", arcsound_category_name(m, c));
                prev_on[c] = on[c];
            }
            fflush(stdout);
        }
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC, dur = samples / 16000.0;
    if (probs) {
        FILE *o = fopen(probs, "wb");
        float *row = malloc(sizeof(float) * 3 * (size_t)C);
        for (int t = 0; t < arcsound_num_frames(s); t++) { arcsound_frame_probs(s, t, row); fwrite(row, sizeof(float), 3 * (size_t)C, o); }
        free(row); fclose(o);
    }
    if (!live) {
        qsort(all, (size_t)nall, sizeof *all, cmp_ev);
        for (int i = 0; i < nall; i++) {
            if (json) printf("{\"start\": %.2f, \"end\": %.2f, \"category\": \"%s\", \"score\": %.3f}\n", all[i].start, all[i].end, arcsound_category_name(m, all[i].category), all[i].score);
            else printf("%8.2f %8.2f  %-13s %.2f\n", all[i].start, all[i].end, arcsound_category_name(m, all[i].category), all[i].score);
        }
    }
    fprintf(stderr, "%d events, %.1f s of audio in %.2f s CPU (%.0fx real time)\n", nall, dur, secs, secs > 0 ? dur / secs : 0);
    free(all); arcsound_stream_free(s); arcsound_free(m);
    if (f != stdin) fclose(f);
    return 0;
}
