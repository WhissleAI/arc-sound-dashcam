/* Streaming invariance: the same audio pushed in different chunk sizes must give bit-identical frame probabilities
 * and the same events. Synthetic audio (noise bed, tones, bursts, a muted gap), so no data files are needed.
 * Exit 77 (skipped) when the model directory is missing. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arcsound.h"

#define SR 16000
#define DUR 40

static void synth(float *x, int n) {
    unsigned s = 12345;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        float noise = ((s >> 9) / 4194304.f - 1.f) * 0.02f;
        double t = (double)i / SR;
        float v = noise + 0.05f * (float)sin(2 * M_PI * 90 * t);                     /* low rumble */
        if (t > 5 && t < 8) v += 0.3f * (float)sin(2 * M_PI * 440 * t) + 0.2f * (float)sin(2 * M_PI * 554 * t);   /* horn-like */
        if (t > 12 && t < 20) v += 0.25f * (float)sin(2 * M_PI * (700 + 300 * sin(2 * M_PI * 0.5 * t)) * t);   /* wail */
        if (fmod(t, 9.0) < 0.05 && t > 21) v += ((s >> 7) & 1 ? 0.8f : -0.8f);         /* bursts */
        if (t > 30 && t < 32) v = 0.f;                                              /* digital silence */
        x[i] = v;
    }
}

static int run(arcsound_model *m, const float *x, int n, int chunk, float **probs, int *nf, arcsound_event *ev, int *nev) {
    arcsound_stream *s = arcsound_stream_new(m);
    *nev = 0;
    for (int i = 0; i < n; i += chunk) {
        arcsound_push(s, x + i, (size_t)(n - i < chunk ? n - i : chunk));
        *nev += arcsound_poll(s, ev + *nev, 4096 - *nev);
    }
    arcsound_flush(s);
    *nev += arcsound_poll(s, ev + *nev, 4096 - *nev);
    int C = arcsound_num_categories(m);
    *nf = arcsound_num_frames(s);
    *probs = malloc(sizeof(float) * 3 * (size_t)C * (size_t)*nf);
    for (int t = 0; t < *nf; t++) arcsound_frame_probs(s, t, *probs + (size_t)t * 3 * C);
    arcsound_stream_free(s);
    return 0;
}

static int cmp(const void *a, const void *b) {
    const arcsound_event *x = a, *y = b;
    if (x->start != y->start) return x->start < y->start ? -1 : 1;
    return x->category - y->category;
}

int main(int argc, char **argv) {
    char err[512];
    arcsound_model *m = arcsound_load(argc > 1 ? argv[1] : "model", err, sizeof err);
    if (!m) { fprintf(stderr, "skip: %s\n", err); return 77; }
    int n = SR * DUR, C = arcsound_num_categories(m);
    float *x = malloc(sizeof(float) * (size_t)n);
    synth(x, n);
    int chunks[] = {n, 1, 160, 1601, 4000, 16000};
    float *p0 = NULL; int nf0 = 0, ne0 = 0, fail = 0;
    static arcsound_event e0[4096], e1[4096];
    for (int c = 0; c < 6; c++) {
        float *p; int nf, ne;
        run(m, x, n, chunks[c], &p, &nf, c ? e1 : e0, &ne);
        if (!c) { p0 = p; nf0 = nf; ne0 = ne; qsort(e0, (size_t)ne, sizeof *e0, cmp); continue; }
        qsort(e1, (size_t)ne, sizeof *e1, cmp);
        int same = nf == nf0 && !memcmp(p, p0, sizeof(float) * 3 * (size_t)C * (size_t)nf) && ne == ne0;
        for (int i = 0; same && i < ne; i++) same = e0[i].start == e1[i].start && e0[i].end == e1[i].end && e0[i].category == e1[i].category && e0[i].score == e1[i].score;
        printf("chunk %6d: %d frames, %d events  %s\n", chunks[c], nf, ne, same ? "identical" : "DIFFERENT");
        fail |= !same;
        free(p);
    }
    printf("expected frames %d (got %d), events %d\n", 1 + ((1 + n / 160) - 1) / 2 / 5 + 0, nf0, ne0);
    free(p0); free(x); arcsound_free(m);
    return fail;
}
