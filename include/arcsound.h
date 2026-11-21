/* arcsound: C API for Arc Sound (dashcam / in-cabin sound events).
 *
 * Streaming: push 16 kHz mono PCM in any chunk size; the library emits presence / onset / offset probabilities for
 * every category at 10 frames per second and turns them into timed events (start, end, category, score).
 *
 *   arcsound_model  *m = arcsound_load("model", err, sizeof err);
 *   arcsound_stream *s = arcsound_stream_new(m);
 *   while (read pcm) {
 *       arcsound_push(s, pcm, n);                         // or arcsound_push_s16
 *       arcsound_event ev[64]; int k = arcsound_poll(s, ev, 64);   // finished events
 *       int act[ARCSOUND_MAX_CATS]; int na = arcsound_active(s, act, ARCSOUND_MAX_CATS);   // what is sounding now
 *   }
 *   arcsound_flush(s);  // end of input: finishes the last frames and events, then poll again
 *   arcsound_stream_free(s); arcsound_free(m);
 *
 * Model frames are causal (no lookahead). A finished event is reported once no later audio can change it: about
 * 1.5 s after its end. arcsound_active() reports a category as soon as its smoothed presence crosses the threshold
 * (0.1-0.2 s). One model can serve many streams; a stream must be used by one thread at a time.
 */
#ifndef ARCSOUND_H
#define ARCSOUND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARCSOUND_VERSION "1.0.0"
#define ARCSOUND_MAX_CATS 64

typedef struct arcsound_model arcsound_model;
typedef struct arcsound_stream arcsound_stream;

typedef struct {
    double start, end;   /* seconds from the start of the stream */
    int category;        /* index, see arcsound_category_name */
    float score;         /* peak presence probability inside the event */
} arcsound_event;

/* Load arcsound.safetensors + arcsound_meta.json from a directory. NULL on error (message in err). */
arcsound_model *arcsound_load(const char *model_dir, char *err, size_t errlen);
void arcsound_free(arcsound_model *m);

int arcsound_num_categories(const arcsound_model *m);
const char *arcsound_category_name(const arcsound_model *m, int i);
int arcsound_category_index(const arcsound_model *m, const char *name);   /* -1 if unknown */
int arcsound_is_impulsive(const arcsound_model *m, int i);
/* Presence threshold per category (the release default is tuned per category); off threshold = 0.6 x on. */
float arcsound_threshold(const arcsound_model *m, int i);
void arcsound_set_threshold(arcsound_model *m, int i, float on);

arcsound_stream *arcsound_stream_new(const arcsound_model *m);
void arcsound_stream_reset(arcsound_stream *s);
void arcsound_stream_free(arcsound_stream *s);

/* Push samples (16 kHz mono; float in [-1, 1] or int16). Returns the number of new 10 Hz frames, or -1 after flush. */
int arcsound_push(arcsound_stream *s, const float *pcm, size_t n);
int arcsound_push_s16(arcsound_stream *s, const int16_t *pcm, size_t n);
/* End of input: computes the remaining frames and finishes all events. Returns new frames. */
int arcsound_flush(arcsound_stream *s);

/* Frames computed so far; probabilities of frame t: out[0..C) presence, [C..2C) onset, [2C..3C) offset. */
int arcsound_num_frames(const arcsound_stream *s);
int arcsound_frame_probs(const arcsound_stream *s, int t, float *out);

/* Finished events not yet returned, in order of completion. Returns how many were written (<= max). */
int arcsound_poll(arcsound_stream *s, arcsound_event *out, int max);
/* Categories currently on (smoothed presence above threshold at the newest decodable frame). Returns the count. */
int arcsound_active(const arcsound_stream *s, int *cats, int max);

#ifdef __cplusplus
}
#endif
#endif
