/* arcsound: native C runtime for Arc Sound. Plain C99 + libm, no other dependency (cJSON is vendored for the model
 * header). Follows python/arcsound/model.py step for step; tests/parity.py checks the two agree. */
#include "arcsound.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../third_party/cjson/cJSON.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define NFFT 512
#define NBIN (NFFT / 2 + 1)
#define HOP 160
#define SIL_HOP 1600

static void set_err(char *err, size_t n, const char *fmt, const char *a) {
    if (err && n) snprintf(err, n, fmt, a ? a : "");
}

/* ------------------------------------------------------------------ model */
typedef struct {
    float *scale, *shift, *norm, *gate_w, *gate_b, *wb_w, *wb_b, *wc_w, *wc_b, *wx_w, *wx_b, *step_w, *step_b, *rot_w,
        *rot_b, *mix_w, *mix_b, *out_w, *out_b, *decay, *nb, *nc, *bb, *bc, *ffn, *up_w, *up_b, *dn_w, *dn_b;
} enc_layer;

typedef struct {
    float *norm, *qkv_w, *qkv_b, *o_w, *o_b, *ffn, *up_w, *up_b, *dn_w, *dn_b;
} att_layer;

struct arcsound_model {
    int n_mels, D, L, N, P, H, R, A, NH, WIN, C, nimp;
    float rms_eps, ln_eps, log_eps, sil_db;
    double off_ratio, refine_min;   /* decode settings in double, as the reference decoder */
    int min_frames, gap_frames;
    float *window, *melfb, *mean, *std, *c1_w, *c1_b, *c2_w, *c2_b, *proj_w, *proj_b, *in_nw, *in_nb, *in_w, *in_b,
        *ad_dn, *ad_up, *out_nw, *head_w, *head_b;
    enc_layer *enc;
    att_layer *att;
    char **cats;
    int *impulsive;
    float *thr;
    void **blobs;
    int nblobs;
    double tw_re[NFFT / 2], tw_im[NFFT / 2];
    int bitrev[NFFT];
};

typedef struct {
    cJSON *hdr;
    const unsigned char *data;
    size_t datalen;
    arcsound_model *m;
    char *err;
    size_t errlen;
    int bad;
} loader;

static float *get(loader *ld, const char *name, size_t numel) {
    if (ld->bad) return NULL;
    cJSON *t = cJSON_GetObjectItemCaseSensitive(ld->hdr, name);
    cJSON *dt = t ? cJSON_GetObjectItemCaseSensitive(t, "dtype") : NULL;
    cJSON *sh = t ? cJSON_GetObjectItemCaseSensitive(t, "shape") : NULL;
    cJSON *off = t ? cJSON_GetObjectItemCaseSensitive(t, "data_offsets") : NULL;
    if (!t || !cJSON_IsString(dt) || strcmp(dt->valuestring, "F32") || !cJSON_IsArray(sh) || cJSON_GetArraySize(off) != 2) {
        set_err(ld->err, ld->errlen, "model tensor %s missing or not float32", name); ld->bad = 1; return NULL;
    }
    size_t n = 1;
    cJSON *d;
    cJSON_ArrayForEach(d, sh) n *= (size_t)d->valuedouble;
    size_t a = (size_t)cJSON_GetArrayItem(off, 0)->valuedouble, b = (size_t)cJSON_GetArrayItem(off, 1)->valuedouble;
    if (n != numel || b - a != n * 4 || b > ld->datalen || a > b) {
        set_err(ld->err, ld->errlen, "model tensor %s has an unexpected shape", name); ld->bad = 1; return NULL;
    }
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { set_err(ld->err, ld->errlen, "out of memory%s", ""); ld->bad = 1; return NULL; }
    memcpy(p, ld->data + a, n * 4);   /* safetensors is little-endian; so are all supported targets */
    ld->m->blobs = (void **)realloc(ld->m->blobs, sizeof(void *) * (size_t)(ld->m->nblobs + 1));
    ld->m->blobs[ld->m->nblobs++] = p;
    return p;
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = n >= 0 ? (char *)malloc((size_t)n + 1) : NULL;
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); b = NULL; }
    fclose(f);
    if (b) { b[n] = 0; *len = (size_t)n; }
    return b;
}

static double jnum(cJSON *o, const char *k, double dflt) {
    cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsNumber(v) ? v->valuedouble : dflt;
}

void arcsound_free(arcsound_model *m) {
    if (!m) return;
    for (int i = 0; i < m->nblobs; i++) free(m->blobs[i]);
    free(m->blobs); free(m->enc); free(m->att);
    if (m->cats) for (int i = 0; i < m->C; i++) free(m->cats[i]);
    free(m->cats); free(m->impulsive); free(m->thr); free(m);
}

arcsound_model *arcsound_load(const char *dir, char *err, size_t errlen) {
    char path[4096];
    size_t mlen = 0, wlen = 0;
    snprintf(path, sizeof path, "%s/arcsound_meta.json", dir);
    char *mtxt = read_file(path, &mlen);
    if (!mtxt) { set_err(err, errlen, "cannot read %s", path); return NULL; }
    cJSON *meta = cJSON_Parse(mtxt); free(mtxt);
    if (!meta) { set_err(err, errlen, "%s is not valid JSON", path); return NULL; }
    arcsound_model *m = (arcsound_model *)calloc(1, sizeof *m);
    m->n_mels = (int)jnum(meta, "n_mels", 64); m->D = (int)jnum(meta, "d_model", 0); m->L = (int)jnum(meta, "n_layers", 0);
    m->N = (int)jnum(meta, "d_state", 0); m->P = (int)jnum(meta, "head_dim", 0); m->R = (int)jnum(meta, "adapters", 0);
    m->A = (int)jnum(meta, "attn_layers", 0); m->NH = (int)jnum(meta, "n_heads", 1); m->WIN = (int)jnum(meta, "attn_window", 64);
    m->rms_eps = (float)jnum(meta, "rms_eps", 1.1920929e-07); m->ln_eps = (float)jnum(meta, "ln_eps", 1e-5);
    m->log_eps = (float)jnum(meta, "log_eps", 1e-6);
    m->H = m->P ? m->D / m->P : 0;
    cJSON *cats = cJSON_GetObjectItemCaseSensitive(meta, "categories");
    m->C = cJSON_GetArraySize(cats);
    if (!m->D || !m->L || !m->N || !m->H || m->C <= 0 || m->C > ARCSOUND_MAX_CATS || m->n_mels != 64 ||
        (int)jnum(meta, "sample_rate", 0) != 16000 || (int)jnum(meta, "hop", 0) != HOP || (int)jnum(meta, "n_fft", 0) != NFFT) {
        set_err(err, errlen, "unsupported model configuration in %s", path); cJSON_Delete(meta); arcsound_free(m); return NULL;
    }
    m->cats = (char **)calloc((size_t)m->C, sizeof(char *)); m->impulsive = (int *)calloc((size_t)m->C, sizeof(int));
    m->thr = (float *)calloc((size_t)m->C, sizeof(float));
    cJSON *dc = cJSON_GetObjectItemCaseSensitive(meta, "decode"), *on = cJSON_GetObjectItemCaseSensitive(dc, "on");
    cJSON *imp = cJSON_GetObjectItemCaseSensitive(meta, "impulsive");
    for (int i = 0; i < m->C; i++) {
        const char *nm = cJSON_GetArrayItem(cats, i)->valuestring;
        m->cats[i] = strdup(nm); m->thr[i] = (float)jnum(on, nm, 0.2);
        cJSON *x;
        cJSON_ArrayForEach(x, imp) if (cJSON_IsString(x) && !strcmp(x->valuestring, nm)) m->impulsive[i] = 1;
    }
    m->off_ratio = jnum(dc, "off_ratio", 0.6); m->refine_min = jnum(dc, "refine_min", 0.3);
    m->min_frames = (int)lround(jnum(dc, "min_dur", 0.3) * 10); m->gap_frames = (int)lround(jnum(dc, "gap", 0.5) * 10);
    m->sil_db = (float)jnum(dc, "silence_db", -70);
    cJSON_Delete(meta);

    snprintf(path, sizeof path, "%s/arcsound.safetensors", dir);
    char *w = read_file(path, &wlen);
    if (!w || wlen < 8) { set_err(err, errlen, "cannot read %s", path); free(w); arcsound_free(m); return NULL; }
    uint64_t hl = 0;
    for (int i = 7; i >= 0; i--) hl = (hl << 8) | (unsigned char)w[i];
    if (hl > wlen - 8) { set_err(err, errlen, "%s: bad header", path); free(w); arcsound_free(m); return NULL; }
    char *htxt = (char *)malloc((size_t)hl + 1); memcpy(htxt, w + 8, (size_t)hl); htxt[hl] = 0;
    loader ld = {cJSON_Parse(htxt), (const unsigned char *)w + 8 + hl, wlen - 8 - (size_t)hl, m, err, errlen, 0};
    free(htxt);
    if (!ld.hdr) { set_err(err, errlen, "%s: bad header", path); free(w); arcsound_free(m); return NULL; }
    int D = m->D, H = m->H, N = m->N, Mm = m->n_mels, F2 = Mm / 4;
    m->window = get(&ld, "fe.window", NFFT); m->melfb = get(&ld, "fe.mel", (size_t)Mm * NBIN);
    m->mean = get(&ld, "fe.mean", Mm); m->std = get(&ld, "fe.std", Mm);
    m->c1_w = get(&ld, "stem.c1.w", 32 * 9); m->c1_b = get(&ld, "stem.c1.b", 32);
    m->c2_w = get(&ld, "stem.c2.w", 64 * 32 * 15); m->c2_b = get(&ld, "stem.c2.b", 64);
    m->proj_w = get(&ld, "stem.proj.w", (size_t)D * 64 * F2); m->proj_b = get(&ld, "stem.proj.b", D);
    m->in_nw = get(&ld, "enc.in_norm.w", D); m->in_nb = get(&ld, "enc.in_norm.b", D);
    m->in_w = get(&ld, "enc.in.w", (size_t)D * D); m->in_b = get(&ld, "enc.in.b", D);
    m->enc = (enc_layer *)calloc((size_t)m->L, sizeof(enc_layer));
    char k[128];
#define G(field, nm, n) (snprintf(k, sizeof k, "enc.%d." nm, i), e->field = get(&ld, k, (size_t)(n)))
    for (int i = 0; i < m->L; i++) {
        enc_layer *e = &m->enc[i];
        G(scale, "scale", D); G(shift, "shift", D); G(norm, "seq.norm.w", D);
        G(gate_w, "seq.gate.w", D * D); G(gate_b, "seq.gate.b", D);
        G(wb_w, "seq.wb.w", H * N * D); G(wb_b, "seq.wb.b", H * N); G(wc_w, "seq.wc.w", H * N * D); G(wc_b, "seq.wc.b", H * N);
        G(wx_w, "seq.wx.w", D * D); G(wx_b, "seq.wx.b", D); G(step_w, "seq.step.w", H * D); G(step_b, "seq.step.b", H);
        G(rot_w, "seq.rot.w", H * N / 2 * D); G(rot_b, "seq.rot.b", H * N / 2); G(mix_w, "seq.mix.w", H * D); G(mix_b, "seq.mix.b", H);
        G(out_w, "seq.out.w", D * D); G(out_b, "seq.out.b", D); G(decay, "seq.decay", H);
        G(nb, "seq.nb.w", N); G(nc, "seq.nc.w", N); G(bb, "seq.bb", H * N); G(bc, "seq.bc", H * N);
        G(ffn, "ff.norm.w", D); G(up_w, "ff.up.w", 2 * D * D); G(up_b, "ff.up.b", 2 * D);
        G(dn_w, "ff.down.w", 2 * D * D); G(dn_b, "ff.down.b", D);
    }
#undef G
    m->ad_dn = get(&ld, "enc.adapt.down", (size_t)m->R * D); m->ad_up = get(&ld, "enc.adapt.up", (size_t)D * m->R);
    m->out_nw = get(&ld, "enc.out_norm.w", D);
    m->att = (att_layer *)calloc((size_t)(m->A ? m->A : 1), sizeof(att_layer));
#define G(field, nm, n) (snprintf(k, sizeof k, "att.%d." nm, i), a->field = get(&ld, k, (size_t)(n)))
    for (int i = 0; i < m->A; i++) {
        att_layer *a = &m->att[i];
        G(norm, "norm.w", D); G(qkv_w, "qkv.w", 3 * D * D); G(qkv_b, "qkv.b", 3 * D); G(o_w, "o.w", D * D); G(o_b, "o.b", D);
        G(ffn, "ff.norm.w", D); G(up_w, "ff.up.w", 2 * D * D); G(up_b, "ff.up.b", 2 * D);
        G(dn_w, "ff.down.w", 2 * D * D); G(dn_b, "ff.down.b", D);
    }
#undef G
    m->head_w = get(&ld, "head.w", (size_t)3 * m->C * D); m->head_b = get(&ld, "head.b", (size_t)3 * m->C);
    cJSON_Delete(ld.hdr); free(w);
    if (ld.bad) { arcsound_free(m); return NULL; }
    for (int i = 0; i < NFFT / 2; i++) { m->tw_re[i] = cos(-2 * M_PI * i / NFFT); m->tw_im[i] = sin(-2 * M_PI * i / NFFT); }
    for (int i = 0; i < NFFT; i++) {
        int r = 0;
        for (int b = 1, j = i; b < NFFT; b <<= 1, j >>= 1) r = (r << 1) | (j & 1);
        m->bitrev[i] = r;
    }
    return m;
}

int arcsound_num_categories(const arcsound_model *m) { return m->C; }
const char *arcsound_category_name(const arcsound_model *m, int i) { return i >= 0 && i < m->C ? m->cats[i] : NULL; }
int arcsound_category_index(const arcsound_model *m, const char *name) {
    for (int i = 0; i < m->C; i++) if (!strcmp(m->cats[i], name)) return i;
    return -1;
}
int arcsound_is_impulsive(const arcsound_model *m, int i) { return i >= 0 && i < m->C && m->impulsive[i]; }
float arcsound_threshold(const arcsound_model *m, int i) { return i >= 0 && i < m->C ? m->thr[i] : 0.f; }
void arcsound_set_threshold(arcsound_model *m, int i, float on) { if (i >= 0 && i < m->C && on > 0) m->thr[i] = on; }

/* ------------------------------------------------------------------ math */
static void matvec(const float *W, const float *b, const float *x, float *y, int out, int in) {
    for (int o = 0; o < out; o++) {
        const float *w = W + (size_t)o * in;
        float s = b ? b[o] : 0.f;
        for (int i = 0; i < in; i++) s += w[i] * x[i];
        y[o] = s;
    }
}
static float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }
static float sigm(float x) { return 1.f / (1.f + expf(-x)); }
static float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
static void rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    float s = 0.f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float r = 1.f / sqrtf(s / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r * (w ? w[i] : 1.f);
}
static float half_round(float f) {   /* float -> IEEE half (round to nearest even) -> float */
    union { float f; uint32_t u; } v = {f};
    uint32_t s = v.u & 0x80000000u, e = (v.u >> 23) & 0xff, mnt = v.u & 0x7fffff;
    if (e == 0xff) return f;
    int ex = (int)e - 127;
    if (ex > 15) { v.u = s | 0x477fe000u; return v.f; }   /* clamp to 65504 */
    if (ex < -24) { v.u = s; return v.f; }
    int shift = ex < -14 ? 13 + (-14 - ex) : 13;          /* subnormal halves keep fewer bits */
    uint32_t full = mnt | 0x800000u;
    uint32_t keep = full >> shift, rem = full & ((1u << shift) - 1), halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (keep & 1))) keep++;
    double r = ldexp((double)keep, ex - 23 + shift);
    return (float)(s ? -r : r);
}

/* ------------------------------------------------------------------ stream */
typedef struct {
    int act, t0, has_pend, pa, pb;
} cat_dec;

typedef struct {
    int a, b, cat;
} seg;

struct arcsound_stream {
    const arcsound_model *m;
    float *pcm; size_t pcm_cap, pcm_n; int64_t pcm_base, total;       /* pcm[0] = sample pcm_base */
    double sil_acc; int sil_cnt; unsigned char *sil; int sil_n, sil_cap;
    int mel_rows, c1_rows, frames, flushed;
    float *mel_ring;   /* 3 x n_mels normalised rows (row r at r % 3) */
    float *c1_ring;    /* 5 x 32 x (n_mels/2) */
    float *st, *bprev, *xprev;   /* per layer: H x N x P state; last write factors H x N and H x P */
    float *kr, *vr; int att_n;   /* per attention layer: WIN x D keys / values */
    float *probs; int probs_cap;  /* frames x 3C */
    cat_dec dec[ARCSOUND_MAX_CATS]; int dec_t;
    seg *done; int done_n, done_cap;            /* closed segments waiting for refinement data */
    arcsound_event *out; int out_n, out_cap, out_r;
    float *tmp;
    double re[NFFT], im[NFFT];
};

arcsound_stream *arcsound_stream_new(const arcsound_model *m) {
    arcsound_stream *s = (arcsound_stream *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->m = m;
    int D = m->D;
    s->mel_ring = (float *)calloc((size_t)3 * m->n_mels, sizeof(float));
    s->c1_ring = (float *)calloc((size_t)5 * 32 * (m->n_mels / 2), sizeof(float));
    s->st = (float *)calloc((size_t)m->L * m->H * m->N * m->P, sizeof(float));
    s->bprev = (float *)calloc((size_t)m->L * m->H * m->N, sizeof(float));
    s->xprev = (float *)calloc((size_t)m->L * m->H * m->P, sizeof(float));
    s->kr = (float *)calloc((size_t)(m->A ? m->A : 1) * m->WIN * D, sizeof(float));
    s->vr = (float *)calloc((size_t)(m->A ? m->A : 1) * m->WIN * D, sizeof(float));
    s->tmp = (float *)calloc((size_t)16 * D + 64 * 16 + 4096, sizeof(float));
    if (!s->mel_ring || !s->c1_ring || !s->st || !s->bprev || !s->xprev || !s->kr || !s->vr || !s->tmp) {
        arcsound_stream_free(s); return NULL;
    }
    return s;
}

void arcsound_stream_free(arcsound_stream *s) {
    if (!s) return;
    free(s->pcm); free(s->sil); free(s->mel_ring); free(s->c1_ring); free(s->st); free(s->bprev); free(s->xprev);
    free(s->kr); free(s->vr); free(s->probs); free(s->done); free(s->out); free(s->tmp); free(s);
}

void arcsound_stream_reset(arcsound_stream *s) {
    const arcsound_model *m = s->m;
    s->pcm_n = 0; s->pcm_base = 0; s->total = 0; s->sil_acc = 0; s->sil_cnt = 0; s->sil_n = 0;
    s->mel_rows = s->c1_rows = s->frames = s->flushed = 0; s->att_n = 0; s->dec_t = 0;
    s->done_n = s->out_n = s->out_r = 0;
    memset(s->dec, 0, sizeof s->dec);
    memset(s->st, 0, sizeof(float) * (size_t)m->L * m->H * m->N * m->P);
    memset(s->bprev, 0, sizeof(float) * (size_t)m->L * m->H * m->N);
    memset(s->xprev, 0, sizeof(float) * (size_t)m->L * m->H * m->P);
}

static float sample_at(const arcsound_stream *s, int64_t i) {   /* pcm sample i (0 outside the stream) */
    if (i < s->pcm_base || i >= s->total) return 0.f;
    return s->pcm[i - s->pcm_base];
}

/* log-mel row r, normalised, into the ring */
static void mel_row(arcsound_stream *s, int r) {
    const arcsound_model *m = s->m;
    int64_t a = (int64_t)r * HOP - NFFT / 2;
    for (int i = 0; i < NFFT; i++) { int j = m->bitrev[i]; s->re[j] = sample_at(s, a + i) * (double)m->window[i]; s->im[j] = 0; }
    for (int len = 2; len <= NFFT; len <<= 1) {
        int half = len / 2, step = NFFT / len;
        for (int i = 0; i < NFFT; i += len)
            for (int k = 0; k < half; k++) {
                double wr = m->tw_re[k * step], wi = m->tw_im[k * step];
                double xr = s->re[i + k + half] * wr - s->im[i + k + half] * wi, xi = s->re[i + k + half] * wi + s->im[i + k + half] * wr;
                s->re[i + k + half] = s->re[i + k] - xr; s->im[i + k + half] = s->im[i + k] - xi;
                s->re[i + k] += xr; s->im[i + k] += xi;
            }
    }
    float pw[NBIN];
    for (int k = 0; k < NBIN; k++) pw[k] = (float)(s->re[k] * s->re[k] + s->im[k] * s->im[k]);
    float *row = s->mel_ring + (r % 3) * m->n_mels;
    for (int b = 0; b < m->n_mels; b++) {
        const float *f = m->melfb + (size_t)b * NBIN;
        float v = 0.f;
        for (int k = 0; k < NBIN; k++) v += f[k] * pw[k];
        row[b] = (half_round(logf(v + m->log_eps)) - m->mean[b]) / m->std[b];
    }
}

/* first conv row j (needs mel rows 2j-2 .. 2j) -> c1 ring, gelu applied */
static void c1_row(arcsound_stream *s, int j) {
    const arcsound_model *m = s->m;
    int M = m->n_mels, F1 = M / 2;
    float *out = s->c1_ring + (size_t)(j % 5) * 32 * F1;
    for (int c = 0; c < 32; c++)
        for (int f = 0; f < F1; f++) {
            float acc = m->c1_b[c];
            for (int kt = 0; kt < 3; kt++) {
                int r = 2 * j - 2 + kt;
                if (r < 0) continue;
                const float *row = s->mel_ring + (r % 3) * M;
                for (int kf = 0; kf < 3; kf++) {
                    int q = 2 * f - 1 + kf;
                    if (q >= 0 && q < M) acc += m->c1_w[(c * 3 + kt) * 3 + kf] * row[q];
                }
            }
            out[c * F1 + f] = gelu(acc);
        }
}

/* one 10 Hz frame k: second conv (c1 rows 5k-4 .. 5k), projection, encoder, attention, heads -> probs */
static void frame(arcsound_stream *s, int k) {
    const arcsound_model *m = s->m;
    int M = m->n_mels, F1 = M / 2, F2 = M / 4, D = m->D, H = m->H, N = m->N, P = m->P, C = m->C;
    (void)P;
    float *t = s->tmp, *x2 = t, *h = x2 + 64 * F2, *u = h + D, *nu = u + D, *a1 = nu + D, *a2 = a1 + 2 * D, *y = a2 + 2 * D,
          *gt = y + D, *Bv = gt + D, *Cv = Bv + H * N, *xv = Cv + H * N, *th = xv + D, *dt = th + H * N / 2, *lam = dt + H;
    for (int o = 0; o < 64; o++)
        for (int f = 0; f < F2; f++) {
            float acc = m->c2_b[o];
            for (int ci = 0; ci < 32; ci++)
                for (int kt = 0; kt < 5; kt++) {
                    int r = 5 * k - 4 + kt;
                    if (r < 0) continue;
                    const float *in = s->c1_ring + (size_t)(r % 5) * 32 * F1 + ci * F1;
                    const float *w = m->c2_w + ((o * 32 + ci) * 5 + kt) * 3;
                    for (int kf = 0; kf < 3; kf++) {
                        int q = 2 * f - 1 + kf;
                        if (q >= 0 && q < F1) acc += w[kf] * in[q];
                    }
                }
            x2[o * F2 + f] = gelu(acc);
        }
    matvec(m->proj_w, m->proj_b, x2, u, D, 64 * F2);
    {   /* layer norm -> input projection */
        float mu = 0, var = 0;
        for (int i = 0; i < D; i++) mu += u[i];
        mu /= D;
        for (int i = 0; i < D; i++) var += (u[i] - mu) * (u[i] - mu);
        float r = 1.f / sqrtf(var / D + m->ln_eps);
        for (int i = 0; i < D; i++) nu[i] = (u[i] - mu) * r * m->in_nw[i] + m->in_nb[i];
        matvec(m->in_w, m->in_b, nu, h, D, D);
    }
    for (int l = 0; l < m->L; l++) {
        const enc_layer *e = &m->enc[l];
        float *st = s->st + (size_t)l * H * N * P, *bp = s->bprev + (size_t)l * H * N, *xp = s->xprev + (size_t)l * H * P;
        for (int i = 0; i < D; i++) h[i] = h[i] * e->scale[i] + e->shift[i];
        rmsnorm(h, e->norm, nu, D, m->rms_eps);
        matvec(e->step_w, e->step_b, nu, dt, H, D);
        matvec(e->mix_w, e->mix_b, nu, lam, H, D);
        matvec(e->wb_w, e->wb_b, nu, Bv, H * N, D);
        matvec(e->wc_w, e->wc_b, nu, Cv, H * N, D);
        matvec(e->wx_w, e->wx_b, nu, xv, D, D);
        matvec(e->rot_w, e->rot_b, nu, th, H * N / 2, D);
        for (int hh = 0; hh < H; hh++) {
            rmsnorm(Bv + hh * N, e->nb, Bv + hh * N, N, m->rms_eps);
            rmsnorm(Cv + hh * N, e->nc, Cv + hh * N, N, m->rms_eps);
            for (int n = 0; n < N; n++) { Bv[hh * N + n] += e->bb[hh * N + n]; Cv[hh * N + n] += e->bc[hh * N + n]; }
            float d = softplus(dt[hh]), alpha = expf(-d * e->decay[hh]), lm = sigm(lam[hh]);
            float gamma = lm * d, beta = (1.f - lm) * d * alpha;
            float *S = st + (size_t)hh * N * P;
            const float *b0 = bp + hh * N, *x0 = xp + hh * P, *b1 = Bv + hh * N, *x1 = xv + hh * P;
            for (int i = 0; i < N / 2; i++) {
                float ang = th[hh * (N / 2) + i] * d, c = cosf(ang), sn = sinf(ang);
                float *ra = S + (size_t)(2 * i) * P, *rb = ra + P;
                for (int p = 0; p < P; p++) {
                    float pa = alpha * ra[p] + beta * b0[2 * i] * x0[p], pb = alpha * rb[p] + beta * b0[2 * i + 1] * x0[p];
                    ra[p] = c * pa - sn * pb + gamma * b1[2 * i] * x1[p];
                    rb[p] = sn * pa + c * pb + gamma * b1[2 * i + 1] * x1[p];
                }
            }
            for (int p = 0; p < P; p++) {
                float acc = 0.f;
                for (int n = 0; n < N; n++) acc += Cv[hh * N + n] * S[(size_t)n * P + p];
                y[hh * P + p] = acc;
            }
        }
        memcpy(bp, Bv, sizeof(float) * (size_t)H * N); memcpy(xp, xv, sizeof(float) * (size_t)H * P);
        matvec(e->gate_w, e->gate_b, nu, gt, D, D);
        for (int i = 0; i < D; i++) y[i] *= gt[i] * sigm(gt[i]);
        matvec(e->out_w, e->out_b, y, u, D, D);
        for (int i = 0; i < D; i++) h[i] += u[i];
        rmsnorm(h, e->ffn, nu, D, m->rms_eps);
        matvec(e->up_w, e->up_b, nu, a1, 2 * D, D);
        for (int i = 0; i < 2 * D; i++) a1[i] = gelu(a1[i]);
        matvec(e->dn_w, e->dn_b, a1, u, D, 2 * D);
        for (int i = 0; i < D; i++) h[i] += u[i];
    }
    matvec(m->ad_dn, NULL, h, a1, m->R, D);
    for (int i = 0; i < m->R; i++) a1[i] = gelu(a1[i]);
    matvec(m->ad_up, NULL, a1, u, D, m->R);
    for (int i = 0; i < D; i++) h[i] += u[i];
    rmsnorm(h, m->out_nw, h, D, m->rms_eps);
    int dh = D / m->NH, W = m->WIN;
    for (int l = 0; l < m->A; l++) {
        const att_layer *a = &m->att[l];
        float *K = s->kr + (size_t)l * W * D, *V = s->vr + (size_t)l * W * D;
        rmsnorm(h, a->norm, nu, D, m->rms_eps);
        matvec(a->qkv_w, a->qkv_b, nu, a2, 3 * D, D);   /* a2 has room for 2D; spill q into a1 */
        float *q = a1; memcpy(q, a2, sizeof(float) * D);
        int slot = k % W;
        memcpy(K + (size_t)slot * D, a2 + D, sizeof(float) * D); memcpy(V + (size_t)slot * D, a2 + 2 * D, sizeof(float) * D);
        int nk = k + 1 < W ? k + 1 : W;
        float sc[1024];
        for (int hh = 0; hh < m->NH; hh++) {
            float mx = -1e30f, sum = 0.f, scale = 1.f / sqrtf((float)dh);
            for (int j = 0; j < nk; j++) {
                const float *kk = K + (size_t)((k - j) % W) * D + hh * dh;
                float d = 0.f;
                for (int i = 0; i < dh; i++) d += q[hh * dh + i] * kk[i];
                sc[j] = d * scale; if (sc[j] > mx) mx = sc[j];
            }
            for (int j = 0; j < nk; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            for (int i = 0; i < dh; i++) {
                float acc = 0.f;
                for (int j = 0; j < nk; j++) acc += sc[j] * V[(size_t)((k - j) % W) * D + hh * dh + i];
                y[hh * dh + i] = acc / sum;
            }
        }
        matvec(a->o_w, a->o_b, y, u, D, D);
        for (int i = 0; i < D; i++) h[i] += u[i];
        rmsnorm(h, a->ffn, nu, D, m->rms_eps);
        matvec(a->up_w, a->up_b, nu, a1, 2 * D, D);
        for (int i = 0; i < 2 * D; i++) a1[i] = gelu(a1[i]);
        matvec(a->dn_w, a->dn_b, a1, u, D, 2 * D);
        for (int i = 0; i < D; i++) h[i] += u[i];
    }
    if (k >= s->probs_cap) {
        s->probs_cap = s->probs_cap ? 2 * s->probs_cap : 1024;
        s->probs = (float *)realloc(s->probs, sizeof(float) * (size_t)s->probs_cap * 3 * C);
    }
    float *pr = s->probs + (size_t)k * 3 * C;
    matvec(m->head_w, m->head_b, h, pr, 3 * C, D);
    for (int i = 0; i < 3 * C; i++) pr[i] = sigm(pr[i]);
}

/* ------------------------------------------------------------------ decoder (same result as the offline decoder) */
static float pm(const arcsound_stream *s, int t, int c) {   /* presence with digital silence masked */
    if (t < 0 || t >= s->frames) return 0.f;
    if (t < s->sil_n && s->sil[t]) return 0.f;
    return s->probs[(size_t)t * 3 * s->m->C + c];
}

static void emit(arcsound_stream *s, int a, int b, int c) {
    const arcsound_model *m = s->m;
    int C = m->C, T = s->frames;
    if (b - a < m->min_frames) return;
    double st = a / 10.0, en = b / 10.0;
    if (m->impulsive[c]) {
        int lo = a - 10 > 0 ? a - 10 : 0, hi = a + 10 < T ? a + 10 : T, best = lo;
        for (int t = lo; t < hi; t++) if (s->probs[(size_t)t * 3 * C + C + c] > s->probs[(size_t)best * 3 * C + C + c]) best = t;
        if (hi > lo && s->probs[(size_t)best * 3 * C + C + c] > m->refine_min) st = best / 10.0;
        lo = b - 15 > 0 ? b - 15 : 0; hi = b + 15 < T ? b + 15 : T; best = lo;
        for (int t = lo; t < hi; t++) if (s->probs[(size_t)t * 3 * C + 2 * C + c] > s->probs[(size_t)best * 3 * C + 2 * C + c]) best = t;
        if (hi > lo && s->probs[(size_t)best * 3 * C + 2 * C + c] > m->refine_min) { en = best / 10.0; if (en < st + 0.3) en = st + 0.3; }
    }
    float sc = 0.f;
    for (int t = a; t < b; t++) if (pm(s, t, c) > sc) sc = pm(s, t, c);
    if (s->out_n == s->out_cap) {
        s->out_cap = s->out_cap ? 2 * s->out_cap : 64;
        s->out = (arcsound_event *)realloc(s->out, sizeof(arcsound_event) * (size_t)s->out_cap);
    }
    s->out[s->out_n++] = (arcsound_event){st, en, c, sc};
}

static void close_seg(arcsound_stream *s, int c, int a, int b) {
    cat_dec *d = &s->dec[c];
    int gapf = s->m->gap_frames;
    if (d->has_pend && a - d->pb <= gapf) { d->pb = b; return; }
    if (d->has_pend) {
        if (s->done_n == s->done_cap) { s->done_cap = s->done_cap ? 2 * s->done_cap : 64; s->done = (seg *)realloc(s->done, sizeof(seg) * (size_t)s->done_cap); }
        s->done[s->done_n++] = (seg){d->pa, d->pb, c};
    }
    d->has_pend = 1; d->pa = a; d->pb = b;
}

static void decode(arcsound_stream *s) {
    const arcsound_model *m = s->m;
    int C = m->C, gapf = m->gap_frames;
    /* frames whose silence status is known */
    int known = s->flushed ? s->frames : (s->sil_n < s->frames ? s->sil_n : s->frames);
    int lim = s->flushed ? s->frames + 1 : known - 1;   /* smoothed value at t needs t+1 (t = T: the closing zero) */
    for (; s->dec_t < lim; s->dec_t++) {
        int t = s->dec_t;
        for (int c = 0; c < C; c++) {
            cat_dec *d = &s->dec[c];
            double v = t < s->frames ? ((double)pm(s, t - 1, c) + pm(s, t, c) + pm(s, t + 1, c)) / 3.0 : 0.0;
            double on = m->thr[c], off = m->off_ratio * (double)m->thr[c];
            if (!d->act && v >= on) { d->act = 1; d->t0 = t; }
            else if (d->act && v < off) { d->act = 0; close_seg(s, c, d->t0, t); }
        }
    }
    for (int c = 0; c < C; c++) {   /* a pending segment is final once no later segment can merge into it */
        cat_dec *d = &s->dec[c];
        if (d->has_pend && !d->act && (s->flushed || s->dec_t - d->pb > gapf)) {
            if (s->done_n == s->done_cap) { s->done_cap = s->done_cap ? 2 * s->done_cap : 64; s->done = (seg *)realloc(s->done, sizeof(seg) * (size_t)s->done_cap); }
            s->done[s->done_n++] = (seg){d->pa, d->pb, c}; d->has_pend = 0;
        }
    }
    int w = 0;   /* emit those whose refinement windows are complete */
    for (int i = 0; i < s->done_n; i++) {
        seg g = s->done[i];
        if (s->flushed || s->frames >= g.b + 15) emit(s, g.a, g.b, g.cat);
        else s->done[w++] = g;
    }
    s->done_n = w;
}

/* ------------------------------------------------------------------ push / flush */
static int advance(arcsound_stream *s) {
    int before = s->frames;
    int64_t total_rows = s->flushed ? 1 + s->total / HOP : -1;
    for (;;) {
        int r = s->mel_rows;
        if (s->flushed ? r >= total_rows : (int64_t)r * HOP + NFFT / 2 > s->total) break;
        mel_row(s, r); s->mel_rows++;
        if (r % 2 == 0) {
            int j = r / 2;
            c1_row(s, j); s->c1_rows++;
            if (j % 5 == 0) { frame(s, j / 5); s->frames++; }
        }
    }
    /* keep only the samples future rows still need */
    int64_t need = (int64_t)s->mel_rows * HOP - NFFT / 2;
    if (need > s->pcm_base + 4096) {
        int64_t drop = need - s->pcm_base;
        memmove(s->pcm, s->pcm + drop, sizeof(float) * (size_t)(s->pcm_n - (size_t)drop));
        s->pcm_n -= (size_t)drop; s->pcm_base += drop;
    }
    decode(s);
    return s->frames - before;
}

int arcsound_push(arcsound_stream *s, const float *x, size_t n) {
    if (s->flushed) return -1;
    if (s->pcm_n + n > s->pcm_cap) {
        while (s->pcm_n + n > s->pcm_cap) s->pcm_cap = s->pcm_cap ? 2 * s->pcm_cap : 65536;
        s->pcm = (float *)realloc(s->pcm, sizeof(float) * s->pcm_cap);
    }
    memcpy(s->pcm + s->pcm_n, x, sizeof(float) * n);
    s->pcm_n += n; s->total += (int64_t)n;
    for (size_t i = 0; i < n; i++) {   /* digital-silence map, 0.1 s frames */
        s->sil_acc += (double)x[i] * x[i];
        if (++s->sil_cnt == SIL_HOP) {
            double r = sqrt(s->sil_acc / SIL_HOP);
            if (s->sil_n == s->sil_cap) { s->sil_cap = s->sil_cap ? 2 * s->sil_cap : 1024; s->sil = (unsigned char *)realloc(s->sil, (size_t)s->sil_cap); }
            s->sil[s->sil_n++] = 20 * log10(r + 1e-9) < s->m->sil_db;
            s->sil_acc = 0; s->sil_cnt = 0;
        }
    }
    return advance(s);
}

int arcsound_push_s16(arcsound_stream *s, const int16_t *x, size_t n) {
    float buf[4096];
    int got = 0;
    for (size_t i = 0; i < n; i += 4096) {
        size_t k = n - i < 4096 ? n - i : 4096;
        for (size_t j = 0; j < k; j++) buf[j] = x[i + j] / 32768.f;
        int r = arcsound_push(s, buf, k);
        if (r < 0) return r;
        got += r;
    }
    return got;
}

int arcsound_flush(arcsound_stream *s) {
    if (s->flushed) return 0;
    s->flushed = 1;
    return advance(s);
}

int arcsound_num_frames(const arcsound_stream *s) { return s->frames; }
int arcsound_frame_probs(const arcsound_stream *s, int t, float *out) {
    if (t < 0 || t >= s->frames) return -1;
    memcpy(out, s->probs + (size_t)t * 3 * s->m->C, sizeof(float) * 3 * (size_t)s->m->C);
    return 0;
}

int arcsound_poll(arcsound_stream *s, arcsound_event *out, int max) {
    int k = 0;
    while (k < max && s->out_r < s->out_n) out[k++] = s->out[s->out_r++];
    if (s->out_r == s->out_n) s->out_r = s->out_n = 0;
    return k;
}

int arcsound_active(const arcsound_stream *s, int *cats, int max) {
    int k = 0;
    for (int c = 0; c < s->m->C && k < max; c++) if (s->dec[c].act) cats[k++] = c;
    return k;
}
