#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dense_forward(const Dense *d, const float *x, int batch, float *y) {
    for (int n = 0; n < batch; n++) {
        const float *xn = x + (size_t)n * d->in;
        float *yn = y + (size_t)n * d->out;
        memcpy(yn, d->b, (size_t)d->out * sizeof *yn);
        for (int i = 0; i < d->in; i++) {
            const float xi = xn[i];
            const float *wi = d->w + (size_t)i * d->out;
            for (int o = 0; o < d->out; o++) {
                yn[o] += xi * wi[o];
            }
        }
    }
}

void dense_backward(Dense *d, const float *x, const float *gy, int batch, float *gx) {
    for (int n = 0; n < batch; n++) {
        const float *xn = x + (size_t)n * d->in;
        const float *gn = gy + (size_t)n * d->out;
        for (int o = 0; o < d->out; o++) {
            d->gb[o] += gn[o];
        }
        for (int i = 0; i < d->in; i++) {
            const float xi = xn[i];
            float *gwi = d->gw + (size_t)i * d->out;
            for (int o = 0; o < d->out; o++) {
                gwi[o] += xi * gn[o];
            }
        }
        if (gx) {
            float *gxn = gx + (size_t)n * d->in;
            for (int i = 0; i < d->in; i++) {
                const float *wi = d->w + (size_t)i * d->out;
                float acc = 0.0f;
                for (int o = 0; o < d->out; o++) {
                    acc += wi[o] * gn[o];
                }
                gxn[i] = acc;
            }
        }
    }
}

static void *grow(void *ptr, size_t bytes) {
    void *p = realloc(ptr, bytes);
    if (!p) {
        fprintf(stderr, "nn: out of memory (%zu bytes)\n", bytes);
        abort();
    }
    return p;
}

static double gaussian(Pcg32 *rng) {
    double u1 = 1.0 - (double)pcg32_float(rng); /* (0, 1] keeps the log finite */
    double u2 = (double)pcg32_float(rng);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* Orthogonal matrix (as in Saxe et al. / torch.nn.init.orthogonal_) scaled by gain, via Gram-Schmidt. */
static int orthogonal_init(float *w, int in, int out, float gain, Pcg32 *rng) {
    int rows = in > out ? in : out;
    int cols = in < out ? in : out;
    size_t n = (size_t)rows * (size_t)cols;
    double *q = calloc(n, sizeof *q); /* column-major */
    if (!q) {
        return -1;
    }
    for (size_t k = 0; k < n; k++) {
        q[k] = gaussian(rng);
    }
    for (int c = 0; c < cols; c++) {
        double *qc = q + (size_t)c * rows;
        for (int pass = 0; pass < 2; pass++) { /* second pass restores orthogonality lost to rounding */
            for (int p = 0; p < c; p++) {
                const double *qp = q + (size_t)p * rows;
                double dot = 0.0;
                for (int r = 0; r < rows; r++) {
                    dot += qc[r] * qp[r];
                }
                for (int r = 0; r < rows; r++) {
                    qc[r] -= dot * qp[r];
                }
            }
        }
        double norm = 0.0;
        for (int r = 0; r < rows; r++) {
            norm += qc[r] * qc[r];
        }
        norm = sqrt(norm);
        for (int r = 0; r < rows; r++) {
            qc[r] /= norm;
        }
    }
    for (int i = 0; i < in; i++) {
        for (int o = 0; o < out; o++) {
            double v = in >= out ? q[(size_t)o * rows + i] : q[(size_t)i * rows + o];
            w[(size_t)i * out + o] = (float)(gain * v);
        }
    }
    free(q);
    return 0;
}

int mlp_init(Mlp *m, int in, int hidden, int out, float out_gain, Pcg32 *rng) {
    memset(m, 0, sizeof *m);
    m->sizes[0] = in;
    m->sizes[1] = hidden;
    m->sizes[2] = hidden;
    m->sizes[3] = out;
    for (int l = 0; l < 3; l++) {
        m->n_params += m->sizes[l] * m->sizes[l + 1] + m->sizes[l + 1];
    }
    m->params = calloc((size_t)m->n_params, sizeof *m->params);
    m->grads = calloc((size_t)m->n_params, sizeof *m->grads);
    if (!m->params || !m->grads) {
        mlp_free(m);
        return -1;
    }
    const float gains[3] = {sqrtf(2.0f), sqrtf(2.0f), out_gain};
    size_t offset = 0;
    for (int l = 0; l < 3; l++) {
        Dense *d = &m->layers[l];
        d->in = m->sizes[l];
        d->out = m->sizes[l + 1];
        d->w = m->params + offset;
        d->gw = m->grads + offset;
        offset += (size_t)d->in * d->out;
        d->b = m->params + offset;
        d->gb = m->grads + offset;
        offset += (size_t)d->out;
        if (orthogonal_init(d->w, d->in, d->out, gains[l], rng) != 0) {
            mlp_free(m);
            return -1;
        }
    }
    return 0;
}

void mlp_free(Mlp *m) {
    free(m->params);
    free(m->grads);
    for (int l = 0; l < 3; l++) {
        free(m->acts[l]);
    }
    free(m->gbuf[0]);
    free(m->gbuf[1]);
    memset(m, 0, sizeof *m);
}

static void reserve(Mlp *m, int batch) {
    if (batch <= m->cap) {
        return;
    }
    for (int l = 0; l < 3; l++) {
        m->acts[l] = grow(m->acts[l], (size_t)batch * m->sizes[l + 1] * sizeof(float));
    }
    for (int k = 0; k < 2; k++) {
        m->gbuf[k] = grow(m->gbuf[k], (size_t)batch * m->sizes[1] * sizeof(float));
    }
    m->cap = batch;
}

const float *mlp_forward(Mlp *m, const float *x, int batch) {
    reserve(m, batch);
    const float *in = x;
    for (int l = 0; l < 3; l++) {
        dense_forward(&m->layers[l], in, batch, m->acts[l]);
        if (l < 2) {
            float *a = m->acts[l];
            for (size_t k = 0; k < (size_t)batch * m->sizes[l + 1]; k++) {
                a[k] = tanhf(a[k]);
            }
        }
        in = m->acts[l];
    }
    return m->acts[2];
}

void mlp_backward(Mlp *m, const float *x, const float *gout, int batch) {
    const size_t hidden = (size_t)batch * m->sizes[1];
    dense_backward(&m->layers[2], m->acts[1], gout, batch, m->gbuf[0]);
    for (size_t k = 0; k < hidden; k++) {
        m->gbuf[0][k] *= 1.0f - m->acts[1][k] * m->acts[1][k];
    }
    dense_backward(&m->layers[1], m->acts[0], m->gbuf[0], batch, m->gbuf[1]);
    for (size_t k = 0; k < hidden; k++) {
        m->gbuf[1][k] *= 1.0f - m->acts[0][k] * m->acts[0][k];
    }
    dense_backward(&m->layers[0], x, m->gbuf[1], batch, NULL);
}

void mlp_zero_grad(Mlp *m) {
    memset(m->grads, 0, (size_t)m->n_params * sizeof *m->grads);
}

int adam_init(Adam *a, int n) {
    memset(a, 0, sizeof *a);
    a->n = n;
    a->beta1 = 0.9f;
    a->beta2 = 0.999f;
    a->eps = 1e-5f;
    a->m = calloc((size_t)n, sizeof *a->m);
    a->v = calloc((size_t)n, sizeof *a->v);
    if (!a->m || !a->v) {
        adam_free(a);
        return -1;
    }
    return 0;
}

void adam_free(Adam *a) {
    free(a->m);
    free(a->v);
    memset(a, 0, sizeof *a);
}

void adam_step(Adam *a, float *params, const float *grads, float lr) {
    a->t++;
    const double c1 = 1.0 - pow(a->beta1, a->t);
    const double c2 = 1.0 - pow(a->beta2, a->t);
    for (int k = 0; k < a->n; k++) {
        a->m[k] = a->beta1 * a->m[k] + (1.0f - a->beta1) * grads[k];
        a->v[k] = a->beta2 * a->v[k] + (1.0f - a->beta2) * grads[k] * grads[k];
        const double m_hat = a->m[k] / c1;
        const double v_hat = a->v[k] / c2;
        params[k] -= (float)(lr * m_hat / (sqrt(v_hat) + a->eps));
    }
}

float clip_grad_norm(float *const *groups, const int *sizes, int count, float max_norm) {
    double sq = 0.0;
    for (int g = 0; g < count; g++) {
        for (int k = 0; k < sizes[g]; k++) {
            sq += (double)groups[g][k] * groups[g][k];
        }
    }
    const double norm = sqrt(sq);
    if (norm > max_norm) {
        const float scale = (float)(max_norm / (norm + 1e-6));
        for (int g = 0; g < count; g++) {
            for (int k = 0; k < sizes[g]; k++) {
                groups[g][k] *= scale;
            }
        }
    }
    return (float)norm;
}
