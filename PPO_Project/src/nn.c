#include "nn.h"

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
