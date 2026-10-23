#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "nn.h"
#include "rng.h"

static void fill_uniform(float *v, int n, Pcg32 *rng, float scale) {
    for (int i = 0; i < n; i++) {
        v[i] = scale * (2.0f * pcg32_float(rng) - 1.0f);
    }
}

/* L = 0.5 * sum(y^2) for y = dense(x); returns L and leaves y in out. */
static double dense_loss(const Dense *d, const float *x, int batch, float *y) {
    dense_forward(d, x, batch, y);
    double loss = 0.0;
    for (int k = 0; k < batch * d->out; k++) {
        loss += 0.5 * (double)y[k] * (double)y[k];
    }
    return loss;
}

void test_nn_dense_gradients_match_finite_differences(void) {
    enum { IN = 5, OUT = 4, B = 3 };
    float w[IN * OUT], b[OUT], gw[IN * OUT], gb[OUT];
    float x[B * IN], y[B * OUT], gx[B * IN];
    Pcg32 rng;
    pcg32_seed(&rng, 11u, 3u);
    fill_uniform(w, IN * OUT, &rng, 0.5f);
    fill_uniform(b, OUT, &rng, 0.5f);
    fill_uniform(x, B * IN, &rng, 1.0f);
    memset(gw, 0, sizeof gw);
    memset(gb, 0, sizeof gb);
    Dense d = {IN, OUT, w, b, gw, gb};

    dense_loss(&d, x, B, y);
    dense_backward(&d, x, y, B, gx); /* dL/dy = y */

    const float eps = 1e-2f;
    double worst = 0.0;
    for (int k = 0; k < IN * OUT; k++) {
        float keep = w[k];
        w[k] = keep + eps;
        double up = dense_loss(&d, x, B, y);
        w[k] = keep - eps;
        double down = dense_loss(&d, x, B, y);
        w[k] = keep;
        worst = fmax(worst, fabs((up - down) / (2.0 * eps) - gw[k]));
    }
    for (int k = 0; k < OUT; k++) {
        float keep = b[k];
        b[k] = keep + eps;
        double up = dense_loss(&d, x, B, y);
        b[k] = keep - eps;
        double down = dense_loss(&d, x, B, y);
        b[k] = keep;
        worst = fmax(worst, fabs((up - down) / (2.0 * eps) - gb[k]));
    }
    for (int k = 0; k < B * IN; k++) {
        float keep = x[k];
        x[k] = keep + eps;
        double up = dense_loss(&d, x, B, y);
        x[k] = keep - eps;
        double down = dense_loss(&d, x, B, y);
        x[k] = keep;
        worst = fmax(worst, fabs((up - down) / (2.0 * eps) - gx[k]));
    }
    CHECK(worst < 1e-3);

    /* Gradients accumulate across calls; gx may be omitted. */
    float before = gw[0];
    dense_loss(&d, x, B, y);
    dense_backward(&d, x, y, B, NULL);
    CHECK_NEAR(gw[0], 2.0f * before, 1e-5);
}
