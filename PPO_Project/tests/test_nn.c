#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

/* Max |(W^T W or W W^T) - gain^2 I| for a [in][out] matrix, using the smaller Gram matrix. */
static double orthogonality_error(const float *w, int in, int out, double gain) {
    int small = in < out ? in : out;
    double worst = 0.0;
    for (int a = 0; a < small; a++) {
        for (int b = 0; b < small; b++) {
            double dot = 0.0;
            if (in >= out) {
                for (int i = 0; i < in; i++) {
                    dot += (double)w[i * out + a] * w[i * out + b];
                }
            } else {
                for (int o = 0; o < out; o++) {
                    dot += (double)w[a * out + o] * w[b * out + o];
                }
            }
            worst = fmax(worst, fabs(dot - (a == b ? gain * gain : 0.0)));
        }
    }
    return worst;
}

static double mlp_loss(Mlp *m, const float *x, int batch, const float *c) {
    const float *y = mlp_forward(m, x, batch);
    double loss = 0.0;
    for (int k = 0; k < batch * m->sizes[3]; k++) {
        loss += (double)c[k] * y[k];
    }
    return loss;
}

void test_nn_mlp_init_shapes_and_gradients(void) {
    Pcg32 rng;
    pcg32_seed(&rng, 5u, 8u);
    Mlp m;
    CHECK(mlp_init(&m, 6, 16, 3, 0.01f, &rng) == 0);
    CHECK(m.n_params == 6 * 16 + 16 + 16 * 16 + 16 + 16 * 3 + 3);
    CHECK(orthogonality_error(m.layers[0].w, 6, 16, sqrt(2.0)) < 1e-4);
    CHECK(orthogonality_error(m.layers[1].w, 16, 16, sqrt(2.0)) < 1e-4);
    CHECK(orthogonality_error(m.layers[2].w, 16, 3, 0.01) < 1e-6);
    int zero_bias = 1;
    for (int l = 0; l < 3; l++) {
        for (int o = 0; o < m.layers[l].out; o++) {
            zero_bias &= m.layers[l].b[o] == 0.0f;
        }
    }
    CHECK(zero_bias);

    Pcg32 again;
    pcg32_seed(&again, 5u, 8u);
    Mlp twin;
    CHECK(mlp_init(&twin, 6, 16, 3, 0.01f, &again) == 0);
    CHECK(memcmp(m.params, twin.params, (size_t)m.n_params * sizeof(float)) == 0);

    float x[4 * 6];
    fill_uniform(x, 4 * 6, &rng, 1.0f);
    float first[4 * 3];
    memcpy(first, mlp_forward(&m, x, 4), sizeof first);
    CHECK(memcmp(first, mlp_forward(&twin, x, 4), sizeof first) == 0);
    CHECK(memcmp(first, mlp_forward(&m, x, 4), sizeof first) == 0);
    mlp_free(&twin);
    mlp_free(&m);

    /* Gradient check through both tanh layers, with a unit-gain output so signals are not tiny. */
    CHECK(mlp_init(&m, 6, 16, 3, 1.0f, &rng) == 0);
    float c[4 * 3];
    fill_uniform(c, 4 * 3, &rng, 1.0f);
    mlp_zero_grad(&m);
    mlp_forward(&m, x, 4);
    mlp_backward(&m, x, c, 4);
    const float eps = 1e-2f;
    double worst = 0.0;
    for (int k = 0; k < m.n_params; k++) {
        float keep = m.params[k];
        m.params[k] = keep + eps;
        double up = mlp_loss(&m, x, 4, c);
        m.params[k] = keep - eps;
        double down = mlp_loss(&m, x, 4, c);
        m.params[k] = keep;
        double numeric = (up - down) / (2.0 * eps);
        worst = fmax(worst, fabs(numeric - m.grads[k]) / fmax(1.0, fabs(numeric)));
    }
    CHECK(worst < 2e-3);
    mlp_zero_grad(&m);
    int cleared = 1;
    for (int k = 0; k < m.n_params; k++) {
        cleared &= m.grads[k] == 0.0f;
    }
    CHECK(cleared);
    mlp_free(&m);
}

void test_nn_adam_and_gradient_clipping(void) {
    Adam adam;
    CHECK(adam_init(&adam, 2) == 0);
    CHECK_NEAR(adam.beta1, 0.9, 1e-7);
    CHECK_NEAR(adam.beta2, 0.999, 1e-7);
    CHECK_NEAR(adam.eps, 1e-5, 1e-12);
    float p[2] = {1.0f, -2.0f};
    const float g1[2] = {0.5f, -0.1f};
    const float g2[2] = {-0.2f, 0.3f};
    /* Reference: PyTorch-style bias-corrected Adam, computed independently in double precision. */
    adam_step(&adam, p, g1, 0.1f);
    CHECK_NEAR(p[0], 0.9000020, 2e-6);
    CHECK_NEAR(p[1], -1.9000100, 2e-6);
    adam_step(&adam, p, g2, 0.1f);
    CHECK_NEAR(p[0], 0.8654423, 2e-6);
    CHECK_NEAR(p[1], -1.9494268, 2e-6);
    CHECK(adam.t == 2);
    adam_free(&adam);

    float a[2] = {3.0f, 0.0f};
    float b[1] = {4.0f};
    float *groups[2] = {a, b};
    const int sizes[2] = {2, 1};
    CHECK_NEAR(clip_grad_norm(groups, sizes, 2, 1.0f), 5.0, 1e-6);
    CHECK_NEAR(a[0], 0.6, 1e-6);
    CHECK_NEAR(a[1], 0.0, 0.0);
    CHECK_NEAR(b[0], 0.8, 1e-6);
    CHECK_NEAR(clip_grad_norm(groups, sizes, 2, 10.0f), 1.0, 1e-6);
    CHECK_NEAR(a[0], 0.6, 1e-6); /* already within the limit: unchanged */
}

static long file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

static void poke_byte(const char *path, long offset) {
    FILE *f = fopen(path, "r+b");
    if (!f) {
        CHECK(f != NULL);
        return;
    }
    fseek(f, offset, SEEK_SET);
    int c = fgetc(f);
    fseek(f, offset, SEEK_SET);
    fputc(c ^ 0x5a, f);
    fclose(f);
}

void test_nn_checkpoint_roundtrip_and_rejection(void) {
    const char *path = "build/test-checkpoint.bin";
    Pcg32 rng;
    pcg32_seed(&rng, 21u, 1u);
    Mlp a, b, fresh_a, fresh_b, narrow;
    CHECK(mlp_init(&a, 7, 12, 8, 0.01f, &rng) == 0);
    CHECK(mlp_init(&b, 7, 12, 1, 1.0f, &rng) == 0);
    CHECK(mlp_init(&fresh_a, 7, 12, 8, 0.01f, &rng) == 0);
    CHECK(mlp_init(&fresh_b, 7, 12, 1, 1.0f, &rng) == 0);
    CHECK(mlp_init(&narrow, 7, 6, 8, 0.01f, &rng) == 0);
    Mlp *src[2] = {&a, &b};
    Mlp *dst[2] = {&fresh_a, &fresh_b};

    CHECK(checkpoint_save(path, src, 2, 37u) == 0);
    uint32_t update = 0;
    CHECK(checkpoint_load(path, dst, 2, &update) == 0);
    CHECK(update == 37u);
    CHECK(memcmp(a.params, fresh_a.params, (size_t)a.n_params * sizeof(float)) == 0);
    CHECK(memcmp(b.params, fresh_b.params, (size_t)b.n_params * sizeof(float)) == 0);
    long size = file_size(path);
    CHECK(size == 8 + 3 * 4 + 2 * 5 * 4 + (long)(a.n_params + b.n_params) * 4 + 8);

    Mlp *wrong_shape[2] = {&narrow, &fresh_b};
    CHECK(checkpoint_load(path, wrong_shape, 2, &update) == CKPT_SHAPE);
    CHECK(checkpoint_load(path, dst, 1, &update) == CKPT_SHAPE);

    float before = fresh_a.params[3];
    poke_byte(path, size / 2);
    CHECK(checkpoint_load(path, dst, 2, &update) == CKPT_CHECKSUM);
    CHECK(fresh_a.params[3] == before); /* a rejected file changes nothing */

    CHECK(checkpoint_save(path, src, 2, 1u) == 0);
    CHECK(truncate(path, size - 3) == 0);
    CHECK(checkpoint_load(path, dst, 2, &update) == CKPT_FORMAT);
    CHECK(checkpoint_load("build/no-such-checkpoint.bin", dst, 2, &update) == CKPT_IO);

    a.params[0] = NAN;
    CHECK(checkpoint_save(path, src, 2, 2u) == 0);
    CHECK(checkpoint_load(path, dst, 2, &update) == CKPT_NONFINITE);
    CHECK(checkpoint_error(CKPT_CHECKSUM) != NULL);

    mlp_free(&a);
    mlp_free(&b);
    mlp_free(&fresh_a);
    mlp_free(&fresh_b);
    mlp_free(&narrow);
    remove(path);
}
