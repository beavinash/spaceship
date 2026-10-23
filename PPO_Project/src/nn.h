#ifndef NN_H
#define NN_H

#include "rng.h"

/*
 * Small float32 neural-network kit for PPO: dense layers over row-major batches with
 * hand-written backward passes. Gradients accumulate until the caller clears them.
 */

typedef struct {
    int in;
    int out;
    float *w;  /* [in][out] */
    float *b;  /* [out] */
    float *gw; /* same shape as w */
    float *gb; /* same shape as b */
} Dense;

/* y[n][o] = b[o] + sum_i x[n][i] * w[i][o] */
void dense_forward(const Dense *d, const float *x, int batch, float *y);

/* Adds dL/dw and dL/db for upstream gradient gy; writes dL/dx into gx unless gx is NULL. */
void dense_backward(Dense *d, const float *x, const float *gy, int batch, float *gx);

/*
 * in -> hidden (tanh) -> hidden (tanh) -> out (linear). Parameters and gradients each live in
 * one contiguous array (for Adam and checkpoints). Orthogonal init: gain sqrt(2) on hidden
 * layers, out_gain on the output layer, zero biases.
 */
typedef struct {
    int sizes[4];
    int n_params;
    float *params;
    float *grads;
    Dense layers[3];
    int cap;        /* batch rows the workspace can hold */
    float *acts[3]; /* outputs of each layer from the last forward pass (tanh applied to 0 and 1) */
    float *gbuf[2];
} Mlp;

/* Returns 0, or -1 if memory runs out. */
int mlp_init(Mlp *m, int in, int hidden, int out, float out_gain, Pcg32 *rng);
void mlp_free(Mlp *m);

/* Returns batch x out outputs, valid until the next forward pass on this network. */
const float *mlp_forward(Mlp *m, const float *x, int batch);

/* Adds parameter gradients for upstream gradient gout; x and batch must match the last forward pass. */
void mlp_backward(Mlp *m, const float *x, const float *gout, int batch);

void mlp_zero_grad(Mlp *m);

#endif
