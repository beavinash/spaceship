#ifndef NN_H
#define NN_H

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

#endif
