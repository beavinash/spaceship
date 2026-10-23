#include "dist.h"

#include <math.h>

const int HEAD_SIZES[ACTION_HEADS] = {3, 3, 2};

/* Stable log-softmax of one head into logp. */
static void log_softmax(const float *z, int n, float *logp) {
    float max = z[0];
    for (int k = 1; k < n; k++) {
        max = z[k] > max ? z[k] : max;
    }
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += expf(z[k] - max);
    }
    float log_sum = logf(sum) + max;
    for (int k = 0; k < n; k++) {
        logp[k] = z[k] - log_sum;
    }
}

float dist_log_prob(const float *logits, const int *action) {
    float total = 0.0f;
    float logp[3];
    for (int h = 0, off = 0; h < ACTION_HEADS; off += HEAD_SIZES[h], h++) {
        log_softmax(logits + off, HEAD_SIZES[h], logp);
        total += logp[action[h]];
    }
    return total;
}

float dist_entropy(const float *logits) {
    float total = 0.0f;
    float logp[3];
    for (int h = 0, off = 0; h < ACTION_HEADS; off += HEAD_SIZES[h], h++) {
        log_softmax(logits + off, HEAD_SIZES[h], logp);
        for (int k = 0; k < HEAD_SIZES[h]; k++) {
            total -= expf(logp[k]) * logp[k];
        }
    }
    return total;
}

void dist_log_prob_grad(const float *logits, const int *action, float *grad) {
    float logp[3];
    for (int h = 0, off = 0; h < ACTION_HEADS; off += HEAD_SIZES[h], h++) {
        log_softmax(logits + off, HEAD_SIZES[h], logp);
        for (int k = 0; k < HEAD_SIZES[h]; k++) {
            grad[off + k] = (k == action[h] ? 1.0f : 0.0f) - expf(logp[k]);
        }
    }
}

void dist_entropy_grad(const float *logits, float *grad) {
    float logp[3];
    for (int h = 0, off = 0; h < ACTION_HEADS; off += HEAD_SIZES[h], h++) {
        int n = HEAD_SIZES[h];
        log_softmax(logits + off, n, logp);
        float entropy = 0.0f;
        for (int k = 0; k < n; k++) {
            entropy -= expf(logp[k]) * logp[k];
        }
        for (int k = 0; k < n; k++) {
            grad[off + k] = -expf(logp[k]) * (logp[k] + entropy);
        }
    }
}

void dist_sample(const float *logits, Pcg32 *rng, int *action) {
    float logp[3];
    for (int h = 0, off = 0; h < ACTION_HEADS; off += HEAD_SIZES[h], h++) {
        int n = HEAD_SIZES[h];
        log_softmax(logits + off, n, logp);
        float u = pcg32_float(rng);
        int pick = n - 1; /* rounding leftovers fall on the last option */
        float cdf = 0.0f;
        for (int k = 0; k < n; k++) {
            cdf += expf(logp[k]);
            if (u < cdf) {
                pick = k;
                break;
            }
        }
        action[h] = pick;
    }
}

void dist_argmax(const float *logits, int *action) {
    for (int h = 0, off = 0; h < ACTION_HEADS; off += HEAD_SIZES[h], h++) {
        int best = 0;
        for (int k = 1; k < HEAD_SIZES[h]; k++) {
            if (logits[off + k] > logits[off + best]) {
                best = k;
            }
        }
        action[h] = best;
    }
}
