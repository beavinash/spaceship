#ifndef DIST_H
#define DIST_H

#include "rng.h"

/*
 * Factorized categorical policy over upstream's three choices per ship: horizontal
 * {left, stay, right}, vertical {up, stay, down}, fire {no, yes}. The 8 logits are laid
 * out head by head; an action is one index per head.
 */
#define ACTION_HEADS 3
#define ACTION_LOGITS 8

extern const int HEAD_SIZES[ACTION_HEADS];

/* Sum over heads of log softmax(logits_head)[action_head]. */
float dist_log_prob(const float *logits, const int *action);

/* Sum over heads of the categorical entropy. */
float dist_entropy(const float *logits);

/* d log_prob / d logits: one-hot minus softmax, per head. */
void dist_log_prob_grad(const float *logits, const int *action, float *grad);

/* d entropy / d logits: -p * (log p + H_head), per head. */
void dist_entropy_grad(const float *logits, float *grad);

void dist_sample(const float *logits, Pcg32 *rng, int *action);

/* Most likely index per head, lowest index on ties. */
void dist_argmax(const float *logits, int *action);

#endif
