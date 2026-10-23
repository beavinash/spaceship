#ifndef TOY_H
#define TOY_H

#include <stdint.h>

#include "rng.h"

/*
 * Learner sanity task with the game's action shape: move a point (h/v heads, 0.1 per step)
 * to a random target and fire (third head) within 0.15 of it. -0.01 per step, plus the
 * distance closed this step (progress shaping), -0.05 per wasted shot, +1 for a hit
 * (terminal); truncated after TOY_MAX_STEPS steps.
 */
#define TOY_OBS 4
#define TOY_MAX_STEPS 40

typedef struct {
    float px, py;
    float tx, ty;
    int steps;
    Pcg32 rng;
} ToyEnv;

void toy_reset(ToyEnv *e, uint64_t seed);
float toy_step(ToyEnv *e, const int *action, int *terminated, int *truncated);
void toy_observe(const ToyEnv *e, float *obs);

typedef struct {
    int updates;
    int episodes;
    double hit_rate; /* share of the last 100 finished episodes that ended in a hit */
} ToyResult;

/* Train PPO; returns the mean return of the last 100 finished episodes. */
double toy_train(uint64_t seed, int updates, ToyResult *out);

/* Mean return of uniformly random actions over the given number of episodes. */
double toy_random_return(uint64_t seed, int episodes);

#endif
