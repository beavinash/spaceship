#ifndef PPO_H
#define PPO_H

#include "dist.h"
#include "nn.h"
#include "rng.h"

/*
 * One agent's rollout: steps x envs samples, index t * envs + n. A sample is active while
 * the agent is alive; after its death it stays inactive (masked out) until the arena resets.
 * dones[i] = 1 when the agent's trajectory ended after step i: death, team defeat, or the
 * time limit. bootstrap[i] holds V(final observation) for time-limit endings, else 0.
 */
typedef struct {
    int steps;
    int envs;
    int obs_dim;
    float *obs;
    int *actions;
    float *logp;
    float *values;
    float *rewards;
    unsigned char *dones;
    float *bootstrap;
    unsigned char *active;
    float *advantages;
    float *returns;
} Rollout;

/* Returns 0, or -1 if memory runs out. */
int rollout_init(Rollout *r, int steps, int envs, int obs_dim);
void rollout_free(Rollout *r);

static inline int rollout_index(const Rollout *r, int t, int n) {
    return t * r->envs + n;
}

/* The decision part of step t in env n. Inactive samples store zeros. */
void rollout_record(Rollout *r, int t, int n, int active, const float *obs, const int *action, float logp,
                    float value);

/* The outcome of step t in env n. Inactive samples are forced to reward 0 and done 1. */
void rollout_outcome(Rollout *r, int t, int n, float reward, int done, float bootstrap);

int rollout_active_count(const Rollout *r);

/*
 * GAE(lambda) advantages and returns. last_values[n] is V of each env's observation after
 * the final step, used when a trajectory is still running at the rollout edge. Endings use
 * bootstrap (V(final obs) for the time limit, 0 for death or defeat); masked samples get 0.
 */
void rollout_gae(Rollout *r, const float *last_values, float gamma, float lambda);

#endif
