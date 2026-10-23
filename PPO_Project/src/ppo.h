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

typedef struct {
    double pg_loss;
    double v_loss;
    double entropy;
    double approx_kl; /* mean of (r - 1) - log r */
    double clipfrac;  /* share of samples with |r - 1| > clip */
} PpoLossStats;

/*
 * Clipped PPO loss over b samples, with r = exp(logp - old_logp):
 *   mean(max(-A r, -A clip(r, 1 - c, 1 + c))) - ent_coef * mean(H) + vf_coef * mean(0.5 (V - R)^2).
 * Writes d loss / d logits (b x ACTION_LOGITS) and d loss / d V unless those pointers are NULL.
 */
double ppo_loss(int b, const float *logits, const float *values, const int *actions, const float *old_logp,
                const float *adv, const float *ret, float clip, float vf_coef, float ent_coef, float *g_logits,
                float *g_values, PpoLossStats *stats);

#endif
