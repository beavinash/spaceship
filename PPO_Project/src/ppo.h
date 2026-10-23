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

/* Separate actor (ACTION_LOGITS outputs) and critic (1 output) networks, each with its own Adam state. */
typedef struct {
    Mlp actor;
    Mlp critic;
    Adam opt_actor;
    Adam opt_critic;
} Agent;

/* Returns 0, or -1 if memory runs out. Policy output gain 0.01, value output gain 1. */
int agent_init(Agent *ag, int obs_dim, int hidden, Pcg32 *rng);
void agent_free(Agent *ag);

typedef struct {
    float gamma;
    float lambda;
    float clip;
    float vf_coef;
    float ent_coef;
    float max_grad_norm;
    float lr;
    int epochs;
    int minibatches;
    int norm_adv; /* normalize advantages within each minibatch */
} PpoConfig;

/* Section 3.3 defaults. */
PpoConfig ppo_config_default(void);

typedef struct {
    double pg_loss;
    double v_loss;
    double entropy;
    double approx_kl;
    double clipfrac;
    double explained_var; /* of the critic's rollout values against GAE returns, before the update */
    double grad_norm;     /* mean pre-clip norm */
    int minibatch_updates;
    int samples;
} PpoStats;

/* Active sample indices in a fresh random order; returns how many. */
int ppo_epoch_order(const Rollout *r, Pcg32 *rng, int *order);

/* Minibatch k of m over count shuffled samples is order[lo, hi). */
void ppo_minibatch_bounds(int count, int m, int k, int *lo, int *hi);

/* Epochs of shuffled minibatch updates on the actor and critic from one rollout with GAE already computed. */
void ppo_update(Agent *ag, const Rollout *r, const PpoConfig *cfg, Pcg32 *rng, PpoStats *stats);

#endif
