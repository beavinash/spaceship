#include "ppo.h"

#include <stdlib.h>
#include <string.h>

int rollout_init(Rollout *r, int steps, int envs, int obs_dim) {
    memset(r, 0, sizeof *r);
    r->steps = steps;
    r->envs = envs;
    r->obs_dim = obs_dim;
    size_t n = (size_t)steps * envs;
    r->obs = calloc(n * obs_dim, sizeof *r->obs);
    r->actions = calloc(n * ACTION_HEADS, sizeof *r->actions);
    r->logp = calloc(n, sizeof *r->logp);
    r->values = calloc(n, sizeof *r->values);
    r->rewards = calloc(n, sizeof *r->rewards);
    r->dones = calloc(n, sizeof *r->dones);
    r->bootstrap = calloc(n, sizeof *r->bootstrap);
    r->active = calloc(n, sizeof *r->active);
    r->advantages = calloc(n, sizeof *r->advantages);
    r->returns = calloc(n, sizeof *r->returns);
    if (!r->obs || !r->actions || !r->logp || !r->values || !r->rewards || !r->dones || !r->bootstrap ||
        !r->active || !r->advantages || !r->returns) {
        rollout_free(r);
        return -1;
    }
    return 0;
}

void rollout_free(Rollout *r) {
    free(r->obs);
    free(r->actions);
    free(r->logp);
    free(r->values);
    free(r->rewards);
    free(r->dones);
    free(r->bootstrap);
    free(r->active);
    free(r->advantages);
    free(r->returns);
    memset(r, 0, sizeof *r);
}

void rollout_record(Rollout *r, int t, int n, int active, const float *obs, const int *action, float logp,
                    float value) {
    int i = rollout_index(r, t, n);
    float *o = r->obs + (size_t)i * r->obs_dim;
    int *a = r->actions + (size_t)i * ACTION_HEADS;
    r->active[i] = (unsigned char)(active != 0);
    if (active) {
        memcpy(o, obs, (size_t)r->obs_dim * sizeof *o);
        memcpy(a, action, ACTION_HEADS * sizeof *a);
        r->logp[i] = logp;
        r->values[i] = value;
    } else {
        memset(o, 0, (size_t)r->obs_dim * sizeof *o);
        memset(a, 0, ACTION_HEADS * sizeof *a);
        r->logp[i] = 0.0f;
        r->values[i] = 0.0f;
    }
}

void rollout_outcome(Rollout *r, int t, int n, float reward, int done, float bootstrap) {
    int i = rollout_index(r, t, n);
    if (r->active[i]) {
        r->rewards[i] = reward;
        r->dones[i] = (unsigned char)(done != 0);
        r->bootstrap[i] = done ? bootstrap : 0.0f;
    } else {
        r->rewards[i] = 0.0f;
        r->dones[i] = 1;
        r->bootstrap[i] = 0.0f;
    }
}

int rollout_active_count(const Rollout *r) {
    int count = 0;
    for (int i = 0; i < r->steps * r->envs; i++) {
        count += r->active[i];
    }
    return count;
}
