#include "ppo.h"

#include <math.h>
#include <stdio.h>
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

void rollout_gae(Rollout *r, const float *last_values, float gamma, float lambda) {
    for (int n = 0; n < r->envs; n++) {
        float carry = 0.0f;
        for (int t = r->steps - 1; t >= 0; t--) {
            int i = rollout_index(r, t, n);
            if (!r->active[i]) {
                r->advantages[i] = 0.0f;
                r->returns[i] = 0.0f;
                carry = 0.0f;
                continue;
            }
            float next;
            if (r->dones[i]) {
                next = r->bootstrap[i];
                carry = 0.0f;
            } else {
                next = t == r->steps - 1 ? last_values[n] : r->values[rollout_index(r, t + 1, n)];
            }
            float delta = r->rewards[i] + gamma * next - r->values[i];
            carry = delta + gamma * lambda * carry;
            r->advantages[i] = carry;
            r->returns[i] = carry + r->values[i];
        }
    }
}

double ppo_loss(int b, const float *logits, const float *values, const int *actions, const float *old_logp,
                const float *adv, const float *ret, float clip, float vf_coef, float ent_coef, float *g_logits,
                float *g_values, PpoLossStats *stats) {
    double pg = 0.0, vl = 0.0, ent = 0.0, kl = 0.0, clipped = 0.0;
    float g_lp[ACTION_LOGITS], g_h[ACTION_LOGITS];
    const float inv_b = 1.0f / (float)b;
    for (int n = 0; n < b; n++) {
        const float *z = logits + (size_t)n * ACTION_LOGITS;
        const int *a = actions + (size_t)n * ACTION_HEADS;
        float logp = dist_log_prob(z, a);
        float log_ratio = logp - old_logp[n];
        float ratio = expf(log_ratio);
        float clipped_ratio = ratio < 1.0f - clip ? 1.0f - clip : (ratio > 1.0f + clip ? 1.0f + clip : ratio);
        float unclipped_term = -adv[n] * ratio;
        float clipped_term = -adv[n] * clipped_ratio;
        /* max of the two terms; on a tie (ratio inside the range) the unclipped branch carries the gradient */
        int use_unclipped = unclipped_term >= clipped_term;
        pg += use_unclipped ? unclipped_term : clipped_term;
        float h = dist_entropy(z);
        ent += h;
        float err = values[n] - ret[n];
        vl += 0.5 * (double)err * err;
        kl += (ratio - 1.0f) - log_ratio;
        clipped += fabsf(ratio - 1.0f) > clip;
        if (g_logits) {
            float d_logp = use_unclipped ? -adv[n] * ratio : 0.0f;
            dist_log_prob_grad(z, a, g_lp);
            dist_entropy_grad(z, g_h);
            float *g = g_logits + (size_t)n * ACTION_LOGITS;
            for (int k = 0; k < ACTION_LOGITS; k++) {
                g[k] = (d_logp * g_lp[k] - ent_coef * g_h[k]) * inv_b;
            }
        }
        if (g_values) {
            g_values[n] = vf_coef * err * inv_b;
        }
    }
    pg /= b;
    vl /= b;
    ent /= b;
    if (stats) {
        stats->pg_loss = pg;
        stats->v_loss = vl;
        stats->entropy = ent;
        stats->approx_kl = kl / b;
        stats->clipfrac = clipped / b;
    }
    return pg - ent_coef * ent + vf_coef * vl;
}

int agent_init(Agent *ag, int obs_dim, int hidden, Pcg32 *rng) {
    memset(ag, 0, sizeof *ag);
    if (mlp_init(&ag->actor, obs_dim, hidden, ACTION_LOGITS, 0.01f, rng) != 0 ||
        mlp_init(&ag->critic, obs_dim, hidden, 1, 1.0f, rng) != 0 ||
        adam_init(&ag->opt_actor, ag->actor.n_params) != 0 || adam_init(&ag->opt_critic, ag->critic.n_params) != 0) {
        agent_free(ag);
        return -1;
    }
    return 0;
}

void agent_free(Agent *ag) {
    mlp_free(&ag->actor);
    mlp_free(&ag->critic);
    adam_free(&ag->opt_actor);
    adam_free(&ag->opt_critic);
}

PpoConfig ppo_config_default(void) {
    PpoConfig cfg = {
        .gamma = 0.99f,
        .lambda = 0.99f, /* 0.95 left ~3% of the credit for a hit ~1-2 s after the shot */
        .clip = 0.2f,
        .vf_coef = 0.5f,
        .ent_coef = 0.01f,
        .max_grad_norm = 0.5f,
        .lr = 3e-4f,
        .epochs = 4,
        .minibatches = 4,
        .norm_adv = 1,
    };
    return cfg;
}

int ppo_epoch_order(const Rollout *r, Pcg32 *rng, int *order) {
    int count = 0;
    for (int i = 0; i < r->steps * r->envs; i++) {
        if (r->active[i]) {
            order[count++] = i;
        }
    }
    for (int k = count - 1; k > 0; k--) { /* Fisher-Yates */
        int j = (int)pcg32_bounded(rng, (uint32_t)k + 1);
        int tmp = order[k];
        order[k] = order[j];
        order[j] = tmp;
    }
    return count;
}

void ppo_minibatch_bounds(int count, int m, int k, int *lo, int *hi) {
    *lo = (int)((long long)count * k / m);
    *hi = (int)((long long)count * (k + 1) / m);
}

static double explained_variance(const Rollout *r) {
    double n = 0.0, mean_ret = 0.0, mean_err = 0.0;
    for (int i = 0; i < r->steps * r->envs; i++) {
        if (r->active[i]) {
            n += 1.0;
            mean_ret += r->returns[i];
            mean_err += r->returns[i] - r->values[i];
        }
    }
    if (n < 2.0) {
        return 0.0;
    }
    mean_ret /= n;
    mean_err /= n;
    double var_ret = 0.0, var_err = 0.0;
    for (int i = 0; i < r->steps * r->envs; i++) {
        if (r->active[i]) {
            double d = r->returns[i] - mean_ret;
            double e = (r->returns[i] - r->values[i]) - mean_err;
            var_ret += d * d;
            var_err += e * e;
        }
    }
    return var_ret > 0.0 ? 1.0 - var_err / var_ret : 0.0;
}

void ppo_update(Agent *ag, const Rollout *r, const PpoConfig *cfg, Pcg32 *rng, PpoStats *stats) {
    const int total = r->steps * r->envs;
    const int d = r->obs_dim;
    const int cap = total / cfg->minibatches + 1;
    int *order = calloc((size_t)total, sizeof *order);
    float *obs = malloc((size_t)cap * d * sizeof *obs);
    int *act = malloc((size_t)cap * ACTION_HEADS * sizeof *act);
    float *old_logp = malloc((size_t)cap * sizeof *old_logp);
    float *adv = malloc((size_t)cap * sizeof *adv);
    float *ret = malloc((size_t)cap * sizeof *ret);
    float *logits = malloc((size_t)cap * ACTION_LOGITS * sizeof *logits);
    float *values = malloc((size_t)cap * sizeof *values);
    float *g_logits = malloc((size_t)cap * ACTION_LOGITS * sizeof *g_logits);
    float *g_values = malloc((size_t)cap * sizeof *g_values);
    memset(stats, 0, sizeof *stats);
    if (!order || !obs || !act || !old_logp || !adv || !ret || !logits || !values || !g_logits || !g_values) {
        fprintf(stderr, "ppo_update: out of memory\n");
        abort();
    }
    stats->explained_var = explained_variance(r);

    for (int epoch = 0; epoch < cfg->epochs; epoch++) {
        int count = ppo_epoch_order(r, rng, order);
        stats->samples = count;
        for (int k = 0; k < cfg->minibatches; k++) {
            int lo, hi;
            ppo_minibatch_bounds(count, cfg->minibatches, k, &lo, &hi);
            int b = hi - lo;
            if (b <= 0) {
                continue;
            }
            for (int j = 0; j < b; j++) {
                int i = order[lo + j];
                memcpy(obs + (size_t)j * d, r->obs + (size_t)i * d, (size_t)d * sizeof *obs);
                memcpy(act + (size_t)j * ACTION_HEADS, r->actions + (size_t)i * ACTION_HEADS,
                       ACTION_HEADS * sizeof *act);
                old_logp[j] = r->logp[i];
                adv[j] = r->advantages[i];
                ret[j] = r->returns[i];
            }
            if (cfg->norm_adv && b > 1) {
                double mean = 0.0, sq = 0.0;
                for (int j = 0; j < b; j++) {
                    mean += adv[j];
                }
                mean /= b;
                for (int j = 0; j < b; j++) {
                    sq += (adv[j] - mean) * (adv[j] - mean);
                }
                double sd = sqrt(sq / (b - 1)); /* unbiased, as torch.std */
                for (int j = 0; j < b; j++) {
                    adv[j] = (float)((adv[j] - mean) / (sd + 1e-8));
                }
            }
            mlp_zero_grad(&ag->actor);
            mlp_zero_grad(&ag->critic);
            memcpy(logits, mlp_forward(&ag->actor, obs, b), (size_t)b * ACTION_LOGITS * sizeof *logits);
            memcpy(values, mlp_forward(&ag->critic, obs, b), (size_t)b * sizeof *values);
            PpoLossStats ls;
            ppo_loss(b, logits, values, act, old_logp, adv, ret, cfg->clip, cfg->vf_coef, cfg->ent_coef, g_logits,
                     g_values, &ls);
            mlp_backward(&ag->actor, obs, g_logits, b);
            mlp_backward(&ag->critic, obs, g_values, b);
            float *groups[2] = {ag->actor.grads, ag->critic.grads};
            const int sizes[2] = {ag->actor.n_params, ag->critic.n_params};
            float norm = clip_grad_norm(groups, sizes, 2, cfg->max_grad_norm);
            adam_step(&ag->opt_actor, ag->actor.params, ag->actor.grads, cfg->lr);
            adam_step(&ag->opt_critic, ag->critic.params, ag->critic.grads, cfg->lr);

            stats->pg_loss += ls.pg_loss;
            stats->v_loss += ls.v_loss;
            stats->entropy += ls.entropy;
            stats->approx_kl += ls.approx_kl;
            stats->clipfrac += ls.clipfrac;
            stats->grad_norm += norm;
            stats->minibatch_updates++;
        }
    }
    if (stats->minibatch_updates > 0) {
        double m = stats->minibatch_updates;
        stats->pg_loss /= m;
        stats->v_loss /= m;
        stats->entropy /= m;
        stats->approx_kl /= m;
        stats->clipfrac /= m;
        stats->grad_norm /= m;
    }
    free(order);
    free(obs);
    free(act);
    free(old_logp);
    free(adv);
    free(ret);
    free(logits);
    free(values);
    free(g_logits);
    free(g_values);
}
