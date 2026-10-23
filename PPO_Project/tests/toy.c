#include "toy.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dist.h"
#include "ppo.h"

static float place(Pcg32 *rng) {
    return 1.8f * pcg32_float(rng) - 0.9f;
}

static void new_episode(ToyEnv *e) {
    e->px = place(&e->rng);
    e->py = place(&e->rng);
    e->tx = place(&e->rng);
    e->ty = place(&e->rng);
    e->steps = 0;
}

void toy_reset(ToyEnv *e, uint64_t seed) {
    pcg32_seed(&e->rng, seed, 0x70e5u);
    new_episode(e);
}

static float clamp_unit(float v) {
    return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
}

static float distance(const ToyEnv *e) {
    float dx = e->tx - e->px, dy = e->ty - e->py;
    return sqrtf(dx * dx + dy * dy);
}

float toy_step(ToyEnv *e, const int *action, int *terminated, int *truncated) {
    float before = distance(e);
    e->px = clamp_unit(e->px + 0.1f * (float)(action[0] - 1));
    e->py = clamp_unit(e->py + 0.1f * (float)(action[1] - 1));
    e->steps++;
    float reward = -0.01f + (before - distance(e)); /* progress shaping */
    *terminated = 0;
    if (action[2]) {
        if (distance(e) < 0.15f) {
            reward += 1.0f;
            *terminated = 1;
        } else {
            reward -= 0.05f;
        }
    }
    *truncated = !*terminated && e->steps >= TOY_MAX_STEPS;
    return reward;
}

void toy_observe(const ToyEnv *e, float *obs) {
    obs[0] = e->px;
    obs[1] = e->py;
    obs[2] = e->tx - e->px;
    obs[3] = e->ty - e->py;
}

#define TOY_ENVS 16
#define TOY_STEPS 32
#define TOY_WINDOW 100

double toy_train(uint64_t seed, int updates, ToyResult *out) {
    Pcg32 rng;
    pcg32_seed(&rng, seed, 0xa9e7u);
    Agent ag;
    Rollout r;
    if (agent_init(&ag, TOY_OBS, 32, &rng) != 0 || rollout_init(&r, TOY_STEPS, TOY_ENVS, TOY_OBS) != 0) {
        abort();
    }
    PpoConfig cfg = ppo_config_default();
    cfg.lr = 1e-3f;
    ToyEnv envs[TOY_ENVS];
    float ep_return[TOY_ENVS] = {0};
    for (int n = 0; n < TOY_ENVS; n++) {
        toy_reset(&envs[n], seed * 1000u + (uint64_t)n);
    }
    double window[TOY_WINDOW] = {0};
    unsigned char hit[TOY_WINDOW] = {0};
    int finished = 0;
    float obs[TOY_ENVS * TOY_OBS], logits[TOY_ENVS * ACTION_LOGITS], values[TOY_ENVS], last[TOY_ENVS];

    for (int u = 0; u < updates; u++) {
        for (int t = 0; t < TOY_STEPS; t++) {
            for (int n = 0; n < TOY_ENVS; n++) {
                toy_observe(&envs[n], obs + n * TOY_OBS);
            }
            memcpy(logits, mlp_forward(&ag.actor, obs, TOY_ENVS), sizeof logits);
            memcpy(values, mlp_forward(&ag.critic, obs, TOY_ENVS), sizeof values);
            for (int n = 0; n < TOY_ENVS; n++) {
                int a[ACTION_HEADS];
                const float *z = logits + n * ACTION_LOGITS;
                dist_sample(z, &rng, a);
                rollout_record(&r, t, n, 1, obs + n * TOY_OBS, a, dist_log_prob(z, a), values[n]);
                int terminated, truncated;
                float reward = toy_step(&envs[n], a, &terminated, &truncated);
                ep_return[n] += reward;
                float bootstrap = 0.0f;
                if (truncated) {
                    float final_obs[TOY_OBS];
                    toy_observe(&envs[n], final_obs);
                    bootstrap = mlp_forward(&ag.critic, final_obs, 1)[0];
                }
                rollout_outcome(&r, t, n, reward, terminated || truncated, bootstrap);
                if (terminated || truncated) {
                    window[finished % TOY_WINDOW] = ep_return[n];
                    hit[finished % TOY_WINDOW] = (unsigned char)terminated;
                    finished++;
                    ep_return[n] = 0.0f;
                    new_episode(&envs[n]);
                }
            }
        }
        for (int n = 0; n < TOY_ENVS; n++) {
            toy_observe(&envs[n], obs + n * TOY_OBS);
        }
        memcpy(last, mlp_forward(&ag.critic, obs, TOY_ENVS), sizeof last);
        rollout_gae(&r, last, cfg.gamma, cfg.lambda);
        PpoStats st;
        ppo_update(&ag, &r, &cfg, &rng, &st);
    }
    int count = finished < TOY_WINDOW ? finished : TOY_WINDOW;
    double mean = 0.0;
    int hits = 0;
    for (int k = 0; k < count; k++) {
        mean += window[k];
        hits += hit[k];
    }
    if (out) {
        out->updates = updates;
        out->episodes = finished;
        out->hit_rate = count ? (double)hits / count : 0.0;
    }
    agent_free(&ag);
    rollout_free(&r);
    return count ? mean / count : 0.0;
}

double toy_random_return(uint64_t seed, int episodes) {
    ToyEnv e;
    toy_reset(&e, seed);
    Pcg32 rng;
    pcg32_seed(&rng, seed, 0x4a7du);
    double total = 0.0;
    for (int k = 0; k < episodes; k++) {
        int terminated = 0, truncated = 0;
        while (!terminated && !truncated) {
            int a[ACTION_HEADS] = {(int)pcg32_bounded(&rng, 3), (int)pcg32_bounded(&rng, 3),
                                   (int)pcg32_bounded(&rng, 2)};
            total += toy_step(&e, a, &terminated, &truncated);
        }
        new_episode(&e);
    }
    return total / episodes;
}
