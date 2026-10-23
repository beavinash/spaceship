#include "trainer.h"

#include <stdlib.h>
#include <string.h>

#include "dist.h"

TrainConfig train_config_default(void) {
    TrainConfig cfg = {
        .envs = 16,
        .steps = 128,
        .hidden = 128,
        .seed = 1,
        .reward = reward_config_default(),
        .ppo = ppo_config_default(),
    };
    return cfg;
}

/* Training scenarios come from a seeded stream and stay clear of the evaluation seeds (< 100000). */
static uint64_t next_episode_seed(Trainer *tr) {
    uint64_t s = ((uint64_t)pcg32_next(&tr->seed_rng) << 32) | pcg32_next(&tr->seed_rng);
    return s < 100000u ? s + 100000u : s;
}

static void observe_all(Trainer *tr, int n) {
    for (int k = 0; k < NUM_SHIPS; k++) {
        env_observe(&tr->envs[n], k, tr->obs[k] + (size_t)n * OBS_DIM);
    }
}

int trainer_init(Trainer *tr, const TrainConfig *cfg) {
    memset(tr, 0, sizeof *tr);
    tr->cfg = *cfg;
    pcg32_seed(&tr->seed_rng, cfg->seed, 0x5eed5u);
    Pcg32 init_rng;
    pcg32_seed(&init_rng, cfg->seed, 0x1417u);
    tr->envs = calloc((size_t)cfg->envs, sizeof *tr->envs);
    tr->logits = calloc((size_t)cfg->envs * ACTION_LOGITS, sizeof *tr->logits);
    tr->values = calloc((size_t)cfg->envs, sizeof *tr->values);
    if (!tr->envs || !tr->logits || !tr->values) {
        trainer_free(tr);
        return -1;
    }
    for (int k = 0; k < NUM_SHIPS; k++) {
        pcg32_seed(&tr->action_rng[k], cfg->seed, 0xac70u + (uint64_t)k);
        pcg32_seed(&tr->update_rng[k], cfg->seed, 0x0bd7u + (uint64_t)k);
        tr->obs[k] = calloc((size_t)cfg->envs * OBS_DIM, sizeof *tr->obs[k]);
        if (!tr->obs[k] || agent_init(&tr->agents[k], OBS_DIM, cfg->hidden, &init_rng) != 0 ||
            rollout_init(&tr->rollouts[k], cfg->steps, cfg->envs, OBS_DIM) != 0) {
            trainer_free(tr);
            return -1;
        }
    }
    for (int n = 0; n < cfg->envs; n++) {
        env_reset(&tr->envs[n], next_episode_seed(tr));
        observe_all(tr, n);
    }
    return 0;
}

void trainer_free(Trainer *tr) {
    for (int k = 0; k < NUM_SHIPS; k++) {
        agent_free(&tr->agents[k]);
        rollout_free(&tr->rollouts[k]);
        free(tr->obs[k]);
    }
    free(tr->envs);
    free(tr->logits);
    free(tr->values);
    memset(tr, 0, sizeof *tr);
}

static void add_episode(EpisodeTotals *tot, const Env *e) {
    const EpisodeStats *s = &e->stats;
    tot->episodes++;
    for (int k = 0; k < NUM_SHIPS; k++) {
        tot->ret[k] += s->episode_return[k];
        tot->team_score += s->score[k];
        tot->kills += s->kills[k];
        tot->hits += s->hits[k];
        tot->shots += s->shots[k];
        tot->damage += s->damage_taken[k];
        tot->deaths += s->died[k];
    }
    tot->escapes += s->escapes;
    tot->spawned += s->spawned;
    tot->seconds += (double)e->tick / TICK_HZ;
}

void trainer_collect(Trainer *tr) {
    const int envs = tr->cfg.envs;
    ShipAction *acts = malloc((size_t)envs * NUM_SHIPS * sizeof *acts);
    if (!acts) {
        abort();
    }
    for (int t = 0; t < tr->cfg.steps; t++) {
        for (int k = 0; k < NUM_SHIPS; k++) {
            Agent *ag = &tr->agents[k];
            memcpy(tr->logits, mlp_forward(&ag->actor, tr->obs[k], envs), (size_t)envs * ACTION_LOGITS * sizeof(float));
            memcpy(tr->values, mlp_forward(&ag->critic, tr->obs[k], envs), (size_t)envs * sizeof(float));
            for (int n = 0; n < envs; n++) {
                int a[ACTION_HEADS] = {H_STAY, V_STAY, 0};
                int alive = tr->envs[n].ships[k].alive;
                float logp = 0.0f;
                if (alive) {
                    const float *z = tr->logits + (size_t)n * ACTION_LOGITS;
                    dist_sample(z, &tr->action_rng[k], a);
                    logp = dist_log_prob(z, a);
                }
                rollout_record(&tr->rollouts[k], t, n, alive, tr->obs[k] + (size_t)n * OBS_DIM, a, logp,
                               tr->values[n]);
                acts[(size_t)n * NUM_SHIPS + k] = (ShipAction){a[0], a[1], a[2]};
            }
        }
        for (int n = 0; n < envs; n++) {
            Env *e = &tr->envs[n];
            float rewards[NUM_SHIPS];
            env_step(e, acts + (size_t)n * NUM_SHIPS);
            env_rewards(e, &tr->cfg.reward, rewards);
            int ended = e->terminated || e->truncated;
            for (int k = 0; k < NUM_SHIPS; k++) {
                float bootstrap = 0.0f;
                if (e->truncated && e->ships[k].alive) {
                    float final_obs[OBS_DIM];
                    env_observe(e, k, final_obs);
                    bootstrap = mlp_forward(&tr->agents[k].critic, final_obs, 1)[0];
                }
                int done = ended || !e->ships[k].alive;
                rollout_outcome(&tr->rollouts[k], t, n, rewards[k], done, bootstrap);
            }
            if (ended) {
                add_episode(&tr->finished, e);
                env_reset(e, next_episode_seed(tr));
            }
            observe_all(tr, n);
        }
        tr->decisions += envs;
    }
    for (int k = 0; k < NUM_SHIPS; k++) {
        memcpy(tr->values, mlp_forward(&tr->agents[k].critic, tr->obs[k], envs), (size_t)envs * sizeof(float));
        rollout_gae(&tr->rollouts[k], tr->values, tr->cfg.ppo.gamma, tr->cfg.ppo.lambda);
    }
    free(acts);
}
