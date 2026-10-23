#ifndef TRAINER_H
#define TRAINER_H

#include <stdint.h>

#include "env.h"
#include "ppo.h"

/*
 * Independent PPO learners, one Agent per ship, collecting from `envs` arenas in lockstep.
 * Each ship acts on its own observation through its own actor; a ship's samples are masked
 * from its death until its arena resets.
 */
typedef struct {
    int envs;
    int steps;
    int hidden;
    uint64_t seed;
    RewardConfig reward;
    PpoConfig ppo;
} TrainConfig;

/* Section 3.3 defaults: 16 arenas x 128 decisions, 128 hidden units. */
TrainConfig train_config_default(void);

/* Sums over episodes finished since the last trainer_take_episodes call. */
typedef struct {
    int episodes;
    double ret[NUM_SHIPS];
    double team_score;
    double kills;
    double hits;
    double shots;
    double damage;
    double deaths;
    double escapes;
    double spawned;
    double seconds;
} EpisodeTotals;

typedef struct {
    TrainConfig cfg;
    Env *envs;
    Agent agents[NUM_SHIPS];
    Rollout rollouts[NUM_SHIPS];
    Pcg32 action_rng[NUM_SHIPS];
    Pcg32 update_rng[NUM_SHIPS];
    Pcg32 seed_rng;
    float *obs[NUM_SHIPS]; /* envs x OBS_DIM current observations */
    float *logits;         /* envs x ACTION_LOGITS scratch */
    float *values;         /* envs scratch */
    long long decisions;   /* arena decisions collected so far */
    EpisodeTotals finished;
} Trainer;

/* Returns 0, or -1 if memory runs out. */
int trainer_init(Trainer *tr, const TrainConfig *cfg);
void trainer_free(Trainer *tr);

/* Fill every ship's rollout with cfg.steps decisions per arena, then compute GAE. */
void trainer_collect(Trainer *tr);

#endif
