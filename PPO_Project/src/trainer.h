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
    int episode_ticks; /* truncation limit per arena episode; EPISODE_TICKS (120 s) by default */
    uint64_t seed;
    RewardConfig reward;
    PpoConfig ppo;
} TrainConfig;

/* Section 3.3 defaults: 16 arenas x 128 decisions, 128 hidden units. */
TrainConfig train_config_default(void);

/* One finished arena episode, for rolling training metrics. */
typedef struct {
    float ret[NUM_SHIPS];
    float team_score;
    float kills;
    float hits;
    float shots;
    float damage;
    float deaths;
    float escapes;
    float spawned;
    float seconds;
} EpisodeRecord;

#define EPISODE_WINDOW 32

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
    long long episodes;    /* arena episodes finished so far */
    EpisodeRecord recent[EPISODE_WINDOW]; /* the most recent finished episodes, ring-indexed by episodes */
} Trainer;

/* Returns 0, or -1 if memory runs out. */
int trainer_init(Trainer *tr, const TrainConfig *cfg);
void trainer_free(Trainer *tr);

/* Fill every ship's rollout with cfg.steps decisions per arena, then compute GAE. */
void trainer_collect(Trainer *tr);

#endif
