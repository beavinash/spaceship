#ifndef EVAL_H
#define EVAL_H

#include <stddef.h>

#include "env.h"

/*
 * Reproducible evaluation on fixed scenario seeds: "dev" = 1000-1019 (checkpoint selection),
 * "test" = 10000-10099 (held out; only for an explicit final run). Each scenario is one full
 * episode under the default reward, with the action RNG seeded from the scenario seed.
 */

typedef enum { POLICY_RANDOM, POLICY_PPO, POLICY_LEGACY } PolicyKind;

typedef struct {
    PolicyKind kind;
    const char *name;
    int action_repeat; /* ticks per decision */
    int sample;        /* PPO: sample actions instead of taking the most likely one */
    void *data;        /* policy-specific state */
} Policy;

/* Uniformly random heads, 4-tick decisions like PPO. */
void policy_random(Policy *p);

/*
 * Each ship acts through its own actor from a training checkpoint (hidden size read from the file).
 * Greedy (most likely index per head) unless sample is set. Returns 0 or a CKPT_ error.
 */
int policy_ppo(Policy *p, const char *checkpoint, int sample);

/*
 * Upstream main's frozen logistic-regression tables, one per ship (Game_Data/trained{0,1,2}.json
 * under game_data_dir), deciding every tick as upstream does. Returns 0, or -1 if a file is unreadable.
 */
int policy_legacy(Policy *p, const char *game_data_dir);

/* The same policy over caller-owned tables (not freed by policy_free). */
void policy_legacy_tables(Policy *p, void *const tables[NUM_SHIPS]);

/* Releases what a policy constructor allocated. */
void policy_free(Policy *p);

/* Actions for every ship; dead ships get stay / no fire. */
void policy_act(Policy *p, const Env *env, Pcg32 *rng, ShipAction out[NUM_SHIPS]);

typedef struct {
    int seed;
    double team_score; /* upstream points, all ships */
    double score0, score1, score2;
    double kills;
    double hits;
    double shots;
    double accuracy; /* hits / shots */
    double damage_taken;
    double deaths;
    double alive_at_end;
    double survival_seconds; /* mean per ship */
    double escapes;
    double spawned;
    double episode_seconds;
    double return_mean; /* shaped return, mean per ship */
} EpisodeResult;

typedef struct {
    double mean;
    double std;  /* sample standard deviation (n - 1) */
    double ci95; /* 1.96 * std / sqrt(n), normal approximation */
} Summary;

/* Summary of the double field at byte offset `field` of each result. */
Summary summarize(const EpisodeResult *res, int n, size_t field);

/* First seed and scenario count of a named split; -1 for an unknown name. */
int eval_split(const char *split, int *first, int *count);

/* One episode per scenario of the split into out (room for 100). allow_test gates the held-out split. */
int eval_run(Policy *p, const char *split, int allow_test, EpisodeResult *out, int *n);

typedef struct {
    const char *policy;
    const char *split;
    const char *checkpoint; /* path or NULL; its FNV-1a hash goes in the report */
    const char *binary;     /* path of the evaluating executable or NULL */
} ReportMeta;

/* JSON report: metadata, per-seed rows, and mean / std / ci95 for every metric. */
int eval_write_json(const char *path, const ReportMeta *meta, const Policy *p, const EpisodeResult *res, int n);

#endif
