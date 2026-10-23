#include "eval.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dist.h"
#include "hash.h"
#include "legacy.h"
#include "trainer.h"

typedef struct {
    LegacyTables *tables[NUM_SHIPS];
    int owned;
} LegacyPolicy;

void policy_random(Policy *p) {
    memset(p, 0, sizeof *p);
    p->kind = POLICY_RANDOM;
    p->name = "random";
    p->action_repeat = ACTION_REPEAT;
}

int policy_ppo(Policy *p, const char *checkpoint, int sample) {
    memset(p, 0, sizeof *p);
    int n_nets, sizes[4];
    int rc = checkpoint_shape(checkpoint, &n_nets, sizes, NULL);
    if (rc != 0) {
        return rc;
    }
    if (n_nets != CHECKPOINT_NETS || sizes[0] != OBS_DIM || sizes[3] != ACTION_LOGITS || sizes[1] <= 0 ||
        sizes[1] > 4096) {
        return CKPT_SHAPE;
    }
    Agent *agents = calloc(NUM_SHIPS, sizeof *agents);
    if (!agents) {
        return CKPT_IO;
    }
    Pcg32 rng;
    pcg32_seed(&rng, 0u, 0u); /* placeholder weights, replaced by the checkpoint */
    for (int k = 0; k < NUM_SHIPS; k++) {
        if (agent_init(&agents[k], OBS_DIM, sizes[1], &rng) != 0) {
            rc = CKPT_IO;
        }
    }
    Mlp *nets[CHECKPOINT_NETS];
    ship_agent_nets(agents, nets);
    if (rc == 0) {
        rc = checkpoint_load(checkpoint, nets, CHECKPOINT_NETS, NULL);
    }
    if (rc != 0) {
        for (int k = 0; k < NUM_SHIPS; k++) {
            agent_free(&agents[k]);
        }
        free(agents);
        return rc;
    }
    p->kind = POLICY_PPO;
    p->name = "ppo";
    p->action_repeat = ACTION_REPEAT;
    p->sample = sample;
    p->data = agents;
    return 0;
}

void policy_legacy_tables(Policy *p, void *const tables[NUM_SHIPS]) {
    memset(p, 0, sizeof *p);
    LegacyPolicy *lp = calloc(1, sizeof *lp);
    if (!lp) {
        abort();
    }
    for (int k = 0; k < NUM_SHIPS; k++) {
        lp->tables[k] = tables[k];
    }
    p->kind = POLICY_LEGACY;
    p->name = "legacy";
    p->action_repeat = 1;
    p->data = lp;
}

int policy_legacy(Policy *p, const char *game_data_dir) {
    LegacyTables *t[NUM_SHIPS] = {NULL, NULL, NULL};
    for (int k = 0; k < NUM_SHIPS; k++) {
        char path[4096];
        snprintf(path, sizeof path, "%s/trained%d.json", game_data_dir, k);
        t[k] = legacy_load(path);
        if (!t[k]) {
            for (int j = 0; j < k; j++) {
                legacy_free(t[j]);
            }
            memset(p, 0, sizeof *p);
            return -1;
        }
    }
    policy_legacy_tables(p, (void *const *)t);
    ((LegacyPolicy *)p->data)->owned = 1;
    return 0;
}

void policy_free(Policy *p) {
    if (p->kind == POLICY_LEGACY && p->data) {
        LegacyPolicy *lp = p->data;
        for (int k = 0; lp->owned && k < NUM_SHIPS; k++) {
            legacy_free(lp->tables[k]);
        }
        free(lp);
    }
    if (p->kind == POLICY_PPO && p->data) {
        Agent *agents = p->data;
        for (int k = 0; k < NUM_SHIPS; k++) {
            agent_free(&agents[k]);
        }
        free(agents);
    }
    memset(p, 0, sizeof *p);
}

void policy_act(Policy *p, const Env *env, Pcg32 *rng, ShipAction out[NUM_SHIPS]) {
    int targets[NUM_SHIPS];
    if (p->kind == POLICY_LEGACY) {
        legacy_targets(env, ((LegacyPolicy *)p->data)->tables, targets);
    }
    for (int k = 0; k < NUM_SHIPS; k++) {
        out[k] = (ShipAction){H_STAY, V_STAY, 0};
        if (!env->ships[k].alive) {
            continue;
        }
        switch (p->kind) {
        case POLICY_PPO: {
            float obs[OBS_DIM];
            int a[ACTION_HEADS];
            Agent *agents = p->data;
            env_observe(env, k, obs);
            const float *z = mlp_forward(&agents[k].actor, obs, 1);
            if (p->sample) {
                dist_sample(z, rng, a);
            } else {
                dist_argmax(z, a);
            }
            out[k] = (ShipAction){a[0], a[1], a[2]};
            break;
        }
        case POLICY_LEGACY: {
            const LegacyTables *t = ((LegacyPolicy *)p->data)->tables[k];
            const Ship *s = &env->ships[k];
            float dx = 0.0f, dy = 0.0f; /* no target: upstream looks up the zero displacement */
            if (targets[k] >= 0) {
                dx = s->x - env->enemies[targets[k]].x;
                dy = s->y - env->enemies[targets[k]].y;
            }
            int x, y;
            legacy_cell(dx, dy, &x, &y);
            int vert = (int)pcg32_bounded(rng, 100) + 1;
            int hori = (int)pcg32_bounded(rng, 100) + 1;
            int shoot = (int)pcg32_bounded(rng, 100) + 1;
            out[k] = legacy_compute_action(t->move_h[x][y], t->move_v[x][y], t->shoot[x][y], vert, hori, shoot);
            break;
        }
        case POLICY_RANDOM:
        default:
            out[k] = (ShipAction){(int)pcg32_bounded(rng, 3), (int)pcg32_bounded(rng, 3), (int)pcg32_bounded(rng, 2)};
            break;
        }
    }
}

int eval_split(const char *split, int *first, int *count) {
    if (!strcmp(split, "dev")) {
        *first = 1000;
        *count = 20;
        return 0;
    }
    if (!strcmp(split, "test")) {
        *first = 10000;
        *count = 100;
        return 0;
    }
    return -1;
}

static void run_episode(Policy *p, int seed, EpisodeResult *r) {
    Env env;
    env_reset(&env, (uint64_t)seed);
    env.action_repeat = p->action_repeat;
    Pcg32 rng;
    pcg32_seed(&rng, (uint64_t)seed, 0xe7a1u);
    RewardConfig cfg = reward_config_default();
    float rewards[NUM_SHIPS];
    while (!env.terminated && !env.truncated) {
        ShipAction a[NUM_SHIPS];
        policy_act(p, &env, &rng, a);
        env_step(&env, a);
        env_rewards(&env, &cfg, rewards);
    }
    const EpisodeStats *s = &env.stats;
    memset(r, 0, sizeof *r);
    r->seed = seed;
    double *score[NUM_SHIPS] = {&r->score0, &r->score1, &r->score2};
    for (int k = 0; k < NUM_SHIPS; k++) {
        *score[k] = s->score[k];
        r->team_score += s->score[k];
        r->kills += s->kills[k];
        r->hits += s->hits[k];
        r->shots += s->shots[k];
        r->damage_taken += s->damage_taken[k];
        r->deaths += s->died[k];
        r->alive_at_end += env.ships[k].alive;
        r->survival_seconds += (double)s->alive_ticks[k] / TICK_HZ / NUM_SHIPS;
        r->return_mean += s->episode_return[k] / NUM_SHIPS;
    }
    r->accuracy = r->shots > 0 ? r->hits / r->shots : 0.0;
    r->escapes = s->escapes;
    r->spawned = s->spawned;
    r->episode_seconds = (double)env.tick / TICK_HZ;
}

int eval_run(Policy *p, const char *split, int allow_test, EpisodeResult *out, int *n) {
    int first, count;
    if (eval_split(split, &first, &count) != 0) {
        fprintf(stderr, "eval: unknown split %s (use dev or test)\n", split);
        return -1;
    }
    if (!strcmp(split, "test") && !allow_test) {
        fputs("eval: the test split is held out; pass --final for the one final evaluation\n", stderr);
        return -1;
    }
    for (int i = 0; i < count; i++) {
        run_episode(p, first + i, &out[i]);
    }
    *n = count;
    return 0;
}

Summary summarize(const EpisodeResult *res, int n, size_t field) {
    Summary s = {0, 0, 0};
    if (n <= 0) {
        return s;
    }
    for (int i = 0; i < n; i++) {
        s.mean += *(const double *)((const char *)&res[i] + field);
    }
    s.mean /= n;
    if (n > 1) {
        double sq = 0.0;
        for (int i = 0; i < n; i++) {
            double d = *(const double *)((const char *)&res[i] + field) - s.mean;
            sq += d * d;
        }
        s.std = sqrt(sq / (n - 1));
        s.ci95 = 1.96 * s.std / sqrt((double)n);
    }
    return s;
}

/* FNV-1a 64 of a file's bytes, as 16 hex digits; "unavailable" if it cannot be read. */
static void hash_file(const char *path, char out[32]) {
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) {
        snprintf(out, 32, "unavailable");
        return;
    }
    uint64_t h = FNV_OFFSET;
    char buf[65536];
    for (;;) {
        size_t got = fread(buf, 1, sizeof buf, f);
        h = fnv1a(h, buf, got);
        if (got < sizeof buf) {
            break;
        }
    }
    fclose(f);
    snprintf(out, 32, "%016llx", (unsigned long long)h);
}

typedef struct {
    const char *name;
    size_t offset;
} Metric;

static const Metric METRICS[] = {
    {"team_score", offsetof(EpisodeResult, team_score)},
    {"score_ship0", offsetof(EpisodeResult, score0)},
    {"score_ship1", offsetof(EpisodeResult, score1)},
    {"score_ship2", offsetof(EpisodeResult, score2)},
    {"kills", offsetof(EpisodeResult, kills)},
    {"hits", offsetof(EpisodeResult, hits)},
    {"shots", offsetof(EpisodeResult, shots)},
    {"accuracy", offsetof(EpisodeResult, accuracy)},
    {"damage_taken", offsetof(EpisodeResult, damage_taken)},
    {"deaths", offsetof(EpisodeResult, deaths)},
    {"alive_at_end", offsetof(EpisodeResult, alive_at_end)},
    {"survival_seconds", offsetof(EpisodeResult, survival_seconds)},
    {"escapes", offsetof(EpisodeResult, escapes)},
    {"spawned", offsetof(EpisodeResult, spawned)},
    {"episode_seconds", offsetof(EpisodeResult, episode_seconds)},
    {"return_mean", offsetof(EpisodeResult, return_mean)},
};
#define N_METRICS ((int)(sizeof METRICS / sizeof METRICS[0]))

int eval_write_json(const char *path, const ReportMeta *meta, const Policy *p, const EpisodeResult *res, int n) {
    FILE *f = fopen(path, "w");
    if (!f) {
        return -1;
    }
    char ckpt_hash[32], bin_hash[32];
    hash_file(meta->checkpoint, ckpt_hash);
    hash_file(meta->binary, bin_hash);
    RewardConfig rc = reward_config_default();
    fprintf(f, "{\n  \"policy\": \"%s\",\n  \"split\": \"%s\",\n  \"episodes\": %d,\n", meta->policy, meta->split, n);
    fprintf(f, "  \"action_repeat\": %d,\n  \"sample\": %s,\n", p->action_repeat, p->sample ? "true" : "false");
    if (meta->checkpoint) {
        fprintf(f, "  \"checkpoint\": {\"path\": \"%s\", \"fnv1a\": \"%s\"},\n", meta->checkpoint, ckpt_hash);
    } else {
        fputs("  \"checkpoint\": null,\n", f);
    }
    fprintf(f, "  \"binary_fnv1a\": \"%s\",\n", bin_hash);
    fprintf(f,
            "  \"env\": {\"num_ships\": %d, \"ship_health\": %g, \"ship_ammo\": %d, \"episode_seconds\": %d, "
            "\"tick_hz\": %d},\n",
            NUM_SHIPS, SHIP_HEALTH, SHIP_AMMO, EPISODE_SECONDS, TICK_HZ);
    fprintf(f,
            "  \"reward\": {\"kill_points\": %g, \"hit\": %g, \"shot\": %g, \"damage\": %g, \"death\": %g, "
            "\"escape\": %g, \"ally_share\": %g, \"align\": %g},\n",
            rc.kill_points, rc.hit, rc.shot, rc.damage, rc.death, rc.escape, rc.ally_share, rc.align);
    fputs("  \"summary\": {\n", f);
    for (int m = 0; m < N_METRICS; m++) {
        Summary s = summarize(res, n, METRICS[m].offset);
        fprintf(f, "    \"%s\": {\"mean\": %.6f, \"std\": %.6f, \"ci95\": %.6f}%s\n", METRICS[m].name, s.mean, s.std,
                s.ci95, m + 1 < N_METRICS ? "," : "");
    }
    fputs("  },\n  \"per_seed\": [\n", f);
    for (int i = 0; i < n; i++) {
        fprintf(f, "    {\"seed\": %d", res[i].seed);
        for (int m = 0; m < N_METRICS; m++) {
            fprintf(f, ", \"%s\": %.6f", METRICS[m].name, *(const double *)((const char *)&res[i] + METRICS[m].offset));
        }
        fprintf(f, "}%s\n", i + 1 < n ? "," : "");
    }
    fputs("  ]\n}\n", f);
    return fclose(f) == 0 ? 0 : -1;
}
