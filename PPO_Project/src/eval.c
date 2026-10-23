#include "eval.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

void policy_random(Policy *p) {
    memset(p, 0, sizeof *p);
    p->kind = POLICY_RANDOM;
    p->name = "random";
    p->action_repeat = ACTION_REPEAT;
}

void policy_act(Policy *p, const Env *env, Pcg32 *rng, ShipAction out[NUM_SHIPS]) {
    for (int k = 0; k < NUM_SHIPS; k++) {
        out[k] = (ShipAction){H_STAY, V_STAY, 0};
        if (!env->ships[k].alive) {
            continue;
        }
        switch (p->kind) {
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
    uint64_t h = 14695981039346656037ULL;
    int c;
    while ((c = fgetc(f)) != EOF) {
        h ^= (unsigned char)c;
        h *= 1099511628211ULL;
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
            "\"escape\": %g, \"ally_share\": %g},\n",
            rc.kill_points, rc.hit, rc.shot, rc.damage, rc.death, rc.escape, rc.ally_share);
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
