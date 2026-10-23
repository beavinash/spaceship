#define _POSIX_C_SOURCE 200809L

#include "train.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define METRICS_HEADER                                                                                         \
    "update,decisions,seconds,decisions_per_sec,lr,episodes,ep_return_mean,ep_return_ship0,ep_return_ship1,"    \
    "ep_return_ship2,team_score,kills,hits,shots,accuracy,damage_taken,deaths,escapes,spawned,ep_seconds,"      \
    "pg_loss,v_loss,entropy,approx_kl,clipfrac,explained_var,grad_norm"

RunConfig run_config_default(void) {
    RunConfig rc = {
        .train = train_config_default(),
        .total_decisions = 5000000,
        .max_seconds = 3600.0,
        .run_dir = NULL,
    };
    return rc;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* mkdir -p */
static int make_dirs(const char *path) {
    char buf[4096];
    if (snprintf(buf, sizeof buf, "%s", path) >= (int)sizeof buf) {
        return -1;
    }
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(buf, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }
    return (mkdir(buf, 0755) != 0 && errno != EEXIST) ? -1 : 0;
}

static void write_episode_columns(FILE *f, const Trainer *tr) {
    int n = tr->episodes < EPISODE_WINDOW ? (int)tr->episodes : EPISODE_WINDOW;
    if (n == 0) {
        fputs(",,,,,,,,,,,,,,", f); /* the 14 episode columns stay empty */
        return;
    }
    EpisodeRecord sum = {0};
    for (int i = 0; i < n; i++) {
        const EpisodeRecord *r = &tr->recent[i];
        for (int k = 0; k < NUM_SHIPS; k++) {
            sum.ret[k] += r->ret[k];
        }
        sum.team_score += r->team_score;
        sum.kills += r->kills;
        sum.hits += r->hits;
        sum.shots += r->shots;
        sum.damage += r->damage;
        sum.deaths += r->deaths;
        sum.escapes += r->escapes;
        sum.spawned += r->spawned;
        sum.seconds += r->seconds;
    }
    double mean_ret = 0.0;
    for (int k = 0; k < NUM_SHIPS; k++) {
        mean_ret += sum.ret[k] / (double)n / NUM_SHIPS;
    }
    fprintf(f, ",%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f", mean_ret,
            sum.ret[0] / n, sum.ret[1] / n, sum.ret[2] / n, sum.team_score / n, sum.kills / n, sum.hits / n,
            sum.shots / n, sum.shots > 0 ? sum.hits / sum.shots : 0.0, sum.damage / n, sum.deaths / n,
            sum.escapes / n, sum.spawned / n, sum.seconds / n);
}

int train_run(const RunConfig *rc) {
    if (!rc->run_dir || make_dirs(rc->run_dir) != 0) {
        fprintf(stderr, "train: cannot create run directory %s\n", rc->run_dir ? rc->run_dir : "(none)");
        return -1;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/metrics.csv", rc->run_dir);
    FILE *metrics = fopen(path, "w");
    if (!metrics) {
        fprintf(stderr, "train: cannot write %s\n", path);
        return -1;
    }
    fprintf(metrics, "%s\n", METRICS_HEADER);

    Trainer tr;
    if (trainer_init(&tr, &rc->train) != 0) {
        fclose(metrics);
        return -1;
    }
    const long long per_update = (long long)rc->train.envs * rc->train.steps;
    const long long planned = (rc->total_decisions + per_update - 1) / per_update;
    const double start = now_seconds();
    int rc_out = 0;

    for (long long u = 1; tr.decisions < rc->total_decisions; u++) {
        PpoConfig pc = tr.cfg.ppo;
        pc.lr = (float)(rc->train.ppo.lr * (1.0 - (double)(u - 1) / (double)planned));
        trainer_collect(&tr);
        PpoStats st[NUM_SHIPS];
        PpoStats mean = {0};
        for (int k = 0; k < NUM_SHIPS; k++) {
            ppo_update(&tr.agents[k], &tr.rollouts[k], &pc, &tr.update_rng[k], &st[k]);
            mean.pg_loss += st[k].pg_loss / NUM_SHIPS;
            mean.v_loss += st[k].v_loss / NUM_SHIPS;
            mean.entropy += st[k].entropy / NUM_SHIPS;
            mean.approx_kl += st[k].approx_kl / NUM_SHIPS;
            mean.clipfrac += st[k].clipfrac / NUM_SHIPS;
            mean.explained_var += st[k].explained_var / NUM_SHIPS;
            mean.grad_norm += st[k].grad_norm / NUM_SHIPS;
        }
        double elapsed = now_seconds() - start;
        fprintf(metrics, "%lld,%lld,%.3f,%.1f,%.6g,%lld", u, tr.decisions, elapsed,
                elapsed > 0 ? tr.decisions / elapsed : 0.0, pc.lr, tr.episodes);
        write_episode_columns(metrics, &tr);
        fprintf(metrics, ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", mean.pg_loss, mean.v_loss, mean.entropy,
                mean.approx_kl, mean.clipfrac, mean.explained_var, mean.grad_norm);
        fflush(metrics);
        if (!isfinite(mean.pg_loss) || !isfinite(mean.v_loss) || !isfinite(mean.entropy)) {
            fprintf(stderr, "train: non-finite loss at update %lld; stopping\n", u);
            rc_out = -1;
            break;
        }
        if (elapsed >= rc->max_seconds) {
            break;
        }
    }
    trainer_free(&tr);
    if (fclose(metrics) != 0) {
        rc_out = -1;
    }
    return rc_out;
}
