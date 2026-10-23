#define _POSIX_C_SOURCE 200809L

#include "train.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifndef PPO_GIT_COMMIT
#define PPO_GIT_COMMIT "unknown"
#endif

#define METRICS_HEADER                                                                                         \
    "update,decisions,seconds,decisions_per_sec,lr,episodes,ep_return_mean,ep_return_ship0,ep_return_ship1,"    \
    "ep_return_ship2,team_score,kills,hits,shots,accuracy,aimed_share,edge_share,damage_taken,deaths,escapes,"  \
    "spawned,ep_seconds,"                                                                                       \
    "pg_loss,v_loss,entropy,approx_kl,clipfrac,explained_var,grad_norm"

RunConfig run_config_default(void) {
    RunConfig rc = {
        .train = train_config_default(),
        .total_decisions = 5000000,
        .max_seconds = 3600.0,
        .checkpoint_every = 50,
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

static int write_manifest(const RunConfig *rc, const char *status, long long updates, long long decisions,
                          double seconds) {
    char path[4096], tmp[4200];
    snprintf(path, sizeof path, "%s/manifest.json", rc->run_dir);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) {
        return -1;
    }
    const TrainConfig *t = &rc->train;
    const RewardConfig *r = &t->reward;
    const PpoConfig *p = &t->ppo;
    fprintf(f,
            "{\n"
            "  \"status\": \"%s\",\n"
            "  \"git_commit\": \"%s\",\n"
            "  \"seed\": %llu,\n"
            "  \"updates\": %lld,\n"
            "  \"decisions\": %lld,\n"
            "  \"seconds\": %.1f,\n"
            "  \"budget\": {\"total_decisions\": %lld, \"max_seconds\": %.1f},\n"
            "  \"env\": {\"num_ships\": %d, \"ship_health\": %g, \"ship_ammo\": %d, \"obs_dim\": %d, "
            "\"tick_hz\": %d, \"action_repeat\": %d, \"episode_ticks\": %d},\n"
            "  \"reward\": {\"kill_points\": %g, \"hit\": %g, \"shot\": %g, \"damage\": %g, \"death\": %g, "
            "\"escape\": %g, \"ally_share\": %g, \"align\": %g, \"edge\": %g},\n"
            "  \"ppo\": {\"envs\": %d, \"steps\": %d, \"hidden\": %d, \"gamma\": %g, \"lambda\": %g, "
            "\"clip\": %g, \"vf_coef\": %g, \"ent_coef\": %g, \"max_grad_norm\": %g, \"lr\": %g, "
            "\"epochs\": %d, \"minibatches\": %d, \"norm_adv\": %d},\n"
            "  \"checkpoint_every\": %d\n"
            "}\n",
            status, PPO_GIT_COMMIT, (unsigned long long)t->seed, updates, decisions, seconds, rc->total_decisions,
            rc->max_seconds, NUM_SHIPS, SHIP_HEALTH, SHIP_AMMO, OBS_DIM, TICK_HZ, ACTION_REPEAT, t->episode_ticks,
            r->kill_points, r->hit, r->shot, r->damage, r->death, r->escape, r->ally_share, r->align, r->edge, t->envs, t->steps,
            t->hidden, p->gamma, p->lambda, p->clip, p->vf_coef, p->ent_coef, p->max_grad_norm, p->lr, p->epochs,
            p->minibatches, p->norm_adv, rc->checkpoint_every);
    if (fclose(f) != 0 || rename(tmp, path) != 0) {
        return -1;
    }
    return 0;
}

static int save_snapshot(Trainer *tr, const char *path, long long update) {
    Mlp *nets[CHECKPOINT_NETS];
    ship_agent_nets(tr->agents, nets);
    int rc = checkpoint_save(path, nets, CHECKPOINT_NETS, (uint32_t)update);
    if (rc != 0) {
        fprintf(stderr, "train: %s: %s\n", path, checkpoint_error(rc));
    }
    return rc;
}

/* Columns ep_return_mean .. ep_seconds in METRICS_HEADER, written by the fprintf below. */
#define EPISODE_COLUMNS 16

static void write_episode_columns(FILE *f, const Trainer *tr) {
    int n = tr->episodes < EPISODE_WINDOW ? (int)tr->episodes : EPISODE_WINDOW;
    if (n == 0) {
        for (int c = 0; c < EPISODE_COLUMNS; c++) {
            fputc(',', f); /* empty until an episode finishes */
        }
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
        sum.aimed += r->aimed;
        sum.edge_ticks += r->edge_ticks;
        sum.alive_ticks += r->alive_ticks;
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
    fprintf(f, ",%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f", mean_ret,
            sum.ret[0] / n, sum.ret[1] / n, sum.ret[2] / n, sum.team_score / n, sum.kills / n, sum.hits / n,
            sum.shots / n, sum.shots > 0 ? sum.hits / sum.shots : 0.0, sum.shots > 0 ? sum.aimed / sum.shots : 0.0,
            sum.alive_ticks > 0 ? sum.edge_ticks / sum.alive_ticks : 0.0, sum.damage / n, sum.deaths / n,
            sum.escapes / n, sum.spawned / n, sum.seconds / n);
}

int train_run(const RunConfig *rc) {
    if (!rc->run_dir || make_dirs(rc->run_dir) != 0) {
        fprintf(stderr, "train: cannot create run directory %s\n", rc->run_dir ? rc->run_dir : "(none)");
        return -1;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/checkpoints", rc->run_dir);
    if (make_dirs(path) != 0 || write_manifest(rc, "running", 0, 0, 0.0) != 0) {
        fprintf(stderr, "train: cannot write into %s\n", rc->run_dir);
        return -1;
    }
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
    const char *status = "completed";
    long long u = 0;

    while (tr.decisions < rc->total_decisions) {
        u++;
        PpoConfig pc = tr.cfg.ppo;
        pc.lr = (float)(rc->train.ppo.lr * (1.0 - (double)(u - 1) / (double)planned));
        trainer_collect(&tr);
        PpoStats st[NUM_SHIPS];
        PpoStats mean = {0};
        trainer_update(&tr, &pc, st, NUM_SHIPS);
        for (int k = 0; k < NUM_SHIPS; k++) {
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
            status = "failed";
            rc_out = -1;
            break;
        }
        if (rc->checkpoint_every > 0 && u % rc->checkpoint_every == 0) {
            snprintf(path, sizeof path, "%s/checkpoints/update_%06lld.bin", rc->run_dir, u);
            if (save_snapshot(&tr, path, u) != 0) {
                rc_out = -1;
            }
        }
        if (elapsed >= rc->max_seconds && tr.decisions < rc->total_decisions) {
            status = "time_limit";
            break;
        }
    }
    if (rc_out == 0) {
        snprintf(path, sizeof path, "%s/final.bin", rc->run_dir);
        if (save_snapshot(&tr, path, u) != 0) {
            rc_out = -1;
        }
    }
    if (write_manifest(rc, rc_out == 0 ? status : "failed", u, tr.decisions, now_seconds() - start) != 0) {
        rc_out = -1;
    }
    trainer_free(&tr);
    if (fclose(metrics) != 0) {
        rc_out = -1;
    }
    return rc_out;
}
