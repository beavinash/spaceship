#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eval.h"
#include "train.h"

static void usage(void) {
    fputs("usage:\n"
          "  ppo train --run-dir DIR [--seed N] [--total-decisions N | --updates N] [--max-seconds S]\n"
          "            [--checkpoint-every N]\n"
          "            [--envs N] [--steps N] [--hidden N] [--episode-seconds S] [--lr X] [--ent-coef X]\n"
          "            [--epochs N] [--minibatches N] [--kill-points X] [--hit X] [--shot X] [--damage X]\n"
          "            [--death X] [--escape X] [--ally-share X]\n"
          "  ppo eval --policy random|ppo|legacy --output FILE.json [--checkpoint PATH] [--sample]\n"
          "           [--game-data DIR (legacy; default ../Game_Data)]\n"
          "           [--split dev|test] [--final]\n",
          stderr);
}

/* Parses "--flag value" pairs; returns 0 or prints an error. */
static int parse_train(int argc, char **argv, RunConfig *rc) {
    long long updates = 0;
    for (int i = 0; i < argc; i += 2) {
        const char *flag = argv[i];
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", flag);
            return -1;
        }
        const char *v = argv[i + 1];
        TrainConfig *t = &rc->train;
        if (!strcmp(flag, "--run-dir")) {
            rc->run_dir = v;
        } else if (!strcmp(flag, "--seed")) {
            t->seed = strtoull(v, NULL, 10);
        } else if (!strcmp(flag, "--total-decisions")) {
            rc->total_decisions = atoll(v);
        } else if (!strcmp(flag, "--updates")) {
            updates = atoll(v);
        } else if (!strcmp(flag, "--max-seconds")) {
            rc->max_seconds = atof(v);
        } else if (!strcmp(flag, "--checkpoint-every")) {
            rc->checkpoint_every = atoi(v);
        } else if (!strcmp(flag, "--envs")) {
            t->envs = atoi(v);
        } else if (!strcmp(flag, "--steps")) {
            t->steps = atoi(v);
        } else if (!strcmp(flag, "--hidden")) {
            t->hidden = atoi(v);
        } else if (!strcmp(flag, "--episode-seconds")) {
            t->episode_ticks = (int)(atof(v) * TICK_HZ);
        } else if (!strcmp(flag, "--lr")) {
            t->ppo.lr = (float)atof(v);
        } else if (!strcmp(flag, "--ent-coef")) {
            t->ppo.ent_coef = (float)atof(v);
        } else if (!strcmp(flag, "--epochs")) {
            t->ppo.epochs = atoi(v);
        } else if (!strcmp(flag, "--minibatches")) {
            t->ppo.minibatches = atoi(v);
        } else if (!strcmp(flag, "--kill-points")) {
            t->reward.kill_points = (float)atof(v);
        } else if (!strcmp(flag, "--hit")) {
            t->reward.hit = (float)atof(v);
        } else if (!strcmp(flag, "--shot")) {
            t->reward.shot = (float)atof(v);
        } else if (!strcmp(flag, "--damage")) {
            t->reward.damage = (float)atof(v);
        } else if (!strcmp(flag, "--death")) {
            t->reward.death = (float)atof(v);
        } else if (!strcmp(flag, "--escape")) {
            t->reward.escape = (float)atof(v);
        } else if (!strcmp(flag, "--ally-share")) {
            t->reward.ally_share = (float)atof(v);
        } else {
            fprintf(stderr, "unknown option %s\n", flag);
            return -1;
        }
    }
    if (updates > 0) {
        rc->total_decisions = updates * rc->train.envs * rc->train.steps;
    }
    if (!rc->run_dir || rc->train.envs <= 0 || rc->train.steps <= 0 || rc->train.hidden <= 0 ||
        rc->train.episode_ticks <= 0 || rc->train.ppo.minibatches <= 0 || rc->total_decisions <= 0) {
        fputs("train needs --run-dir and positive sizes\n", stderr);
        return -1;
    }
    return 0;
}

static int run_eval(int argc, char **argv, const char *binary) {
    const char *policy = NULL, *split = "dev", *output = NULL, *checkpoint = NULL, *game_data = "../Game_Data";
    int final = 0, sample = 0;
    for (int i = 0; i < argc; i++) {
        const char *flag = argv[i];
        if (!strcmp(flag, "--final")) {
            final = 1;
            continue;
        }
        if (!strcmp(flag, "--sample")) {
            sample = 1;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", flag);
            return 2;
        }
        const char *v = argv[++i];
        if (!strcmp(flag, "--policy")) {
            policy = v;
        } else if (!strcmp(flag, "--split")) {
            split = v;
        } else if (!strcmp(flag, "--output")) {
            output = v;
        } else if (!strcmp(flag, "--checkpoint")) {
            checkpoint = v;
        } else if (!strcmp(flag, "--game-data")) {
            game_data = v;
        } else {
            fprintf(stderr, "unknown option %s\n", flag);
            return 2;
        }
    }
    if (!policy || !output) {
        fputs("eval needs --policy and --output\n", stderr);
        return 2;
    }
    Policy p;
    if (!strcmp(policy, "random")) {
        policy_random(&p);
    } else if (!strcmp(policy, "legacy")) {
        if (policy_legacy(&p, game_data) != 0) {
            fprintf(stderr, "eval: cannot load legacy tables from %s\n", game_data);
            return 1;
        }
    } else if (!strcmp(policy, "ppo")) {
        int rc = checkpoint ? policy_ppo(&p, checkpoint, sample) : CKPT_IO;
        if (rc != 0) {
            fprintf(stderr, "eval: %s: %s\n", checkpoint ? checkpoint : "--checkpoint missing", checkpoint_error(rc));
            return 1;
        }
    } else {
        fprintf(stderr, "unknown policy %s\n", policy);
        return 2;
    }
    static EpisodeResult res[100];
    int n = 0;
    if (eval_run(&p, split, final, res, &n) != 0) {
        policy_free(&p);
        return 1;
    }
    ReportMeta meta = {.policy = p.name, .split = split, .checkpoint = checkpoint, .binary = binary};
    int written = eval_write_json(output, &meta, &p, res, n);
    const char *name = p.name; /* static string; policy_free clears the struct */
    policy_free(&p);
    if (written != 0) {
        fprintf(stderr, "eval: cannot write %s\n", output);
        return 1;
    }
    Summary score = summarize(res, n, offsetof(EpisodeResult, team_score));
    Summary deaths = summarize(res, n, offsetof(EpisodeResult, deaths));
    printf("%s on %s (%d scenarios): team score %.2f +/- %.2f, deaths %.2f +/- %.2f -> %s\n", name, split, n,
           score.mean, score.ci95, deaths.mean, deaths.ci95, output);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "eval")) {
        int rc = run_eval(argc - 2, argv + 2, argv[0]);
        if (rc == 2) {
            usage();
        }
        return rc;
    }
    if (argc >= 2 && !strcmp(argv[1], "train")) {
        RunConfig rc = run_config_default();
        if (parse_train(argc - 2, argv + 2, &rc) != 0) {
            usage();
            return 2;
        }
        return train_run(&rc) == 0 ? 0 : 1;
    }
    usage();
    return 2;
}
