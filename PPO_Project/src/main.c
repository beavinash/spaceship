#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eval.h"
#include "selection.h"
#include "train.h"

static void usage(void) {
    fputs("usage:\n"
          "  ppo train --run-dir DIR [--seed N] [--total-decisions N | --updates N] [--max-seconds S]\n"
          "            [--checkpoint-every N]\n"
          "            [--envs N] [--steps N] [--hidden N] [--episode-seconds S] [--lr X] [--gamma X] [--lambda X]\n"
          "            [--ent-coef X]\n"
          "            [--epochs N] [--minibatches N] [--kill-points X] [--hit X] [--shot X] [--damage X]\n"
          "            [--death X] [--escape X] [--ally-share X] [--align X] [--edge X]\n"
          "  ppo eval --policy random|ppo|legacy --output FILE.json [--checkpoint PATH] [--sample]\n"
          "           [--game-data DIR (legacy; default ../Game_Data)]\n"
          "           [--split dev|test] [--final]\n"
          "  ppo select --run-dir DIR   (scores DIR's checkpoints on dev; writes selection.json, selected.bin)\n"
          "  ppo stress [--decisions N] [--seed N]   (random play with invariant checks after every decision)\n",
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
        } else if (!strcmp(flag, "--gamma")) {
            t->ppo.gamma = (float)atof(v);
        } else if (!strcmp(flag, "--lambda")) {
            t->ppo.lambda = (float)atof(v);
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
        } else if (!strcmp(flag, "--align")) {
            t->reward.align = (float)atof(v);
        } else if (!strcmp(flag, "--edge")) {
            t->reward.edge = (float)atof(v);
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

static int run_select(int argc, char **argv) {
    if (argc != 2 || strcmp(argv[0], "--run-dir") != 0) {
        return 2;
    }
    static Candidate candidates[1024];
    int n = 0;
    int best = select_run(argv[1], candidates, 1024, &n);
    if (best < 0) {
        return 1;
    }
    printf("selected %s (update %u): dev team score %.2f, deaths %.2f, from %d checkpoints\n", candidates[best].path,
           candidates[best].update, candidates[best].mean_score, candidates[best].mean_deaths, n);
    return 0;
}

static int run_stress(int argc, char **argv) {
    long long decisions = 1000000;
    uint64_t seed = 1;
    for (int i = 0; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--decisions")) {
            decisions = atoll(argv[i + 1]);
        } else if (!strcmp(argv[i], "--seed")) {
            seed = strtoull(argv[i + 1], NULL, 10);
        } else {
            return 2;
        }
    }
    Env env;
    Pcg32 policy;
    pcg32_seed(&policy, seed, 0x57e5u);
    env_reset(&env, seed);
    long long episodes = 0, max_enemies = 0, max_missiles = 0;
    char why[256];
    for (long long d = 0; d < decisions; d++) {
        ShipAction a[NUM_SHIPS];
        for (int k = 0; k < NUM_SHIPS; k++) {
            a[k] = (ShipAction){(int)pcg32_bounded(&policy, 3), (int)pcg32_bounded(&policy, 3),
                                (int)pcg32_bounded(&policy, 2)};
        }
        env_step(&env, a);
        max_enemies = env.n_enemies > max_enemies ? env.n_enemies : max_enemies;
        max_missiles = env.n_missiles > max_missiles ? env.n_missiles : max_missiles;
        if (env_check_invariants(&env, why, sizeof why) != 0) {
            fprintf(stderr, "stress: violation after decision %lld (episode seed %llu): %s\n", d,
                    (unsigned long long)env.seed, why);
            return 1;
        }
        if (env.terminated || env.truncated) {
            episodes++;
            env_reset(&env, seed + (uint64_t)episodes * 1000003u);
        }
    }
    printf("stress: %lld decisions, %lld episodes, peak %lld/%d enemies, %lld/%d missiles, 0 violations\n", decisions,
           episodes, max_enemies, MAX_ENEMIES, max_missiles, MAX_MISSILES);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "stress")) {
        int rc = run_stress(argc - 2, argv + 2);
        if (rc == 2) {
            usage();
        }
        return rc;
    }
    if (argc >= 2 && !strcmp(argv[1], "select")) {
        int rc = run_select(argc - 2, argv + 2);
        if (rc == 2) {
            usage();
        }
        return rc;
    }
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
