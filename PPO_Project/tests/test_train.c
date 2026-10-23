#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "train.h"

#define DOCUMENTED_COLUMNS                                                                                      \
    "update,decisions,seconds,decisions_per_sec,lr,episodes,ep_return_mean,ep_return_ship0,ep_return_ship1,"    \
    "ep_return_ship2,team_score,kills,hits,shots,accuracy,damage_taken,deaths,escapes,spawned,ep_seconds,"      \
    "pg_loss,v_loss,entropy,approx_kl,clipfrac,explained_var,grad_norm"

static int count_fields(const char *line) {
    int n = 1;
    for (const char *p = line; *p && *p != '\n'; p++) {
        n += *p == ',';
    }
    return n;
}

void test_train_run_writes_metrics_csv(void) {
    RunConfig rc = run_config_default();
    rc.train.envs = 2;
    rc.train.steps = 16;
    rc.train.hidden = 16;
    rc.train.seed = 3u;
    rc.train.episode_ticks = 32; /* 8 decisions, so episodes finish inside the smoke run */
    rc.total_decisions = 2 * 2 * 16;
    rc.max_seconds = 60.0;
    rc.run_dir = "build/test-run";
    remove("build/test-run/metrics.csv");
    CHECK(train_run(&rc) == 0);

    FILE *f = fopen("build/test-run/metrics.csv", "r");
    CHECK(f != NULL);
    if (!f) {
        return;
    }
    char line[4096];
    if (!fgets(line, sizeof line, f)) {
        CHECK(!"metrics.csv has no header");
        fclose(f);
        return;
    }
    line[strcspn(line, "\n")] = '\0';
    CHECK(strcmp(line, DOCUMENTED_COLUMNS) == 0);
    int columns = count_fields(DOCUMENTED_COLUMNS);
    int rows = 0;
    double lr[2] = {0, 0};
    long long decisions[2] = {0, 0};
    double ep_return = 0.0, entropy = 0.0;
    while (fgets(line, sizeof line, f)) {
        CHECK(count_fields(line) == columns);
        int update;
        double seconds, sps;
        if (rows < 2 && sscanf(line, "%d,%lld,%lf,%lf,%lf", &update, &decisions[rows], &seconds, &sps, &lr[rows]) == 5) {
            CHECK(update == rows + 1);
        }
        char *field = line;
        for (int c = 0; c < 6; c++) {
            field = strchr(field, ',') + 1;
        }
        ep_return = strtod(field, NULL);
        char *ent = line;
        for (int c = 0; c < 22; c++) {
            ent = strchr(ent, ',') + 1;
        }
        entropy = strtod(ent, NULL);
        rows++;
    }
    fclose(f);
    CHECK(rows == 2);
    CHECK(decisions[0] == 32 && decisions[1] == 64);
    CHECK_NEAR(lr[0], 3e-4, 1e-9);
    CHECK_NEAR(lr[1], 1.5e-4, 1e-9); /* linear annealing over the planned 2 updates */
    CHECK(isfinite(ep_return) && ep_return != 0.0);
    CHECK(entropy > 0.0);
}

static int file_contains(const char *path, const char *needle) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    static char buf[65536];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

void test_train_run_writes_manifest_and_checkpoints(void) {
    RunConfig rc = run_config_default();
    rc.train.envs = 2;
    rc.train.steps = 8;
    rc.train.hidden = 16;
    rc.train.seed = 3u;
    rc.total_decisions = 2 * 2 * 8;
    rc.checkpoint_every = 1;
    rc.run_dir = "build/test-run-ckpt";
    remove("build/test-run-ckpt/checkpoints/update_000001.bin");
    remove("build/test-run-ckpt/checkpoints/update_000002.bin");
    remove("build/test-run-ckpt/final.bin");
    CHECK(train_run(&rc) == 0);

    const char *manifest = "build/test-run-ckpt/manifest.json";
    CHECK(file_contains(manifest, "\"status\": \"completed\""));
    CHECK(file_contains(manifest, "\"seed\": 3"));
    CHECK(file_contains(manifest, "\"updates\": 2"));
    CHECK(file_contains(manifest, "\"decisions\": 32"));
    CHECK(file_contains(manifest, "\"git_commit\": \""));
    CHECK(file_contains(manifest, "\"ship_health\": 200"));
    CHECK(file_contains(manifest, "\"ship_ammo\": 100"));
    CHECK(file_contains(manifest, "\"shot\": -0.01"));

    TrainConfig tc = rc.train;
    tc.seed = 99u; /* different initial weights, so a successful load is visible */
    Trainer fresh;
    CHECK(trainer_init(&fresh, &tc) == 0);
    Mlp *nets[CHECKPOINT_NETS];
    ship_agent_nets(fresh.agents, nets);
    uint32_t update = 0;
    CHECK(checkpoint_load("build/test-run-ckpt/checkpoints/update_000001.bin", nets, CHECKPOINT_NETS, &update) == 0);
    CHECK(update == 1);
    CHECK(checkpoint_load("build/test-run-ckpt/checkpoints/update_000002.bin", nets, CHECKPOINT_NETS, &update) == 0);
    CHECK(update == 2);
    float probe = fresh.agents[2].critic.params[5];
    CHECK(checkpoint_load("build/test-run-ckpt/final.bin", nets, CHECKPOINT_NETS, &update) == 0);
    CHECK(update == 2);
    CHECK(fresh.agents[2].critic.params[5] == probe); /* final equals the last periodic snapshot */
    trainer_free(&fresh);
}
