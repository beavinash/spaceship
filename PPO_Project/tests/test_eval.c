#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "eval.h"
#include "trainer.h"

static int file_has(const char *path, const char *needle) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    static char buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

void test_eval_random_policy_dev_report(void) {
    int first = 0, count = 0;
    CHECK(eval_split("dev", &first, &count) == 0 && first == 1000 && count == 20);
    CHECK(eval_split("test", &first, &count) == 0 && first == 10000 && count == 100);
    CHECK(eval_split("train", &first, &count) != 0);

    Policy p;
    policy_random(&p);
    EpisodeResult res[100], again[100];
    int n = 0;
    CHECK(eval_run(&p, "test", 0, res, &n) != 0); /* held-out scenarios need an explicit final run */
    CHECK(eval_run(&p, "dev", 0, res, &n) == 0);
    CHECK(n == 20);
    CHECK(res[0].seed == 1000 && res[19].seed == 1019);
    int m = 0;
    CHECK(eval_run(&p, "dev", 0, again, &m) == 0);
    int same = m == n;
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        same &= res[i].team_score == again[i].team_score && res[i].escapes == again[i].escapes;
        mean += res[i].team_score;
        CHECK(res[i].episode_seconds > 0.0 && res[i].episode_seconds <= 120.0);
        CHECK(res[i].spawned >= res[i].kills);
    }
    mean /= n;
    CHECK(same);

    Summary s = summarize(res, n, offsetof(EpisodeResult, team_score));
    CHECK_NEAR(s.mean, mean, 1e-9);
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        sq += (res[i].team_score - mean) * (res[i].team_score - mean);
    }
    CHECK_NEAR(s.std, sqrt(sq / (n - 1)), 1e-9);
    CHECK_NEAR(s.ci95, 1.96 * s.std / sqrt((double)n), 1e-9);

    const char *path = "build/test-eval-random.json";
    remove(path);
    ReportMeta meta = {.policy = "random", .split = "dev", .checkpoint = NULL, .binary = NULL};
    CHECK(eval_write_json(path, &meta, &p, res, n) == 0);
    CHECK(file_has(path, "\"policy\": \"random\""));
    CHECK(file_has(path, "\"split\": \"dev\""));
    CHECK(file_has(path, "\"episodes\": 20"));
    CHECK(file_has(path, "\"per_seed\": ["));
    CHECK(file_has(path, "\"seed\": 1019"));
    CHECK(file_has(path, "\"team_score\": {\"mean\": "));
    CHECK(file_has(path, "\"ci95\": "));
    CHECK(file_has(path, "\"deaths\": {\"mean\": "));
    CHECK(file_has(path, "\"escapes\": {\"mean\": "));
    CHECK(file_has(path, "\"ship_health\": 200"));
    CHECK(file_has(path, "\"edge_share\": {\"mean\": "));
    CHECK(file_has(path, "\"aimed_shot_share\": {\"mean\": "));
    int shares_ok = 1;
    for (int i = 0; i < n; i++) {
        shares_ok &= res[i].edge_share >= 0.0 && res[i].edge_share <= 1.0;
        shares_ok &= res[i].aimed_shot_share >= 0.0 && res[i].aimed_shot_share <= 1.0;
    }
    CHECK(shares_ok);
}

/* Keys from the "summary" section onward, in order, separated by '|'. */
static void summary_keys(const char *path, char *out, size_t cap) {
    FILE *f = fopen(path, "r");
    out[0] = '\0';
    if (!f) {
        return;
    }
    static char buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = '\0';
    const char *p = strstr(buf, "\"summary\"");
    size_t used = 0;
    while (p && (p = strchr(p, '"')) != NULL) {
        const char *end = strchr(p + 1, '"');
        if (!end) {
            break;
        }
        if (end[1] == ':' && used + (size_t)(end - p) + 2 < cap) {
            memcpy(out + used, p + 1, (size_t)(end - p - 1));
            used += (size_t)(end - p - 1);
            out[used++] = '|';
            out[used] = '\0';
        }
        p = end + 1;
    }
}

void test_eval_ppo_checkpoint_report_matches_schema(void) {
    Pcg32 rng;
    pcg32_seed(&rng, 8u, 8u);
    Agent agents[NUM_SHIPS];
    for (int k = 0; k < NUM_SHIPS; k++) {
        CHECK(agent_init(&agents[k], OBS_DIM, 24, &rng) == 0);
    }
    Mlp *nets[CHECKPOINT_NETS];
    ship_agent_nets(agents, nets);
    const char *ckpt = "build/test-eval-ckpt.bin";
    CHECK(checkpoint_save(ckpt, nets, CHECKPOINT_NETS, 12u) == 0);

    Policy p;
    CHECK(policy_ppo(&p, ckpt, 0) == 0);
    CHECK(p.kind == POLICY_PPO && p.action_repeat == 4);
    policy_free(&p);
    CHECK(policy_ppo(&p, "build/missing.bin", 0) != 0);
    CHECK(policy_ppo(&p, ckpt, 0) == 0);
    static EpisodeResult res[100], again[100];
    int n = 0, m = 0;
    CHECK(eval_run(&p, "dev", 0, res, &n) == 0 && n == 20);
    CHECK(eval_run(&p, "dev", 0, again, &m) == 0);
    int same = 1;
    for (int i = 0; i < n; i++) {
        same &= res[i].team_score == again[i].team_score && res[i].survival_seconds == again[i].survival_seconds;
    }
    CHECK(same); /* greedy evaluation is reproducible */

    ReportMeta meta = {.policy = "ppo", .split = "dev", .checkpoint = ckpt, .binary = NULL};
    CHECK(eval_write_json("build/test-eval-ppo.json", &meta, &p, res, n) == 0);
    CHECK(file_has("build/test-eval-ppo.json", "\"checkpoint\": {\"path\": \"build/test-eval-ckpt.bin\", \"fnv1a\": \""));
    CHECK(file_has("build/test-eval-ppo.json", "\"policy\": \"ppo\""));
    char keys_ppo[8192], keys_random[8192];
    summary_keys("build/test-eval-ppo.json", keys_ppo, sizeof keys_ppo);
    summary_keys("build/test-eval-random.json", keys_random, sizeof keys_random);
    CHECK(strlen(keys_ppo) > 100);
    CHECK(strcmp(keys_ppo, keys_random) == 0);
    policy_free(&p);
    for (int k = 0; k < NUM_SHIPS; k++) {
        agent_free(&agents[k]);
    }
}
