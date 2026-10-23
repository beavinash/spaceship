#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "eval.h"

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
}
