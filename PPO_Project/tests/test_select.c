#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "selection.h"
#include "train.h"

void test_select_best_breaks_ties_deterministically(void) {
    const Candidate c[] = {
        {"a.bin", 50, 10.0, 1.0},
        {"b.bin", 100, 12.0, 2.0},
        {"c.bin", 150, 12.0, 1.0},
        {"d.bin", 120, 12.0, 1.0},
        {"e.bin", 120, 12.0, 1.0},
    };
    CHECK(select_best(c, 5) == 3);  /* top score, fewer deaths, earlier update, then listing order */
    CHECK(select_best(c, 3) == 2);
    CHECK(select_best(c, 1) == 0);
    CHECK(select_best(c, 0) == -1);

    /* End to end over a tiny run's checkpoints: every candidate is scored on dev and the winner copied. */
    RunConfig rc = run_config_default();
    rc.train.envs = 2;
    rc.train.steps = 8;
    rc.train.hidden = 16;
    rc.train.seed = 4u;
    rc.total_decisions = 3 * 2 * 8;
    rc.checkpoint_every = 1;
    rc.run_dir = "build/test-run-select";
    remove("build/test-run-select/selected.bin");
    CHECK(train_run(&rc) == 0);
    Candidate out[16];
    int n = 0;
    int best = select_run("build/test-run-select", out, 16, &n);
    CHECK(n == 3); /* update 1, 2, 3; final.bin repeats update 3 and is not scored twice */
    CHECK(best >= 0 && best < n);
    for (int i = 0; i < n; i++) {
        CHECK(out[i].update == (uint32_t)(i + 1));
    }
    FILE *a = fopen("build/test-run-select/selected.bin", "rb");
    FILE *b = best >= 0 ? fopen(out[best].path, "rb") : NULL;
    CHECK(a != NULL && b != NULL);
    if (a && b) {
        int same = 1, ca, cb;
        do {
            ca = fgetc(a);
            cb = fgetc(b);
            same &= ca == cb;
        } while (ca != EOF && cb != EOF);
        CHECK(same);
    }
    if (a) {
        fclose(a);
    }
    if (b) {
        fclose(b);
    }
    FILE *j = fopen("build/test-run-select/selection.json", "r");
    CHECK(j != NULL);
    if (j) {
        char buf[8192];
        size_t got = fread(buf, 1, sizeof buf - 1, j);
        buf[got] = '\0';
        fclose(j);
        CHECK(strstr(buf, "\"split\": \"dev\"") != NULL);
        CHECK(strstr(buf, "\"selected\": {\"path\": ") != NULL);
        CHECK(strstr(buf, "\"candidates\": [") != NULL);
    }
}

void test_select_finds_latest_checkpoint(void) {
    char path[512];
    uint32_t update = 0;
    CHECK(latest_checkpoint("build/no-such-run", path, sizeof path, &update) != 0);
    /* build/test-run-select holds update_000001..3 plus final.bin (update 3) from the test above. */
    CHECK(latest_checkpoint("build/test-run-select", path, sizeof path, &update) == 0);
    CHECK(update == 3);
    CHECK(strstr(path, "update_000003.bin") != NULL || strstr(path, "final.bin") != NULL);
    RunConfig rc = run_config_default();
    rc.train.envs = 2;
    rc.train.steps = 8;
    rc.train.hidden = 16;
    rc.total_decisions = 5 * 2 * 8;
    rc.checkpoint_every = 2; /* snapshots at 2 and 4; final.bin at 5 is the newest */
    rc.run_dir = "build/test-run-latest";
    remove("build/test-run-latest/checkpoints/update_000002.bin");
    remove("build/test-run-latest/checkpoints/update_000004.bin");
    CHECK(train_run(&rc) == 0);
    CHECK(latest_checkpoint("build/test-run-latest", path, sizeof path, &update) == 0);
    CHECK(update == 5 && strstr(path, "final.bin") != NULL);
}
