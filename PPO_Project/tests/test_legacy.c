#include "check.h"
#include "legacy.h"

void test_legacy_tables_reproduce_upstream_lookups(void) {
    if (!legacy_available()) {
        printf("  legacy: built without cJSON; skipped\n");
        return;
    }
    /* Upstream Normalize: displacement / 50 + 19, clamped to [0, 39], truncated. */
    int x, y;
    legacy_cell(0.0f, 0.0f, &x, &y);
    CHECK(x == 19 && y == 19);
    legacy_cell(-49.0f, 49.0f, &x, &y);
    CHECK(x == 18 && y == 19);
    legacy_cell(300.0f, -2000.0f, &x, &y);
    CHECK(x == 25 && y == 0);
    legacy_cell(5000.0f, 5000.0f, &x, &y);
    CHECK(x == 39 && y == 39);

    LegacyTables *t0 = legacy_load("../Game_Data/trained0.json");
    LegacyTables *t2 = legacy_load("../Game_Data/trained2.json");
    CHECK(t0 != NULL && t2 != NULL);
    if (!t0 || !t2) {
        legacy_free(t0);
        legacy_free(t2);
        return;
    }
    /* Reference values read independently from the JSON with Python's json module. */
    CHECK_NEAR(t0->shoot[19][19], 0.0014226387720555067, 1e-8);
    CHECK_NEAR(t0->move_h[19][19][2], 0.9954113364219666, 1e-7);
    CHECK_NEAR(t0->move_v[19][19][1], 0.9958080649375916, 1e-7);
    CHECK_NEAR(t0->move_h[19][25][1], 0.027080457657575607, 1e-8);
    CHECK_NEAR(t0->move_v[10][30][2], 0.006114667281508446, 1e-8);
    CHECK_NEAR(t2->shoot[19][19], 0.6256151795387268, 1e-7);
    CHECK_NEAR(t2->move_h[10][30][2], 0.6971445083618164, 1e-7);
    CHECK(legacy_load("../Game_Data/no-such-file.json") == NULL);

    /* Upstream Compute_Action with fixed draws in 1..100: thresholds are (int)(100 * p). */
    const float h[3] = {0.2f, 0.5f, 0.3f};
    const float v[3] = {0.1f, 0.1f, 0.8f};
    ShipAction a = legacy_compute_action(h, v, 0.4f, 10, 20, 40);
    CHECK(a.h == H_LEFT && a.v == V_UP && a.fire == 1);
    a = legacy_compute_action(h, v, 0.4f, 11, 21, 41);
    CHECK(a.h == H_STAY && a.v == V_STAY && a.fire == 0);
    a = legacy_compute_action(h, v, 0.4f, 20, 70, 1);
    CHECK(a.h == H_STAY && a.v == V_STAY && a.fire == 1);
    a = legacy_compute_action(h, v, 0.4f, 21, 71, 100);
    CHECK(a.h == H_RIGHT && a.v == V_DOWN && a.fire == 0);
    legacy_free(t0);
    legacy_free(t2);
}

#include <stdlib.h>

#include "eval.h"

static LegacyTables *uniform_tables(void) {
    LegacyTables *t = calloc(1, sizeof *t);
    for (int x = 0; x < LEGACY_GRID; x++) {
        for (int y = 0; y < LEGACY_GRID; y++) {
            t->shoot[x][y] = 0.5f;
            for (int k = 0; k < 3; k++) {
                t->move_h[x][y][k] = 1.0f / 3.0f;
                t->move_v[x][y][k] = 1.0f / 3.0f;
            }
        }
    }
    return t;
}

void test_legacy_target_locking_matches_upstream(void) {
    LegacyTables *t[NUM_SHIPS] = {uniform_tables(), uniform_tables(), uniform_tables()};
    Env env;
    env_reset(&env, 1u);
    env.spawn_timer = 1 << 30;
    env_add_enemy(&env, 0, 400.0f, 600.0f); /* E0: 2 points, 100 px above ship 0 */
    env_add_enemy(&env, 2, 400.0f, 350.0f); /* E1: 8 points, 350 px above ship 0 */
    Enemy *x = env_add_enemy(&env, 2, 450.0f, 690.0f);
    x->dead = 1; /* explosions are never targets */
    int targets[NUM_SHIPS];

    /* Equal shoot odds: ship 0 maximises points / distance (8/350 > 2/100); ship 1 skips E1 (taken by
     * a lower-index ship) and takes E0; ship 2 finds nothing free and borrows ship 0's target. */
    legacy_targets(&env, t, targets);
    CHECK(targets[0] == 1 && targets[1] == 0 && targets[2] == 1);

    /* Learned shoot odds reweight the choice: E0's cell certain, E1's cell unlikely. */
    t[0]->shoot[19][21] = 1.0f; /* ship 0 - E0 = (0, 100) */
    t[0]->shoot[19][26] = 0.1f; /* ship 0 - E1 = (0, 350) */
    legacy_targets(&env, t, targets);
    CHECK(targets[0] == 0 && targets[1] == 1 && targets[2] == 0);

    /* A dead ship locks nothing, so ship 2 can take what ship 1 would have. */
    env.ships[1].alive = 0;
    legacy_targets(&env, t, targets);
    CHECK(targets[0] == 0 && targets[1] == -1 && targets[2] == 1);

    /* No live enemies: nobody has a target and the ally fallback finds none. */
    env.n_enemies = 0;
    env.ships[1].alive = 1;
    legacy_targets(&env, t, targets);
    CHECK(targets[0] == -1 && targets[1] == -1 && targets[2] == -1);

    /* The policy reads its table at the target's cell: here certain left, down, and fire. */
    env_add_enemy(&env, 2, 400.0f, 350.0f);
    t[0]->shoot[19][26] = 1.0f;
    t[0]->move_h[19][26][0] = 1.0f;
    t[0]->move_h[19][26][1] = 0.0f;
    t[0]->move_h[19][26][2] = 0.0f;
    t[0]->move_v[19][26][0] = 0.0f;
    t[0]->move_v[19][26][1] = 0.0f;
    t[0]->move_v[19][26][2] = 1.0f;
    Policy p;
    policy_legacy_tables(&p, (void *const *)t);
    CHECK(p.kind == POLICY_LEGACY && p.action_repeat == 1);
    Pcg32 rng;
    pcg32_seed(&rng, 5u, 5u);
    ShipAction a[NUM_SHIPS];
    policy_act(&p, &env, &rng, a);
    CHECK(a[0].h == H_LEFT && a[0].v == V_DOWN && a[0].fire == 1);
    policy_free(&p);
    for (int k = 0; k < NUM_SHIPS; k++) {
        free(t[k]);
    }
}
