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
