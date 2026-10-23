#include <string.h>

#include "check.h"
#include "env.h"

void test_env_reset_matches_upstream_ships(void) {
    Env env;
    env_reset(&env, 1234u);
    for (int i = 0; i < NUM_SHIPS; i++) {
        const Ship *s = &env.ships[i];
        CHECK_NEAR(s->x, 400.0f + 50.0f * (float)i, 0.0);
        CHECK_NEAR(s->y, 700.0f, 0.0);
        CHECK_NEAR(s->health, 200.0f, 0.0);
        CHECK(s->ammo == 100);
        CHECK(s->alive == 1);
        CHECK(s->weapon_cd == 0);
    }
    CHECK_NEAR(SHIP_SPEED, 400.0f, 0.0);
    CHECK(env.n_enemies == 0);
    CHECK(env.n_missiles == 0);
    CHECK(env.tick == 0);
    CHECK(env.decision == 0);
    CHECK(env.spawn_timer == 90);
    CHECK(env.max_decisions == 3600);
    CHECK(env.action_repeat == 4);
    CHECK(env.terminated == 0 && env.truncated == 0);
}
