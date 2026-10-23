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

/* An arena with spawning pushed far into the future, so rules can be tested in isolation. */
static void quiet_env(Env *env) {
    env_reset(env, 1u);
    env->spawn_timer = 1 << 30;
}

void test_env_movement_matches_upstream_control(void) {
    Env env;
    quiet_env(&env);
    ShipAction a[NUM_SHIPS] = {{H_LEFT, V_STAY, 0}, {H_RIGHT, V_UP, 0}, {H_STAY, V_STAY, 0}};
    env_tick(&env, a);
    float step = 400.0f / 120.0f;
    float diag = step / sqrtf(2.0f);
    CHECK_NEAR(env.ships[0].x, 400.0f - step, 1e-4);
    CHECK_NEAR(env.ships[0].y, 700.0f, 1e-4);
    CHECK_NEAR(env.ships[1].x, 450.0f + diag, 1e-4);
    CHECK_NEAR(env.ships[1].y, 700.0f - diag, 1e-4);
    CHECK_NEAR(env.ships[2].x, 500.0f, 0.0);
    CHECK_NEAR(env.ships[2].y, 700.0f, 0.0);
    CHECK(env.tick == 1);

    quiet_env(&env);
    env.ships[0].x = 1.0f;
    env.ships[0].y = 949.0f;
    a[0] = (ShipAction){H_LEFT, V_DOWN, 0};
    env_tick(&env, a);
    CHECK_NEAR(env.ships[0].x, 0.0f, 0.0);
    CHECK_NEAR(env.ships[0].y, 950.0f, 0.0);
}

static void hold(Env *env, ShipAction a0, ShipAction a1, ShipAction a2, int ticks) {
    ShipAction a[NUM_SHIPS] = {a0, a1, a2};
    for (int t = 0; t < ticks; t++) {
        env_tick(env, a);
    }
}

static const ShipAction IDLE = {H_STAY, V_STAY, 0};
static const ShipAction FIRE = {H_STAY, V_STAY, 1};

void test_env_weapon_cooldown_is_30_ticks(void) {
    Env env;
    quiet_env(&env);
    hold(&env, FIRE, IDLE, IDLE, 90);
    CHECK(env.n_missiles == 3);
    CHECK(env.stats.shots[0] == 3);
    CHECK(env.stats.shots[1] == 0);
    for (int m = 0; m < env.n_missiles; m++) {
        CHECK(env.missiles[m].owner == 0);
        CHECK_NEAR(env.missiles[m].x, 400.0f, 0.0);
    }
    CHECK(env.ships[0].weapon_cd == 1);

    quiet_env(&env);
    hold(&env, FIRE, IDLE, IDLE, 1);
    CHECK(env.n_missiles == 1);
    /* Launched at y = 700 with 30 px/s, then moved once in the same tick (upstream frame order). */
    CHECK_NEAR(env.missiles[0].vy, 30.0f + 300.0f / 120.0f, 1e-4);
    CHECK_NEAR(env.missiles[0].y, 700.0f - (30.0f + 300.0f / 120.0f) / 120.0f, 1e-4);
    hold(&env, FIRE, IDLE, IDLE, 29);
    CHECK(env.n_missiles == 1);
    hold(&env, FIRE, IDLE, IDLE, 1);
    CHECK(env.n_missiles == 2);
}

void test_env_ammo_is_100_without_refill(void) {
    Env env;
    quiet_env(&env);
    ShipAction a[NUM_SHIPS] = {FIRE, IDLE, IDLE};
    for (int t = 0; t < 3300; t++) {
        env_tick(&env, a);
        env.n_missiles = 0; /* isolate the ammo rule from missile flight */
    }
    CHECK(env.stats.shots[0] == 100);
    CHECK(env.ships[0].ammo == 0);
    CHECK(env.ships[1].ammo == 100);
    CHECK(env.ships[0].weapon_cd <= 0);
}

void test_env_missile_flight_matches_closed_form(void) {
    Env env;
    quiet_env(&env);
    hold(&env, FIRE, IDLE, IDLE, 1);
    const double dt = 1.0 / 120.0;
    int n = 1;
    int matched = 1;
    double y = 700.0 - dt * (30.0 * n + 300.0 * dt * n * (n + 1) / 2.0);
    while (y >= -10.0) {
        matched &= env.n_missiles == 1 && fabs(env.missiles[0].y - y) < 1e-2;
        hold(&env, IDLE, IDLE, IDLE, 1);
        n++;
        y = 700.0 - dt * (30.0 * n + 300.0 * dt * n * (n + 1) / 2.0);
    }
    CHECK(matched);
    CHECK(n > 100 && n < 400);
    /* The tick that carries it above y = -10 leaves it in flight; the next tick's hit pass removes it. */
    CHECK(env.n_missiles == 1);
    CHECK_NEAR(env.missiles[0].y, y, 1e-2);
    CHECK_NEAR(env.missiles[0].vy, 30.0 + 300.0 * dt * n, 1e-2);
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK(env.n_missiles == 0);
}

void test_env_spawns_seeded_enemies_every_90_ticks(void) {
    Env env;
    env_reset(&env, 77u);
    hold(&env, IDLE, IDLE, IDLE, 89);
    CHECK(env.stats.spawned == 0);
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK(env.stats.spawned == 1);
    hold(&env, IDLE, IDLE, IDLE, 89);
    CHECK(env.stats.spawned == 1);
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK(env.stats.spawned == 2);

    env_reset(&env, 78u);
    int type_counts[3] = {0, 0, 0};
    int stats_ok = 1;
    for (int k = 0; k < 50; k++) {
        hold(&env, IDLE, IDLE, IDLE, 90);
        const Enemy *e = &env.enemies[env.n_enemies - 1];
        const EnemyType *t = &ENEMY_TYPES[e->type];
        type_counts[e->type]++;
        stats_ok &= e->health == t->health && e->damage == t->damage && e->accel == t->accel;
        stats_ok &= e->points == t->points && e->attack_cd_ticks == t->attack_cooldown_ticks;
        stats_ok &= e->radius == 25.0f && e->dead == 0 && e->attack_cd_left <= 0;
        stats_ok &= e->x >= 0.0f && e->x <= 802.0f && e->y >= -25.0f && e->y <= -24.0f;
        stats_ok &= e->id == (uint32_t)(k + 1);
    }
    CHECK(stats_ok);
    CHECK(env.stats.spawned == 50);
    for (int k = 0; k < 3; k++) {
        CHECK(type_counts[k] >= 5 && type_counts[k] <= 30);
    }
    CHECK(ENEMY_TYPES[0].points == 2 && ENEMY_TYPES[1].points == 4 && ENEMY_TYPES[2].points == 8);
    CHECK(ENEMY_TYPES[2].attack_cooldown_ticks == 168);

    Env a, b;
    env_reset(&a, 5u);
    env_reset(&b, 5u);
    hold(&a, IDLE, IDLE, IDLE, 900);
    hold(&b, IDLE, IDLE, IDLE, 900);
    int same = a.n_enemies == b.n_enemies && a.n_enemies == 10;
    for (int k = 0; same && k < a.n_enemies; k++) {
        same &= a.enemies[k].x == b.enemies[k].x && a.enemies[k].type == b.enemies[k].type;
    }
    CHECK(same);
}
