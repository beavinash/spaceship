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
    CHECK(env.max_ticks == 14400); /* 120 s at 120 ticks/s = 3600 decisions of 4 ticks */
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
    int same = a.n_enemies == b.n_enemies && a.stats.spawned == 10 && b.stats.spawned == 10;
    for (int k = 0; same && k < a.n_enemies; k++) {
        same &= a.enemies[k].x == b.enemies[k].x && a.enemies[k].type == b.enemies[k].type;
    }
    CHECK(same);
}

void test_env_enemy_zigzag_reverses_in_window(void) {
    Env env;
    quiet_env(&env);
    Enemy *e = env_add_enemy(&env, 0, 340.0f, 0.0f);
    CHECK(e != NULL && env.n_enemies == 1);
    hold(&env, IDLE, IDLE, IDLE, 7);
    CHECK_NEAR(env.enemies[0].vx, 200.0f, 0.0);
    CHECK_NEAR(env.enemies[0].x, 340.0f + 7.0f * 200.0f / 120.0f, 1e-3);
    /* Tick 8 sees floor(x) % 350 == 1, flips vx, then moves left. */
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK_NEAR(env.enemies[0].vx, -200.0f, 0.0);
    CHECK_NEAR(env.enemies[0].x, 350.0f, 1e-3);
    CHECK_NEAR(env.enemies[0].vy, 16.0f, 1e-4);
    CHECK_NEAR(env.enemies[0].y, 107.0f / 120.0f, 1e-4);
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK_NEAR(env.enemies[0].vx, -200.0f, 0.0);

    quiet_env(&env);
    env_add_enemy(&env, 1, 949.0f, 0.0f);
    env_add_enemy(&env, 2, 1.0f, 0.0f);
    env.enemies[1].vx = -200.0f;
    hold(&env, IDLE, IDLE, IDLE, 10);
    /* Upstream quirk kept: 950 % 350 == 250 is outside the window, so an enemy clamped there never reverses. */
    CHECK_NEAR(env.enemies[0].x, 950.0f, 0.0);
    CHECK_NEAR(env.enemies[0].vx, 200.0f, 0.0);
    /* Upstream quirk kept: inside the window every tick flips again, so an enemy at x = 1 jitters 1 <-> 2.67. */
    CHECK_NEAR(env.enemies[1].vx, -200.0f, 0.0);
    CHECK_NEAR(env.enemies[1].x, 1.0f, 1e-3);
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK_NEAR(env.enemies[1].vx, 200.0f, 0.0);
    CHECK_NEAR(env.enemies[1].x, 1.0f + 200.0f / 120.0f, 1e-3);
}

static void place_missile(Env *env, float x, float y, int owner) {
    Missile *m = &env->missiles[env->n_missiles++];
    m->x = x;
    m->y = y;
    m->vy = 30.0f;
    m->owner = owner;
    m->id = env->next_missile_id++;
}

void test_env_missile_damages_every_overlapping_live_enemy(void) {
    Env env;
    quiet_env(&env);
    env_add_enemy(&env, 2, 100.0f, 300.0f); /* centre (125, 325) */
    env_add_enemy(&env, 2, 105.0f, 305.0f); /* centre (130, 330) */
    env_add_enemy(&env, 2, 600.0f, 300.0f); /* far away */
    env_add_enemy(&env, 2, 102.0f, 302.0f); /* overlapping, but already an explosion */
    env.enemies[3].dead = 1;
    place_missile(&env, 117.0f, 317.0f, 1); /* centre (127, 327) */
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK_NEAR(env.enemies[0].health, 30.0f, 0.0);
    CHECK_NEAR(env.enemies[1].health, 30.0f, 0.0);
    CHECK_NEAR(env.enemies[2].health, 40.0f, 0.0);
    CHECK_NEAR(env.enemies[3].health, 40.0f, 0.0);
    CHECK(env.n_missiles == 0);
    CHECK(env.stats.hits[1] == 2);
    CHECK(env.events[1].hits == 2);
    CHECK(env.stats.hits[0] == 0);

    /* A missile just outside 10 + 25 px of every centre keeps flying. */
    quiet_env(&env);
    env_add_enemy(&env, 0, 100.0f, 300.0f);           /* centre (125, 325) */
    place_missile(&env, 115.0f, 300.0f - 35.0f, 0); /* centre (125, 275): 50 px away */
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK(env.n_missiles == 1);
    CHECK_NEAR(env.enemies[0].health, 20.0f, 0.0);
}

void test_env_kill_becomes_explosion_credited_to_shooter(void) {
    Env env;
    quiet_env(&env);
    env_add_enemy(&env, 0, 100.0f, 300.0f); /* centre (125, 325) */
    env.enemies[0].health = 10.0f;
    place_missile(&env, 117.0f, 317.0f, 2); /* kills it */
    place_missile(&env, 118.0f, 318.0f, 0); /* same spot, but the enemy is already an explosion */
    hold(&env, IDLE, IDLE, IDLE, 1);
    const Enemy *e = &env.enemies[0];
    CHECK(e->dead == 1);
    CHECK_NEAR(e->radius, 50.0f, 0.0);
    CHECK_NEAR(e->damage, 1.0f, 0.0);
    CHECK(e->attack_cd_ticks == 18);
    CHECK_NEAR(e->accel, -1.0f, 0.0);
    CHECK_NEAR(e->vx, 1.0f, 0.0);
    CHECK_NEAR(e->vy, 100.0f - 1.0f / 120.0f, 1e-4);
    CHECK_NEAR(e->x, 100.0f + 1.0f / 120.0f, 1e-4);
    CHECK(env.stats.score[2] == 2);
    CHECK(env.stats.kills[2] == 1);
    CHECK(env.events[2].kills == 1 && env.events[2].kill_points == 2);
    CHECK(env.events[0].ally_kill_points == 2 && env.events[1].ally_kill_points == 2);
    CHECK(env.events[2].ally_kill_points == 0);
    CHECK(env.stats.hits[2] == 1 && env.stats.hits[0] == 0);
    CHECK(env.n_missiles == 1 && env.missiles[0].owner == 0);
}

static Enemy *frozen_enemy(Env *env, int type, float x, float y) {
    Enemy *e = env_add_enemy(env, type, x, y);
    e->vx = 0.0f;
    e->vy = 0.0f;
    e->accel = 0.0f;
    return e;
}

void test_env_contact_damage_respects_enemy_cooldown(void) {
    Env env;
    quiet_env(&env);
    frozen_enemy(&env, 0, 400.0f, 700.0f); /* exactly on ship 0; ship 1 is 50 px away (not < 50) */
    hold(&env, IDLE, IDLE, IDLE, 240);
    CHECK_NEAR(env.ships[0].health, 190.0f, 0.0);
    CHECK_NEAR(env.stats.damage_taken[0], 10.0f, 0.0);
    CHECK_NEAR(env.ships[1].health, 200.0f, 0.0);
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK_NEAR(env.ships[0].health, 185.0f, 0.0);

    /* Upstream quirk kept: one cooldown per enemy, checked in ship order, so ship 0 always takes it. */
    quiet_env(&env);
    frozen_enemy(&env, 2, 425.0f, 700.0f); /* 25 px from both ship 0 and ship 1 */
    hold(&env, IDLE, IDLE, IDLE, 240);
    CHECK_NEAR(env.ships[0].health, 180.0f, 0.0);
    CHECK_NEAR(env.ships[1].health, 200.0f, 0.0);

    /* Explosions hurt on contact too: 1 damage every 18 ticks. */
    quiet_env(&env);
    Enemy *x = frozen_enemy(&env, 0, 375.0f, 675.0f);
    x->dead = 1;
    x->radius = 50.0f;
    x->damage = 1.0f;
    x->attack_cd_ticks = 18;
    hold(&env, IDLE, IDLE, IDLE, 36);
    CHECK_NEAR(env.ships[0].health, 198.0f, 0.0);
}

void test_env_ship_dies_at_zero_health(void) {
    Env env;
    quiet_env(&env);
    env.ships[0].health = 3.0f;
    frozen_enemy(&env, 2, 400.0f, 700.0f); /* 10 damage, more than the ship has left */
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK(env.ships[0].alive == 0);
    CHECK(env.stats.died[0] == 1);
    CHECK(env.events[0].died == 1);
    CHECK_NEAR(env.stats.damage_taken[0], 3.0f, 0.0); /* loss clamped to the health that was left */
    CHECK(env.stats.alive_ticks[0] == 0);

    ShipAction go = {H_LEFT, V_UP, 1};
    hold(&env, go, IDLE, IDLE, 400);
    CHECK_NEAR(env.ships[0].x, 400.0f, 0.0);
    CHECK_NEAR(env.ships[0].y, 700.0f, 0.0);
    CHECK(env.n_missiles == 0 && env.stats.shots[0] == 0);
    CHECK_NEAR(env.stats.damage_taken[0], 3.0f, 0.0);
    CHECK(env.stats.died[0] == 1 && env.events[0].died == 1);
    CHECK(env.stats.alive_ticks[1] == 401);
    CHECK(env.ships[1].alive == 1);
}

void test_env_counts_live_enemies_escaping_bottom(void) {
    Env env;
    quiet_env(&env);
    env.ships[2].alive = 0;
    env_add_enemy(&env, 0, 100.0f, 1000.0f); /* crosses y > 1000 on this tick */
    Enemy *x = env_add_enemy(&env, 0, 300.0f, 1000.0f);
    x->dead = 1; /* an explosion leaving is not an escape */
    Enemy *stay = env_add_enemy(&env, 1, 500.0f, 900.0f);
    uint32_t stay_id = stay->id;
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK(env.n_enemies == 1);
    CHECK(env.enemies[0].id == stay_id);
    CHECK(env.stats.escapes == 1);
    CHECK(env.events[0].escapes == 1 && env.events[1].escapes == 1);
    CHECK(env.events[2].escapes == 0);

    quiet_env(&env);
    env_add_enemy(&env, 0, 100.0f, 999.0f); /* still inside after one tick */
    hold(&env, IDLE, IDLE, IDLE, 1);
    CHECK(env.n_enemies == 1 && env.stats.escapes == 0);
}

void test_env_step_repeats_action_for_4_ticks(void) {
    Env env;
    quiet_env(&env);
    env.events[0].hits = 7; /* stale events from an earlier decision */
    ShipAction a[NUM_SHIPS] = {{H_LEFT, V_STAY, 1}, IDLE, IDLE};
    env_step(&env, a);
    CHECK(env.tick == 4);
    CHECK(env.decision == 1);
    CHECK_NEAR(env.ships[0].x, 400.0f - 4.0f * 400.0f / 120.0f, 1e-3);
    CHECK(env.events[0].hits == 0);
    CHECK(env.events[0].shots == 1);
    env_step(&env, a);
    CHECK(env.tick == 8 && env.decision == 2);
    CHECK(env.events[0].shots == 0); /* still cooling down; events cover this decision only */

    quiet_env(&env);
    env.action_repeat = 1;
    env_step(&env, a);
    CHECK(env.tick == 1 && env.decision == 1);
}

void test_env_episode_ends_terminated_or_truncated(void) {
    Env env;
    ShipAction a[NUM_SHIPS] = {IDLE, IDLE, IDLE};

    quiet_env(&env);
    env.max_ticks = 8;
    env_step(&env, a);
    CHECK(env.terminated == 0 && env.truncated == 0);
    env_step(&env, a);
    CHECK(env.terminated == 0 && env.truncated == 1);
    env_step(&env, a); /* finished episodes do not advance until reset */
    CHECK(env.tick == 8 && env.decision == 2);

    quiet_env(&env);
    for (int i = 0; i < NUM_SHIPS; i++) {
        env.ships[i].health = 1.0f;
        frozen_enemy(&env, 0, env.ships[i].x, env.ships[i].y);
    }
    env_step(&env, a);
    CHECK(env.terminated == 1 && env.truncated == 0);
    CHECK(env.tick == 1); /* the decision stops at the tick the last ship dies */

    quiet_env(&env);
    env.max_ticks = 4;
    for (int i = 0; i < NUM_SHIPS; i++) {
        env.ships[i].health = 1.0f;
        frozen_enemy(&env, 0, env.ships[i].x, env.ships[i].y);
    }
    env_step(&env, a);
    CHECK(env.terminated == 1 && env.truncated == 0); /* team defeat takes precedence */

    quiet_env(&env);
    env.ships[0].alive = 0;
    env.ships[1].alive = 0;
    env.max_ticks = 4;
    env_step(&env, a);
    CHECK(env.terminated == 0 && env.truncated == 1); /* one survivor keeps the episode going */
}

void test_env_rewards_follow_configured_coefficients(void) {
    RewardConfig cfg = reward_config_default();
    CHECK_NEAR(cfg.kill_points, 0.125, 1e-7);
    CHECK_NEAR(cfg.hit, 0.02, 1e-7);
    CHECK_NEAR(cfg.shot, -0.01, 1e-7);
    CHECK_NEAR(cfg.damage, -0.01, 1e-7);
    CHECK_NEAR(cfg.death, -2.0, 1e-7);
    CHECK_NEAR(cfg.escape, -0.05, 1e-7);
    CHECK_NEAR(cfg.ally_share, 0.0, 0.0);

    Env env;
    quiet_env(&env);
    float r[NUM_SHIPS];
    env.events[0] = (ShipEvents){.kills = 1, .kill_points = 8, .ally_kill_points = 4, .hits = 3, .shots = 2,
                                 .health_lost = 7.0f, .died = 0, .escapes = 2};
    env.events[1] = (ShipEvents){.health_lost = 3.0f, .died = 1};
    env_rewards(&env, &cfg, r);
    CHECK_NEAR(r[0], 1.0 + 0.06 - 0.02 - 0.07 - 0.10, 1e-6);
    CHECK_NEAR(r[1], -0.03 - 2.0, 1e-6);
    CHECK_NEAR(r[2], 0.0, 0.0);

    cfg.ally_share = 0.5f;
    env_rewards(&env, &cfg, r);
    CHECK_NEAR(r[0], 0.87 + 0.5 * 0.125 * 4.0, 1e-6);

    const RewardParts *p = &env.stats.parts[0];
    CHECK_NEAR(p->kill, 2.0, 1e-6);
    CHECK_NEAR(p->hit, 0.12, 1e-6);
    CHECK_NEAR(p->shot, -0.04, 1e-6);
    CHECK_NEAR(p->damage, -0.14, 1e-6);
    CHECK_NEAR(p->escape, -0.20, 1e-6);
    CHECK_NEAR(p->ally, 0.25, 1e-6);
    CHECK_NEAR(p->kill + p->hit + p->shot + p->damage + p->death + p->escape + p->ally,
               env.stats.episode_return[0], 1e-5);
    CHECK_NEAR(env.stats.episode_return[0], 0.87 + 1.12, 1e-5);
    CHECK_NEAR(env.stats.parts[1].death, -4.0, 1e-6);
}

void test_env_observation_layout_and_bounds(void) {
    CHECK(OBS_DIM == 139);
    CHECK(OBS_ALLY_OFFSET == 7 && OBS_ENEMY_OFFSET == 19 && OBS_MISSILE_OFFSET == 107);
    Env env;
    quiet_env(&env);
    float o[OBS_DIM];
    env_observe(&env, 0, o);
    CHECK_NEAR(o[0], 0.425, 1e-6); /* centre x 425 */
    CHECK_NEAR(o[1], 0.725, 1e-6);
    CHECK_NEAR(o[2], 1.0, 0.0);
    CHECK_NEAR(o[3], 0.0, 0.0);
    CHECK_NEAR(o[4], 1.0, 0.0);
    CHECK_NEAR(o[5], 1.0, 0.0);
    CHECK_NEAR(o[6], 1.0, 0.0);
    /* Allies in order ship 1, ship 2. */
    CHECK_NEAR(o[OBS_ALLY_OFFSET + 0], 1.0, 0.0);
    CHECK_NEAR(o[OBS_ALLY_OFFSET + 1], 0.05, 1e-6);
    CHECK_NEAR(o[OBS_ALLY_OFFSET + 6 + 1], 0.10, 1e-6);
    int empty = 1;
    for (int k = OBS_ENEMY_OFFSET; k < OBS_DIM; k++) {
        empty &= o[k] == 0.0f;
    }
    CHECK(empty);

    env.ships[2].alive = 0;
    env_add_enemy(&env, 2, 400.0f, 100.0f); /* 600 px above ship 0 */
    env_add_enemy(&env, 1, 400.0f, 500.0f); /* 200 px above: nearest */
    env_add_enemy(&env, 0, 700.0f, 700.0f); /* 300 px right */
    env.enemies[1].attack_cd_left = 5;
    place_missile(&env, 420.0f, 600.0f, 0);
    env.ships[0].ammo = 40;
    env_observe(&env, 0, o);
    CHECK_NEAR(o[4], 0.4, 1e-6);
    int a2 = OBS_ALLY_OFFSET + 6;
    CHECK(o[a2] == 0.0f && o[a2 + 1] == 0.0f && o[a2 + 5] == 0.0f); /* a dead ally is all zeros */
    const float *e0 = o + OBS_ENEMY_OFFSET;
    CHECK_NEAR(e0[0], 1.0, 0.0);
    CHECK_NEAR(e0[1], 0.0, 1e-6);
    CHECK_NEAR(e0[2], -0.2, 1e-6);
    CHECK_NEAR(e0[3], 0.25, 1e-6);   /* vx 200 / 800 */
    CHECK_NEAR(e0[4], 0.01, 1e-6);   /* vy 10 / 1000 */
    CHECK_NEAR(e0[5], 0.75, 1e-6);   /* health 30 / 40 */
    CHECK(e0[6] == 0.0f);            /* not an explosion */
    CHECK(e0[7] == 0.0f && e0[8] == 1.0f && e0[9] == 0.0f);
    CHECK(e0[10] == 0.0f);           /* on cooldown */
    CHECK_NEAR(o[OBS_ENEMY_OFFSET + 11 + 1], 0.3, 1e-6);   /* second: 300 px right */
    CHECK_NEAR(o[OBS_ENEMY_OFFSET + 22 + 2], -0.6, 1e-6);  /* third: 600 px up */
    CHECK(o[OBS_ENEMY_OFFSET + 33] == 0.0f);               /* fourth slot empty */
    const float *m0 = o + OBS_MISSILE_OFFSET;
    CHECK(m0[0] == 1.0f && m0[3] == 1.0f);
    CHECK_NEAR(m0[1], (430.0 - 425.0) / 1000.0, 1e-6);
    env_observe(&env, 1, o);
    CHECK(o[OBS_MISSILE_OFFSET + 3] == 0.0f); /* not ship 1's missile */

    quiet_env(&env);
    for (int k = 0; k < 10; k++) {
        env_add_enemy(&env, 0, 400.0f, 600.0f - 40.0f * (float)k);
    }
    env_observe(&env, 0, o);
    CHECK(o[OBS_ENEMY_OFFSET + 7 * 11] == 1.0f);
    CHECK_NEAR(o[OBS_ENEMY_OFFSET + 7 * 11 + 2], -(100.0 + 7 * 40.0) / 1000.0, 1e-6);

    /* Random play: every value stays in [-1, 1] and observing never consumes random numbers. */
    env_reset(&env, 4242u);
    Pcg32 policy;
    pcg32_seed(&policy, 9u, 9u);
    int bounded = 1;
    int rng_untouched = 1;
    for (int d = 0; d < 1500 && !env.terminated && !env.truncated; d++) {
        ShipAction a[NUM_SHIPS];
        for (int i = 0; i < NUM_SHIPS; i++) {
            a[i] = (ShipAction){(int)pcg32_bounded(&policy, 3), (int)pcg32_bounded(&policy, 3),
                                (int)pcg32_bounded(&policy, 2)};
        }
        env_step(&env, a);
        for (int i = 0; i < NUM_SHIPS; i++) {
            Pcg32 before = env.rng;
            env_observe(&env, i, o);
            rng_untouched &= before.state == env.rng.state;
            for (int k = 0; k < OBS_DIM; k++) {
                bounded &= o[k] >= -1.0f && o[k] <= 1.0f;
            }
        }
    }
    CHECK(bounded);
    CHECK(rng_untouched);
}

static void random_actions(Pcg32 *policy, ShipAction a[NUM_SHIPS]) {
    for (int i = 0; i < NUM_SHIPS; i++) {
        a[i] = (ShipAction){(int)pcg32_bounded(policy, 3), (int)pcg32_bounded(policy, 3),
                            (int)pcg32_bounded(policy, 2)};
    }
}

void test_env_replay_is_deterministic(void) {
    Env a, b, c;
    env_reset(&a, 2024u);
    env_reset(&b, 2024u);
    env_reset(&c, 2025u);
    Pcg32 pa, pb, pc;
    pcg32_seed(&pa, 1u, 2u);
    pcg32_seed(&pb, 1u, 2u);
    pcg32_seed(&pc, 1u, 2u);
    CHECK(env_hash(&a) == env_hash(&b));
    int same = 1;
    int diverged = 0;
    for (int t = 0; t < 10000; t++) {
        ShipAction xa[NUM_SHIPS], xb[NUM_SHIPS], xc[NUM_SHIPS];
        random_actions(&pa, xa);
        random_actions(&pb, xb);
        random_actions(&pc, xc);
        env_tick(&a, xa);
        env_tick(&b, xb);
        env_tick(&c, xc);
        same &= env_hash(&a) == env_hash(&b);
        diverged |= env_hash(&a) != env_hash(&c);
    }
    CHECK(same);
    CHECK(diverged);
    CHECK(a.stats.spawned > 100);

    /* The hash sees gameplay state, not just the clock. */
    env_reset(&a, 7u);
    env_reset(&b, 7u);
    b.ships[1].x += 1.0f;
    CHECK(env_hash(&a) != env_hash(&b));
}
