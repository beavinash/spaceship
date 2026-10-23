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
    CHECK(OBS_DIM == 148);
    CHECK(OBS_ALLY_OFFSET == 8 && OBS_ENEMY_OFFSET == 20 && OBS_MISSILE_OFFSET == 116);
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
    CHECK_NEAR(o[OBS_ENEMY_OFFSET + OBS_ENEMY + 1], 0.3, 1e-6);      /* second: 300 px right */
    CHECK_NEAR(o[OBS_ENEMY_OFFSET + 2 * OBS_ENEMY + 2], -0.6, 1e-6); /* third: 600 px up */
    CHECK(o[OBS_ENEMY_OFFSET + 3 * OBS_ENEMY] == 0.0f);              /* fourth slot empty */
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
    CHECK(o[OBS_ENEMY_OFFSET + 7 * OBS_ENEMY] == 1.0f);
    CHECK_NEAR(o[OBS_ENEMY_OFFSET + 7 * OBS_ENEMY + 2], -(100.0 + 7 * 40.0) / 1000.0, 1e-6);

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

void test_env_invariants_hold_and_catch_corruption(void) {
    char why[128];
    Env env;
    env_reset(&env, 31u);
    CHECK(env_check_invariants(&env, why, sizeof why) == 0);
    Pcg32 policy;
    pcg32_seed(&policy, 4u, 4u);
    int held = 1;
    while (!env.terminated && !env.truncated) {
        ShipAction a[NUM_SHIPS];
        random_actions(&policy, a);
        env_step(&env, a);
        held &= env_check_invariants(&env, why, sizeof why) == 0;
    }
    CHECK(held);
    CHECK(env.stats.spawned > 100);

    Env bad = env;
    bad.stats.escapes++; /* spawned no longer equals kills + escapes + live enemies */
    CHECK(env_check_invariants(&bad, why, sizeof why) != 0);
    CHECK(strstr(why, "accounting") != NULL);
    bad = env;
    bad.ships[0].ammo = 101;
    CHECK(env_check_invariants(&bad, why, sizeof why) != 0);
    bad = env;
    bad.ships[1].health = 250.0f;
    CHECK(env_check_invariants(&bad, why, sizeof why) != 0);
    bad = env;
    bad.n_enemies = MAX_ENEMIES + 1;
    CHECK(env_check_invariants(&bad, why, sizeof why) != 0);
    bad = env;
    if (bad.n_enemies > 0) {
        bad.enemies[0].x = NAN;
        CHECK(env_check_invariants(&bad, why, sizeof why) != 0);
    }
    bad = env;
    bad.stats.dropped_shots = 1;
    CHECK(env_check_invariants(&bad, why, sizeof why) != 0);
}

/* Fire one missile from ship 0 at a lone enemy in a copy of env; report whether it hit. */
static int fires_and_hits(const Env *start) {
    Env env = *start;
    ShipAction a[NUM_SHIPS] = {FIRE, IDLE, IDLE};
    env_tick(&env, a);
    a[0] = IDLE;
    for (int t = 0; t < 400 && env.n_missiles > 0; t++) {
        env_tick(&env, a);
    }
    return env.stats.hits[0] > 0;
}

void test_env_fire_solution_matches_real_missiles(void) {
    int cases = 0, agree = 0, hits = 0;
    for (int type = 0; type < 3; type++) {
        for (float ex = 290.0f; ex <= 510.0f; ex += 7.0f) {
            for (float ey = 100.0f; ey <= 650.0f; ey += 110.0f) {
                for (int dir = -1; dir <= 1; dir += 2) {
                    Env env;
                    quiet_env(&env);
                    env.ships[1].alive = env.ships[2].alive = 0; /* keep other ships out of the way */
                    Enemy *e = env_add_enemy(&env, type, ex, ey);
                    e->vx = 200.0f * (float)dir;
                    int predicted = -1;
                    float offset = env_fire_solution(&env, 0, 0, &predicted);
                    int actual = fires_and_hits(&env);
                    cases++;
                    agree += predicted == actual && (!predicted || offset < MISSILE_RADIUS + ENEMY_RADIUS);
                    hits += actual;
                }
            }
        }
    }
    CHECK(cases == 3 * 32 * 6 * 2);
    CHECK(agree == cases);
    CHECK(hits > 20 && hits < cases - 20); /* the grid covers both hits and misses */

    Env env;
    quiet_env(&env);
    env_add_enemy(&env, 0, 400.0f, 800.0f); /* below the ship: a missile can never meet it */
    int hit = -1;
    env_fire_solution(&env, 0, 0, &hit);
    CHECK(hit == 0);
    env.enemies[0].dead = 1;
    CHECK(env_fire_solution(&env, 0, 0, &hit) >= FIRE_NO_SOLUTION && hit == 0);
    env.ships[0].alive = 0;
    env.enemies[0].dead = 0;
    CHECK(env_fire_solution(&env, 0, 0, NULL) >= FIRE_NO_SOLUTION);

    /* Straight above and still: a centred hit, so the horizontal offset at contact is ~0. */
    quiet_env(&env);
    Enemy *e = env_add_enemy(&env, 2, 385.0f, 400.0f); /* centre x 410 = missile centre x */
    e->vx = 0.0f;
    CHECK(env_fire_solution(&env, 0, 0, &hit) < 0.5f && hit == 1);
    /* 50 px to the side and still: the missile passes 50 px away. */
    e->x = 435.0f;
    CHECK_NEAR(env_fire_solution(&env, 0, 0, &hit), 50.0, 0.5);
    CHECK(hit == 0);
}

void test_env_observation_fire_solution_features(void) {
    Env env;
    quiet_env(&env);
    env.ships[1].alive = env.ships[2].alive = 0;
    float o[OBS_DIM];
    env_observe(&env, 0, o);
    CHECK(o[7] == 0.0f); /* no enemies: nothing to hit */

    Enemy *above = env_add_enemy(&env, 2, 385.0f, 400.0f); /* dead centre above ship 0, held still */
    above->vx = 0.0f;
    Enemy *aside = env_add_enemy(&env, 2, 435.0f, 300.0f); /* 50 px to the side, held still */
    aside->vx = 0.0f;
    env_add_enemy(&env, 0, 400.0f, 800.0f);                /* below: no shot */
    env_observe(&env, 0, o);
    CHECK(o[7] == 1.0f);
    int ok = 1;
    for (int slot = 0; slot < 3; slot++) {
        const float *f = o + OBS_ENEMY_OFFSET + slot * OBS_ENEMY;
        float dy = f[2] * 1000.0f;
        float q = f[OBS_ENEMY - 1];
        if (dy < -350.0f) {
            ok &= fabsf(q - (1.0f - 50.0f / SHOT_RAMP)) < 0.01f; /* aside: partial quality */
        } else if (dy < 0.0f) {
            ok &= q > 0.99f;                                     /* above: dead centre */
        } else {
            ok &= q == 0.0f;                                     /* below: no shot */
        }
    }
    CHECK(ok);
    int hit = 0;
    CHECK(env_fire_solution(&env, 0, 2, &hit) >= FIRE_NO_SOLUTION && hit == 0);
    CHECK_NEAR(env_shot_quality(&env, 0, &hit), 1.0, 0.01);
    CHECK(hit == 1);

    env.enemies[0].x = 600.0f; /* move the centred enemy away: only the 50 px near miss remains */
    env_observe(&env, 0, o);
    CHECK(o[7] == 0.0f);
    CHECK_NEAR(env_shot_quality(&env, 0, &hit), 1.0 - 50.0 / SHOT_RAMP, 0.01);
    CHECK(hit == 0);
}

void test_env_alignment_shaping_is_potential_based(void) {
    RewardConfig cfg = reward_config_default();
    CHECK_NEAR(cfg.align, 0.5, 1e-7);
    CHECK_NEAR(cfg.gamma, 0.99, 1e-7);
    Env env;
    float r[NUM_SHIPS];
    int hit;

    /* Step 1: an enemy is held dead centre above ship 0, so Phi goes 0 -> q. */
    quiet_env(&env);
    Enemy *e = env_add_enemy(&env, 2, 385.0f, 300.0f);
    e->vx = 0.0f;
    e->vy = 0.0f;
    e->accel = 0.0f;
    ShipAction idle[NUM_SHIPS] = {IDLE, IDLE, IDLE};
    env_step(&env, idle);
    float q1 = env_shot_quality(&env, 0, &hit);
    CHECK(q1 > 0.9f);
    env_rewards(&env, &cfg, r);
    CHECK_NEAR(env.stats.parts[0].align, 0.5 * 0.99 * q1, 1e-5);

    /* Step 2: the enemy moves out of reach, so Phi goes q -> 0. */
    env.enemies[0].x = 800.0f;
    env_step(&env, idle);
    env_rewards(&env, &cfg, r);
    CHECK_NEAR(env.stats.parts[0].align, 0.5 * 0.99 * q1 - 0.5 * q1, 1e-5);

    /* No ammo means no shot: Phi is 0 even when lined up. */
    env.enemies[0].x = 385.0f;
    env.ships[0].ammo = 0;
    float before = env.stats.parts[0].align;
    env_step(&env, idle);
    env_rewards(&env, &cfg, r);
    CHECK_NEAR(env.stats.parts[0].align, before, 1e-6);

    /* Death is terminal for the ship: Phi' = 0, so the shaping returns -0.5 * Phi. */
    quiet_env(&env);
    e = env_add_enemy(&env, 2, 385.0f, 300.0f);
    e->vx = e->vy = e->accel = 0.0f;
    env_step(&env, idle);
    env_rewards(&env, &cfg, r);
    float q = env_shot_quality(&env, 0, &hit);
    before = env.stats.parts[0].align;
    env.ships[0].health = 1.0f;
    frozen_enemy(&env, 0, env.ships[0].x, env.ships[0].y);
    env_step(&env, idle);
    CHECK(!env.ships[0].alive);
    env_rewards(&env, &cfg, r);
    CHECK_NEAR(env.stats.parts[0].align - before, -0.5 * q, 1e-5);

    /* With gamma = 1 the shaping telescopes over an episode: sum = 0.5 * (Phi_end - Phi_0) = 0, since
     * Phi_0 = 0 at reset and every ending (death, defeat, 120 s limit) is terminal with Phi_end = 0. */
    cfg.gamma = 1.0f;
    env_reset(&env, 77u);
    env.max_ticks = 2400;
    Pcg32 policy;
    pcg32_seed(&policy, 8u, 1u);
    while (!env.terminated && !env.truncated) {
        ShipAction a[NUM_SHIPS];
        random_actions(&policy, a);
        env_step(&env, a);
        env_rewards(&env, &cfg, r);
    }
    for (int i = 0; i < NUM_SHIPS; i++) {
        CHECK_NEAR(env.stats.parts[i].align, 0.0, 1e-4);
    }
}

void test_env_edge_penalty_only_inside_band(void) {
    RewardConfig cfg = reward_config_default();
    CHECK_NEAR(cfg.edge, -0.003, 1e-7);
    CHECK_NEAR(EDGE_BAND, 60.0, 0.0);
    float r[NUM_SHIPS];
    ShipAction idle[NUM_SHIPS] = {IDLE, IDLE, IDLE};
    const struct {
        float x, y;
        int alive;
        int charged;
    } cases[] = {
        {10.0f, 500.0f, 1, 1},  /* centre x 35: left band */
        {35.0f, 500.0f, 1, 0},  /* centre x 60: on the boundary, outside */
        {920.0f, 500.0f, 1, 1}, /* centre x 945: right band */
        {400.0f, 930.0f, 1, 1}, /* centre y 955: bottom band */
        {400.0f, 900.0f, 1, 0}, /* centre y 925: outside */
        {400.0f, 10.0f, 1, 0},  /* the top edge is not penalised */
        {10.0f, 940.0f, 0, 0},  /* dead ships are never charged */
    };
    int ok = 1;
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        Env env;
        quiet_env(&env);
        env.ships[0].x = cases[c].x;
        env.ships[0].y = cases[c].y;
        env.ships[0].alive = cases[c].alive;
        env.ships[0].health = cases[c].alive ? SHIP_HEALTH : 0.0f;
        env_step(&env, idle);
        env_rewards(&env, &cfg, r);
        ok &= fabsf(env.stats.parts[0].edge - (cases[c].charged ? -0.003f : 0.0f)) < 1e-7f;
        ok &= env.stats.parts[1].edge == 0.0f;
    }
    CHECK(ok);
}

void test_env_counts_edge_ticks_and_aimed_shots(void) {
    Env env;
    quiet_env(&env);
    env.ships[0].x = 10.0f; /* inside the left band */
    hold(&env, IDLE, IDLE, IDLE, 50);
    CHECK(env.stats.edge_ticks[0] == 50);
    CHECK(env.stats.edge_ticks[1] == 0);
    CHECK(env.stats.alive_ticks[0] == 50);

    quiet_env(&env);
    Enemy *e = env_add_enemy(&env, 2, 385.0f, 300.0f); /* dead centre above ship 0 */
    e->vx = e->vy = e->accel = 0.0f;
    hold(&env, FIRE, FIRE, IDLE, 1); /* ship 0 has a shot; ship 1, 50 px aside, does not */
    CHECK(env.stats.shots[0] == 1 && env.stats.aimed_shots[0] == 1);
    CHECK(env.stats.shots[1] == 1 && env.stats.aimed_shots[1] == 0);
}

/* Largest per-slot shot-quality input in ship 0's observation. */
static float best_slot_quality(const Env *env) {
    float o[OBS_DIM], best = 0.0f;
    env_observe(env, 0, o);
    for (int s = 0; s < OBS_ENEMY_SLOTS; s++) {
        float q = o[OBS_ENEMY_OFFSET + s * OBS_ENEMY + OBS_ENEMY - 1];
        best = q > best ? q : best;
    }
    return best;
}

void test_env_potential_uses_observed_slots(void) {
    Env env;
    quiet_env(&env);
    env.ships[1].alive = env.ships[2].alive = 0;
    env.ships[1].health = env.ships[2].health = 0.0f;
    Enemy *target = env_add_enemy(&env, 2, 385.0f, 100.0f); /* dead centre above ship 0, ~600 px away */
    target->vx = target->vy = target->accel = 0.0f;
    const float xs[8] = {100, 200, 600, 700, 100, 200, 600, 700};
    const float ys[8] = {500, 500, 500, 500, 350, 350, 350, 350};
    for (int k = 0; k < 8; k++) { /* eight still explosions, all nearer than the target */
        Enemy *x = env_add_enemy(&env, 0, xs[k], ys[k]);
        x->dead = 1;
        x->radius = EXPLOSION_RADIUS;
        x->vx = x->vy = x->accel = 0.0f;
    }
    int hit = 0;
    CHECK(env_shot_quality(&env, 0, &hit) > 0.99f && hit == 1); /* the shot exists... */
    CHECK_NEAR(env_potential(&env, 0), 0.0, 0.0);               /* ...but ship 0 cannot see it */
    CHECK_NEAR(env_potential(&env, 0), best_slot_quality(&env), 1e-6);

    env.n_enemies--; /* drop the last explosion: the target takes the 8th slot */
    CHECK(env_potential(&env, 0) > 0.99f);
    CHECK_NEAR(env_potential(&env, 0), best_slot_quality(&env), 1e-6);

    /* The reward's shaping term follows the observed potential. */
    RewardConfig cfg = reward_config_default();
    float r[NUM_SHIPS];
    ShipAction idle[NUM_SHIPS] = {IDLE, IDLE, IDLE};
    env_step(&env, idle);
    env_rewards(&env, &cfg, r);
    float seen = env_potential(&env, 0);
    CHECK_NEAR(env.stats.parts[0].align, 0.5 * 0.99 * seen, 1e-5);
    env.n_enemies++; /* the eighth explosion is back (its slot data was kept): the target is hidden again */
    env_step(&env, idle);
    env_rewards(&env, &cfg, r);
    CHECK_NEAR(env.stats.parts[0].align, 0.5 * 0.99 * seen - 0.5 * seen, 1e-5);
}

void test_env_time_limit_is_terminal_for_shaping(void) {
    Env env;
    quiet_env(&env);
    env.max_ticks = 8; /* two decisions */
    Enemy *e = env_add_enemy(&env, 2, 385.0f, 300.0f);
    e->vx = e->vy = e->accel = 0.0f;
    RewardConfig cfg = reward_config_default();
    float r[NUM_SHIPS];
    ShipAction idle[NUM_SHIPS] = {IDLE, IDLE, IDLE};
    env_step(&env, idle);
    env_rewards(&env, &cfg, r);
    float q = env_potential(&env, 0);
    CHECK(q > 0.9f && !env.truncated);
    CHECK_NEAR(env.stats.parts[0].align, 0.5 * 0.99 * q, 1e-5);
    env_step(&env, idle); /* still lined up, but the 120 s limit (here 8 ticks) ends the episode */
    CHECK(env.truncated);
    env_rewards(&env, &cfg, r);
    CHECK_NEAR(env.stats.parts[0].align, 0.5 * 0.99 * q - 0.5 * q, 1e-5);
    CHECK_NEAR(env.potential[0], 0.0, 0.0);
}
