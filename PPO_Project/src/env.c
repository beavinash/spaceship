#include "env.h"

#include <math.h>
#include <string.h>

/* Upstream Aliencraft_Type_1/2/3: ENEMY_HEALTH 40 scaled 0.5/0.75/1, acceleration 90 scaled 1/0.8/0.6. */
const EnemyType ENEMY_TYPES[3] = {
    {20.0f, 5.0f, 90.0f, 2, 120},
    {30.0f, 7.0f, 72.0f, 4, 144},
    {40.0f, 10.0f, 54.0f, 8, 168},
};

void env_reset(Env *env, uint64_t seed) {
    memset(env, 0, sizeof *env);
    env->seed = seed;
    pcg32_seed(&env->rng, seed, 0x5eedu);
    for (int i = 0; i < NUM_SHIPS; i++) {
        Ship *s = &env->ships[i];
        s->x = SHIP_START_X + SHIP_START_SPACING * (float)i;
        s->y = SHIP_START_Y;
        s->health = SHIP_HEALTH;
        s->alive = 1;
        s->ammo = SHIP_AMMO;
        s->weapon_cd = 0;
    }
    env->spawn_timer = ENEMY_SPAWN_TICKS;
    env->max_decisions = MAX_DECISIONS;
    env->action_repeat = ACTION_REPEAT;
    env->next_enemy_id = 1;
    env->next_missile_id = 1;
}

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Upstream Control_Spaceship: unit direction scaled to ship speed, then clamped to the arena. */
static void move_ship(Ship *s, ShipAction a) {
    float dx = (a.h == H_LEFT) ? -1.0f : (a.h == H_RIGHT ? 1.0f : 0.0f);
    float dy = (a.v == V_UP) ? -1.0f : (a.v == V_DOWN ? 1.0f : 0.0f);
    float mag = sqrtf(dx * dx + dy * dy);
    if (mag > 0.0f) {
        dx = SHIP_SPEED * dx / mag;
        dy = SHIP_SPEED * dy / mag;
    }
    s->x = clampf(s->x + dx * TICK_DT, 0.0f, ARENA_SIZE - 2.0f * SHIP_RADIUS);
    s->y = clampf(s->y + dy * TICK_DT, 0.0f, ARENA_SIZE - 2.0f * SHIP_RADIUS);
}

/* Upstream ShootMissile: launched from the ship's top-left corner with 30 px/s upward speed. */
static void fire_missile(Env *env, int owner) {
    Ship *s = &env->ships[owner];
    if (env->n_missiles >= MAX_MISSILES) {
        env->stats.dropped_shots++;
        return;
    }
    Missile *m = &env->missiles[env->n_missiles++];
    m->x = s->x;
    m->y = s->y;
    m->vy = MISSILE_START_VY;
    m->owner = owner;
    m->id = env->next_missile_id++;
    s->ammo--;
    s->weapon_cd = WEAPON_COOLDOWN_TICKS;
    env->events[owner].shots++;
    env->stats.shots[owner]++;
}

/* Upstream Control_Spaceship: move, count down the weapon, then fire if ready and (unlike upstream) loaded. */
static void control_ship(Env *env, int i, ShipAction a) {
    Ship *s = &env->ships[i];
    move_ship(s, a);
    if (s->weapon_cd > 0) {
        s->weapon_cd--;
    }
    if (a.fire && s->weapon_cd <= 0 && s->ammo > 0) {
        fire_missile(env, i);
    }
}

static void remove_missile(Env *env, int idx) {
    memmove(&env->missiles[idx], &env->missiles[idx + 1],
            (size_t)(env->n_missiles - idx - 1) * sizeof env->missiles[0]);
    env->n_missiles--;
}

/* Upstream Iterate_Projectiles(Detect_Projectile_Collisions): spent or off-screen missiles leave the list. */
static void resolve_missiles(Env *env) {
    int m = 0;
    while (m < env->n_missiles) {
        if (env->missiles[m].y < -MISSILE_RADIUS) {
            remove_missile(env, m);
            continue;
        }
        m++;
    }
}

/* Upstream AccelerateStraightVPath: speed up, then move up. */
static void move_missiles(Env *env) {
    for (int m = 0; m < env->n_missiles; m++) {
        Missile *mi = &env->missiles[m];
        mi->vy += MISSILE_ACCEL * TICK_DT;
        mi->y -= mi->vy * TICK_DT;
    }
}

/* Upstream Aliencraft_Type_N: random type, then random x in [0, 800], entering just above the arena. */
static void spawn_enemy(Env *env) {
    int type = (int)pcg32_bounded(&env->rng, 3);
    float x = (float)pcg32_bounded(&env->rng, ENEMY_SPAWN_MAX_X + 1);
    if (env->n_enemies >= MAX_ENEMIES) {
        env->stats.dropped_spawns++;
        return;
    }
    const EnemyType *t = &ENEMY_TYPES[type];
    Enemy *e = &env->enemies[env->n_enemies++];
    memset(e, 0, sizeof *e);
    e->x = x;
    e->y = -ENEMY_RADIUS;
    e->vx = ENEMY_START_VX;
    e->vy = ENEMY_START_VY;
    e->accel = t->accel;
    e->health = t->health;
    e->damage = t->damage;
    e->radius = ENEMY_RADIUS;
    e->type = type;
    e->points = t->points;
    e->attack_cd_ticks = t->attack_cooldown_ticks;
    e->id = env->next_enemy_id++;
    env->stats.spawned++;
}

static void update_spawner(Env *env) {
    if (--env->spawn_timer <= 0) {
        spawn_enemy(env);
        env->spawn_timer = ENEMY_SPAWN_TICKS;
    }
}

void env_tick(Env *env, const ShipAction actions[NUM_SHIPS]) {
    update_spawner(env);
    for (int i = 0; i < NUM_SHIPS; i++) {
        if (env->ships[i].alive) {
            control_ship(env, i, actions[i]);
        }
    }
    resolve_missiles(env);
    move_missiles(env);
    env->tick++;
}
