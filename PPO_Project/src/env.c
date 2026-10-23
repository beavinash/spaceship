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

void env_tick(Env *env, const ShipAction actions[NUM_SHIPS]) {
    for (int i = 0; i < NUM_SHIPS; i++) {
        if (env->ships[i].alive) {
            move_ship(&env->ships[i], actions[i]);
        }
    }
    env->tick++;
}
