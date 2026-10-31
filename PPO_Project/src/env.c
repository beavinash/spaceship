#include "env.h"

#include "hash.h"

#include <math.h>
#include <stdio.h>
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
    env->max_ticks = EPISODE_TICKS;
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

static int near_edge(const Ship *s) {
    float cx = s->x + SHIP_RADIUS, cy = s->y + SHIP_RADIUS;
    return s->alive && (cx < EDGE_BAND || cx > ARENA_SIZE - EDGE_BAND || cy > ARENA_SIZE - EDGE_BAND);
}

/* Upstream ShootMissile: launched from the ship's top-left corner with 30 px/s upward speed. */
static void fire_missile(Env *env, int owner) {
    Ship *s = &env->ships[owner];
    if (env->n_missiles >= MAX_MISSILES) {
        env->stats.dropped_shots++;
        return;
    }
    int aimed;
    env_shot_quality(env, owner, &aimed);
    env->stats.aimed_shots[owner] += aimed;
    Missile *m = &env->missiles[env->n_missiles++];
    m->x = s->x;
    m->y = s->y;
    m->vy = MISSILE_START_VY;
    m->owner = owner;
    m->id = env->next_missile_id++;
    s->ammo--;
    if (s->ammo == 0) {
        env->stats.ammo_out_tick[owner] = env->tick + 1;
    }
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

static float centre_distance(float ax, float ay, float ar, float bx, float by, float br) {
    float dx = (bx + br) - (ax + ar);
    float dy = (by + br) - (ay + ar);
    return sqrtf(dx * dx + dy * dy);
}

/* Upstream kill branch: the wreck keeps its top-left corner and drifts down as a 50 px explosion. */
static void explode(Env *env, Enemy *e, int shooter) {
    e->dead = 1;
    e->vx = EXPLOSION_VX;
    e->vy = EXPLOSION_VY;
    e->accel = EXPLOSION_ACCEL;
    e->radius = EXPLOSION_RADIUS;
    e->damage = EXPLOSION_DAMAGE;
    e->attack_cd_ticks = EXPLOSION_COOLDOWN_TICKS; /* the remaining cooldown carries over, as upstream */
    env->stats.score[shooter] += e->points;
    env->stats.kills[shooter]++;
    env->events[shooter].kills++;
    env->events[shooter].kill_points += e->points;
    for (int i = 0; i < NUM_SHIPS; i++) {
        if (i != shooter) {
            env->events[i].ally_kill_points += e->points;
        }
    }
}

/*
 * Upstream Detect_Enemy_Collisions: the missile damages every live enemy it overlaps
 * (upstream keeps iterating after the first hit), and is spent if it hit anything.
 */
static int missile_hits(Env *env, const Missile *mi) {
    int collided = 0;
    for (int k = 0; k < env->n_enemies; k++) {
        Enemy *e = &env->enemies[k];
        if (e->dead) {
            continue;
        }
        if (centre_distance(mi->x, mi->y, MISSILE_RADIUS, e->x, e->y, e->radius) < MISSILE_RADIUS + e->radius) {
            e->health -= MISSILE_DAMAGE;
            env->events[mi->owner].hits++;
            env->stats.hits[mi->owner]++;
            collided = 1;
            if (e->health <= 0.0f) {
                explode(env, e, mi->owner);
            }
        }
    }
    return collided;
}

/* Upstream Iterate_Projectiles(Detect_Projectile_Collisions): spent or off-screen missiles leave the list. */
static void resolve_missiles(Env *env) {
    int m = 0;
    while (m < env->n_missiles) {
        int collided = missile_hits(env, &env->missiles[m]);
        if (collided || env->missiles[m].y < -MISSILE_RADIUS) {
            remove_missile(env, m);
            continue;
        }
        m++;
    }
}

/* Upstream AccelerateStraightVPath: speed up, then move up. */
static void step_missile(Missile *mi) {
    mi->vy += MISSILE_ACCEL * TICK_DT;
    mi->y -= mi->vy * TICK_DT;
}

static void move_missiles(Env *env) {
    for (int m = 0; m < env->n_missiles; m++) {
        step_missile(&env->missiles[m]);
    }
}

Enemy *env_add_enemy(Env *env, int type, float x, float y) {
    if (env->n_enemies >= MAX_ENEMIES) {
        env->stats.dropped_spawns++;
        return NULL;
    }
    const EnemyType *t = &ENEMY_TYPES[type];
    Enemy *e = &env->enemies[env->n_enemies++];
    memset(e, 0, sizeof *e);
    e->x = x;
    e->y = y;
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
    return e;
}

/* Upstream Aliencraft_Type_N: random type, then random x in [0, 800], entering just above the arena. */
static void spawn_enemy(Env *env) {
    int type = (int)pcg32_bounded(&env->rng, 3);
    float x = (float)pcg32_bounded(&env->rng, ENEMY_SPAWN_MAX_X + 1);
    env_add_enemy(env, type, x, -ENEMY_RADIUS);
}

static void update_spawner(Env *env) {
    if (--env->spawn_timer <= 0) {
        spawn_enemy(env);
        env->spawn_timer = ENEMY_SPAWN_TICKS;
    }
}

/* Upstream AccelerateZigZagVPath plus Move_Enemy's horizontal clamp. */
static void step_enemy(Enemy *e) {
    e->vy += e->accel * TICK_DT;
    e->y += e->vy * TICK_DT;
    int c = (int)floorf(e->x) % ZIGZAG_PERIOD;
    if (c < ZIGZAG_WINDOW && c > 0) {
        e->vx = -e->vx;
    }
    e->x = clampf(e->x + e->vx * TICK_DT, 0.0f, ARENA_SIZE - 2.0f * e->radius);
}

static void move_enemies(Env *env) {
    for (int k = 0; k < env->n_enemies; k++) {
        step_enemy(&env->enemies[k]);
    }
}

/*
 * Upstream Detect_Spaceship_Collision, run per alive ship in index order: an enemy or explosion
 * touching the ship deals its damage, then waits out its own cooldown before hurting any ship again.
 */
static void resolve_contacts(Env *env) {
    for (int i = 0; i < NUM_SHIPS; i++) {
        Ship *s = &env->ships[i];
        if (!s->alive) {
            continue;
        }
        for (int k = 0; k < env->n_enemies; k++) {
            Enemy *e = &env->enemies[k];
            float dist = centre_distance(s->x, s->y, SHIP_RADIUS, e->x, e->y, e->radius);
            if (dist < SHIP_RADIUS + e->radius && e->attack_cd_left <= 0) {
                float lost = e->damage < s->health ? e->damage : s->health;
                s->health -= e->damage;
                env->events[i].health_lost += lost;
                if (e->dead) {
                    env->events[i].bomb_health_lost += lost;
                    env->stats.bomb_damage[i] += lost;
                }
                env->stats.damage_taken[i] += lost;
                e->attack_cd_left = e->attack_cd_ticks;
                if (s->health <= 0.0f) {
                    /* Upstream parks a dead ship off-screen; here it simply leaves the physics. */
                    s->health = 0.0f;
                    s->alive = 0;
                    env->events[i].died = 1;
                    env->stats.died[i] = 1;
                    break;
                }
            }
        }
    }
}

/* Upstream Update_Enemy_Attack_Cooldown. */
static void cool_down_enemies(Env *env) {
    for (int k = 0; k < env->n_enemies; k++) {
        if (env->enemies[k].attack_cd_left > 0) {
            env->enemies[k].attack_cd_left--;
        }
    }
}

/*
 * Upstream Iterate_Enemies drops anything below y = 1000 on its next pass; that cannot touch a ship,
 * so removing it at the end of this tick is equivalent. A live enemy leaving counts as an escape.
 */
static void remove_escaped(Env *env) {
    int k = 0;
    while (k < env->n_enemies) {
        if (env->enemies[k].y <= ARENA_SIZE) {
            k++;
            continue;
        }
        if (!env->enemies[k].dead) {
            env->stats.escapes++;
            for (int i = 0; i < NUM_SHIPS; i++) {
                env->events[i].escapes += env->ships[i].alive;
            }
        }
        memmove(&env->enemies[k], &env->enemies[k + 1], (size_t)(env->n_enemies - k - 1) * sizeof env->enemies[0]);
        env->n_enemies--;
    }
}

void env_tick(Env *env, const ShipAction actions[NUM_SHIPS]) {
    update_spawner(env);
    resolve_contacts(env);
    for (int i = 0; i < NUM_SHIPS; i++) {
        if (env->ships[i].alive) {
            control_ship(env, i, actions[i]);
        }
    }
    resolve_missiles(env);
    move_enemies(env);
    cool_down_enemies(env);
    move_missiles(env);
    remove_escaped(env);
    for (int i = 0; i < NUM_SHIPS; i++) {
        env->stats.alive_ticks[i] += env->ships[i].alive;
        env->stats.edge_ticks[i] += near_edge(&env->ships[i]);
    }
    env->tick++;
}

static int team_alive(const Env *env) {
    int alive = 0;
    for (int i = 0; i < NUM_SHIPS; i++) {
        alive += env->ships[i].alive;
    }
    return alive;
}

void env_step(Env *env, const ShipAction actions[NUM_SHIPS]) {
    if (env->terminated || env->truncated) {
        return;
    }
    memset(env->events, 0, sizeof env->events);
    for (int t = 0; t < env->action_repeat && team_alive(env) && env->tick < env->max_ticks; t++) {
        env_tick(env, actions);
    }
    env->decision++;
    if (!team_alive(env)) {
        env->terminated = 1;
    } else if (env->tick >= env->max_ticks) {
        env->truncated = 1;
    }
}

RewardConfig reward_config_default(void) {
    RewardConfig cfg = {
        .kill_points = 0.125f, /* +0.25 / +0.5 / +1.0 for 2 / 4 / 8 upstream points */
        .hit = 0.02f,
        .shot = -0.05f, /* a blind shot (18% land) loses reward; an aimed one (63% land) still pays */
        .damage = -0.01f,
        .death = -2.0f,
        .escape = -0.05f,
        .ally_share = 0.0f,
        .align = 0.5f,
        .gamma = 0.99f,
        .edge = -0.003f,
        .bomb = -0.04f, /* explosions ("fire bombs") then cost -0.05 per HP in total */
    };
    return cfg;
}

void env_rewards(Env *env, const RewardConfig *cfg, float rewards[NUM_SHIPS]) {
    for (int i = 0; i < NUM_SHIPS; i++) {
        const ShipEvents *ev = &env->events[i];
        const Ship *s = &env->ships[i];
        /* every ending is terminal, including the 120 s limit (time left is observed), so Phi' = 0 there */
        float phi = env->terminated || env->truncated ? 0.0f : env_potential(env, i);
        RewardParts part = {
            .kill = cfg->kill_points * (float)ev->kill_points,
            .hit = cfg->hit * (float)ev->hits,
            .shot = cfg->shot * (float)ev->shots,
            .damage = cfg->damage * ev->health_lost,
            .death = cfg->death * (float)ev->died,
            .escape = cfg->escape * (float)ev->escapes,
            .ally = cfg->ally_share * cfg->kill_points * (float)ev->ally_kill_points,
            .align = cfg->align * (cfg->gamma * phi - env->potential[i]),
            .edge = cfg->edge * (float)near_edge(s),
            .bomb = cfg->bomb * ev->bomb_health_lost,
        };
        env->potential[i] = phi;
        float r = part.kill + part.hit + part.shot + part.damage + part.death + part.escape + part.ally + part.align +
                  part.edge + part.bomb;
        RewardParts *sum = &env->stats.parts[i];
        sum->kill += part.kill;
        sum->hit += part.hit;
        sum->shot += part.shot;
        sum->damage += part.damage;
        sum->death += part.death;
        sum->escape += part.escape;
        sum->ally += part.ally;
        sum->align += part.align;
        sum->edge += part.edge;
        sum->bomb += part.bomb;
        env->stats.episode_return[i] += r;
        rewards[i] = r;
    }
}

static float shot_quality(float offset) {
    return offset < SHOT_RAMP ? 1.0f - offset / SHOT_RAMP : 0.0f;
}

static float unit(float v) {
    return clampf(v, -1.0f, 1.0f);
}

/* Indices of the k nearest of n distances, nearest first, lower index winning ties. */
static int nearest(const float *dist, int n, int k, int *out) {
    unsigned char used[MAX_ENEMIES > MAX_MISSILES ? MAX_ENEMIES : MAX_MISSILES] = {0};
    int count = n < k ? n : k;
    for (int s = 0; s < count; s++) {
        int best = -1;
        for (int j = 0; j < n; j++) {
            if (!used[j] && (best < 0 || dist[j] < dist[best])) {
                best = j;
            }
        }
        used[best] = 1;
        out[s] = best;
    }
    return count;
}

/* The enemy indices in ship's observation slots, nearest first (as env_observe fills them). */
static int observed_enemies(const Env *env, int ship, int idx[OBS_ENEMY_SLOTS]) {
    const Ship *self = &env->ships[ship];
    float dist[MAX_ENEMIES];
    for (int k = 0; k < env->n_enemies; k++) {
        const Enemy *e = &env->enemies[k];
        dist[k] = centre_distance(self->x, self->y, SHIP_RADIUS, e->x, e->y, e->radius);
    }
    return nearest(dist, env->n_enemies, OBS_ENEMY_SLOTS, idx);
}

void env_observe(const Env *env, int ship, float *obs) {
    memset(obs, 0, OBS_DIM * sizeof *obs);
    const Ship *self = &env->ships[ship];
    float cx = self->x + SHIP_RADIUS;
    float cy = self->y + SHIP_RADIUS;

    obs[0] = cx / ARENA_SIZE;
    obs[1] = cy / ARENA_SIZE;
    obs[2] = self->health / SHIP_HEALTH;
    obs[3] = unit((float)self->weapon_cd / WEAPON_COOLDOWN_TICKS);
    obs[4] = (float)self->ammo / SHIP_AMMO;
    obs[5] = (float)self->alive;
    obs[6] = unit(1.0f - (float)env->tick / (float)env->max_ticks);

    for (int j = 1; j < NUM_SHIPS; j++) {
        const Ship *ally = &env->ships[(ship + j) % NUM_SHIPS];
        float *a = obs + OBS_ALLY_OFFSET + (j - 1) * OBS_ALLY;
        if (!ally->alive) {
            continue;
        }
        a[0] = 1.0f;
        a[1] = unit((ally->x + SHIP_RADIUS - cx) / ARENA_SIZE);
        a[2] = unit((ally->y + SHIP_RADIUS - cy) / ARENA_SIZE);
        a[3] = ally->health / SHIP_HEALTH;
        a[4] = unit((float)ally->weapon_cd / WEAPON_COOLDOWN_TICKS);
        a[5] = (float)ally->ammo / SHIP_AMMO;
    }

    float dist[MAX_ENEMIES > MAX_MISSILES ? MAX_ENEMIES : MAX_MISSILES];
    int idx[OBS_ENEMY_SLOTS > OBS_MISSILE_SLOTS ? OBS_ENEMY_SLOTS : OBS_MISSILE_SLOTS];

    float offset[MAX_ENEMIES];
    int would_hit = 0;
    for (int k = 0; k < env->n_enemies; k++) {
        int hit;
        offset[k] = env_fire_solution(env, ship, k, &hit);
        would_hit |= hit;
    }
    obs[7] = (float)would_hit;
    int ne = observed_enemies(env, ship, idx);
    for (int s = 0; s < ne; s++) {
        const Enemy *e = &env->enemies[idx[s]];
        float *f = obs + OBS_ENEMY_OFFSET + s * OBS_ENEMY;
        f[0] = 1.0f;
        f[1] = unit((e->x + e->radius - cx) / ARENA_SIZE);
        f[2] = unit((e->y + e->radius - cy) / ARENA_SIZE);
        f[3] = unit(e->vx / 800.0f);
        f[4] = unit(e->vy / 1000.0f);
        f[5] = clampf(e->health / ENEMY_TYPES[2].health, 0.0f, 1.0f);
        f[6] = (float)e->dead;
        f[7 + e->type] = 1.0f;
        f[10] = e->attack_cd_left <= 0 ? 1.0f : 0.0f;
        f[11] = shot_quality(offset[idx[s]]);
    }

    for (int m = 0; m < env->n_missiles; m++) {
        const Missile *mi = &env->missiles[m];
        dist[m] = centre_distance(self->x, self->y, SHIP_RADIUS, mi->x, mi->y, MISSILE_RADIUS);
    }
    int nm = nearest(dist, env->n_missiles, OBS_MISSILE_SLOTS, idx);
    for (int s = 0; s < nm; s++) {
        const Missile *mi = &env->missiles[idx[s]];
        float *f = obs + OBS_MISSILE_OFFSET + s * OBS_MISSILE;
        f[0] = 1.0f;
        f[1] = unit((mi->x + MISSILE_RADIUS - cx) / ARENA_SIZE);
        f[2] = unit((mi->y + MISSILE_RADIUS - cy) / ARENA_SIZE);
        f[3] = mi->owner == ship ? 1.0f : 0.0f;
    }
}

static uint64_t fnv_float(uint64_t h, float v) {
    return fnv1a(h, &v, sizeof v);
}

static uint64_t fnv_int(uint64_t h, int64_t v) {
    return fnv1a(h, &v, sizeof v);
}

uint64_t env_hash(const Env *env) {
    uint64_t h = FNV_OFFSET;
    h = fnv_int(h, (int64_t)env->rng.state);
    h = fnv_int(h, env->tick);
    h = fnv_int(h, env->spawn_timer);
    for (int i = 0; i < NUM_SHIPS; i++) {
        const Ship *s = &env->ships[i];
        h = fnv_float(fnv_float(fnv_float(h, s->x), s->y), s->health);
        h = fnv_int(fnv_int(fnv_int(h, s->alive), s->ammo), s->weapon_cd);
        h = fnv_int(h, env->stats.score[i]);
    }
    h = fnv_int(h, env->n_enemies);
    for (int k = 0; k < env->n_enemies; k++) {
        const Enemy *e = &env->enemies[k];
        h = fnv_float(fnv_float(fnv_float(fnv_float(h, e->x), e->y), e->vx), e->vy);
        h = fnv_float(fnv_float(h, e->health), e->radius);
        h = fnv_int(fnv_int(fnv_int(fnv_int(h, e->type), e->dead), e->attack_cd_left), e->id);
    }
    h = fnv_int(h, env->n_missiles);
    for (int m = 0; m < env->n_missiles; m++) {
        const Missile *mi = &env->missiles[m];
        h = fnv_float(fnv_float(fnv_float(h, mi->x), mi->y), mi->vy);
        h = fnv_int(fnv_int(h, mi->owner), mi->id);
    }
    return h;
}

#define FAIL(...)                                  \
    do {                                           \
        snprintf(why, why_size, __VA_ARGS__);      \
        return -1;                                 \
    } while (0)

int env_check_invariants(const Env *env, char *why, size_t why_size) {
    if (env->n_enemies < 0 || env->n_enemies > MAX_ENEMIES || env->n_missiles < 0 || env->n_missiles > MAX_MISSILES) {
        FAIL("capacity: %d enemies, %d missiles", env->n_enemies, env->n_missiles);
    }
    if (env->stats.dropped_spawns || env->stats.dropped_shots) {
        FAIL("capacity: dropped %d spawns, %d shots", env->stats.dropped_spawns, env->stats.dropped_shots);
    }
    const float edge = ARENA_SIZE - 2.0f * SHIP_RADIUS;
    int kills = 0;
    for (int i = 0; i < NUM_SHIPS; i++) {
        const Ship *s = &env->ships[i];
        if (!isfinite(s->x) || !isfinite(s->y) || s->x < 0.0f || s->x > edge || s->y < 0.0f || s->y > edge) {
            FAIL("ship %d position (%g, %g)", i, s->x, s->y);
        }
        if (!(s->health >= 0.0f && s->health <= SHIP_HEALTH) || s->alive != (s->health > 0.0f)) {
            FAIL("ship %d health %g alive %d", i, s->health, s->alive);
        }
        if (s->ammo < 0 || s->ammo > SHIP_AMMO || env->stats.shots[i] != SHIP_AMMO - s->ammo) {
            FAIL("ship %d ammo %d after %d shots", i, s->ammo, env->stats.shots[i]);
        }
        if (s->weapon_cd < 0 || s->weapon_cd > WEAPON_COOLDOWN_TICKS) {
            FAIL("ship %d weapon cooldown %d", i, s->weapon_cd);
        }
        kills += env->stats.kills[i];
    }
    int live = 0;
    for (int k = 0; k < env->n_enemies; k++) {
        const Enemy *e = &env->enemies[k];
        if (!isfinite(e->x) || !isfinite(e->y) || e->x < 0.0f || e->x > ARENA_SIZE - 2.0f * e->radius ||
            e->y > ARENA_SIZE) {
            FAIL("enemy %u position (%g, %g)", e->id, e->x, e->y);
        }
        if (e->radius != (e->dead ? EXPLOSION_RADIUS : ENEMY_RADIUS) || e->type < 0 || e->type > 2) {
            FAIL("enemy %u radius %g type %d", e->id, e->radius, e->type);
        }
        live += !e->dead;
    }
    for (int m = 0; m < env->n_missiles; m++) {
        const Missile *mi = &env->missiles[m];
        if (!isfinite(mi->x) || !isfinite(mi->y) || mi->owner < 0 || mi->owner >= NUM_SHIPS) {
            FAIL("missile %u state", mi->id);
        }
    }
    if (env->stats.spawned != kills + env->stats.escapes + live) {
        FAIL("accounting: spawned %d != kills %d + escapes %d + live %d", env->stats.spawned, kills,
             env->stats.escapes, live);
    }
    return 0;
}

float env_fire_solution(const Env *env, int ship, int enemy, int *hit) {
    const Ship *s = &env->ships[ship];
    if (hit) {
        *hit = 0;
    }
    if (!s->alive || env->enemies[enemy].dead) {
        return FIRE_NO_SOLUTION;
    }
    /* Replay the tick order of env_tick for one missile and one enemy: check, then move both. */
    Missile m = {.x = s->x, .y = s->y, .vy = MISSILE_START_VY, .owner = ship};
    Enemy e = env->enemies[enemy];
    const float reach = MISSILE_RADIUS + e.radius;
    float closest = FIRE_NO_SOLUTION, offset = FIRE_NO_SOLUTION;
    for (int t = 0; t < 4 * TICK_HZ; t++) {
        float d = centre_distance(m.x, m.y, MISSILE_RADIUS, e.x, e.y, e.radius);
        if (d < closest) {
            closest = d;
            offset = fabsf((e.x + e.radius) - (m.x + MISSILE_RADIUS));
        }
        if (d < reach) {
            if (hit) {
                *hit = 1;
            }
            break;
        }
        float past = (e.y + e.radius) - (m.y + MISSILE_RADIUS); /* > 0: the missile is above the enemy */
        if (t == 0 && past > reach) {
            return FIRE_NO_SOLUTION; /* the enemy is below the ship: no shot */
        }
        if (m.y < -MISSILE_RADIUS || past > reach) {
            break; /* missile gone, or the gap can only grow from here */
        }
        step_enemy(&e);
        step_missile(&m);
        if (e.y > ARENA_SIZE) {
            break; /* escaped */
        }
    }
    return offset;
}

float env_shot_quality(const Env *env, int ship, int *would_hit) {
    float best = 0.0f;
    *would_hit = 0;
    for (int k = 0; k < env->n_enemies; k++) {
        int hit = 0;
        float q = shot_quality(env_fire_solution(env, ship, k, &hit));
        best = q > best ? q : best;
        *would_hit |= hit;
    }
    return best;
}

float env_potential(const Env *env, int ship) {
    const Ship *s = &env->ships[ship];
    if (!s->alive || s->ammo <= 0) {
        return 0.0f;
    }
    int idx[OBS_ENEMY_SLOTS];
    int n = observed_enemies(env, ship, idx);
    float best = 0.0f;
    for (int k = 0; k < n; k++) {
        float q = shot_quality(env_fire_solution(env, ship, idx[k], NULL));
        best = q > best ? q : best;
    }
    return best;
}
