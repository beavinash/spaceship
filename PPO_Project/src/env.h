#ifndef ENV_H
#define ENV_H

#include <stddef.h>
#include <stdint.h>

#include "rng.h"

/*
 * Headless port of the upstream Spaceship game (Drjakll/spaceship main e683c7d:
 * init.c, Headers/objects.h, Headers/detections.h) at a fixed 1/120 s tick.
 * Positions are top-left corners; an entity's centre is (x + radius, y + radius).
 * Upstream cooldowns in seconds become whole ticks so float drift cannot move them.
 */

#define ARENA_SIZE 1000.0f
#define NUM_SHIPS 3
#define TICK_HZ 120
#define TICK_DT (1.0f / 120.0f)
#define ACTION_REPEAT 4

#define SHIP_RADIUS 25.0f
#define SHIP_HEALTH 200.0f /* user-required; upstream SPACESHIP_HEALTH is 1000 */
#define SHIP_AMMO 100      /* user-required; upstream ammo is unlimited */
#define SHIP_SPEED 400.0f
#define SHIP_START_X 400.0f
#define SHIP_START_SPACING 50.0f
#define SHIP_START_Y 700.0f
#define WEAPON_COOLDOWN_TICKS 30 /* 0.25 s */

#define MISSILE_RADIUS 10.0f
#define MISSILE_DAMAGE 10.0f
#define MISSILE_START_VY 30.0f
#define MISSILE_ACCEL 300.0f

#define ENEMY_RADIUS 25.0f
#define ENEMY_START_VX 200.0f
#define ENEMY_START_VY 10.0f
#define ENEMY_SPAWN_TICKS 90 /* 0.75 s */
#define ENEMY_SPAWN_MAX_X 800
#define ZIGZAG_PERIOD 350
#define ZIGZAG_WINDOW 10

#define EXPLOSION_RADIUS 50.0f
#define EXPLOSION_DAMAGE 1.0f
#define EXPLOSION_COOLDOWN_TICKS 18 /* 0.15 s */
#define EXPLOSION_VX 1.0f
#define EXPLOSION_VY 100.0f
#define EXPLOSION_ACCEL (-1.0f)

#define MAX_ENEMIES 64
#define MAX_MISSILES 64
#define EPISODE_SECONDS 120
#define EPISODE_TICKS (EPISODE_SECONDS * TICK_HZ)

/* Observation layout (section 3.2): self, two allies, 8 nearest enemies/explosions, 8 nearest missiles. */
#define OBS_SELF 8
#define OBS_ALLY 6
#define OBS_ENEMY 12
#define OBS_MISSILE 4
#define OBS_ENEMY_SLOTS 8
#define OBS_MISSILE_SLOTS 8
#define OBS_ALLY_OFFSET OBS_SELF
#define OBS_ENEMY_OFFSET (OBS_ALLY_OFFSET + (NUM_SHIPS - 1) * OBS_ALLY)
#define OBS_MISSILE_OFFSET (OBS_ENEMY_OFFSET + OBS_ENEMY_SLOTS * OBS_ENEMY)
#define OBS_DIM (OBS_MISSILE_OFFSET + OBS_MISSILE_SLOTS * OBS_MISSILE)

typedef struct {
    float health;
    float damage;
    float accel;
    int points;
    int attack_cooldown_ticks;
} EnemyType;

extern const EnemyType ENEMY_TYPES[3];

enum { H_LEFT, H_STAY, H_RIGHT };
enum { V_UP, V_STAY, V_DOWN };

typedef struct {
    int h;    /* H_LEFT, H_STAY, H_RIGHT */
    int v;    /* V_UP, V_STAY, V_DOWN */
    int fire; /* 0 or 1 */
} ShipAction;

typedef struct {
    float x, y;
    float health;
    int alive;
    int ammo;
    int weapon_cd; /* ticks until the weapon may fire; ready when <= 0 */
} Ship;

typedef struct {
    float x, y;
    float vx, vy;
    float accel;
    float health;
    float damage;
    float radius;
    int type;
    int dead; /* destroyed enemies become drifting explosions */
    int points;
    int attack_cd_ticks;
    int attack_cd_left; /* contact damage allowed when <= 0 */
    uint32_t id;
} Enemy;

typedef struct {
    float x, y;
    float vy; /* upward speed */
    int owner;
    uint32_t id;
} Missile;

/* What happened to one ship during the current decision. */
typedef struct {
    int kills;
    int kill_points;
    int ally_kill_points;
    int hits;
    int shots;
    float health_lost;
    int died;
    int escapes; /* live enemies that escaped while this ship was alive */
} ShipEvents;

typedef struct {
    float kill_points;
    float hit;
    float shot;
    float damage;
    float death;
    float escape;
    float ally_share;
    float align; /* potential-based shaping weight on the best shot quality */
    float gamma; /* shaping discount; must equal the learner's gamma */
    float edge;  /* per decision with the ship's centre inside EDGE_BAND of a side wall or the bottom */
} RewardConfig;

#define EDGE_BAND 60.0f

/* Reward components summed over an episode, per ship. */
typedef struct {
    float kill;
    float hit;
    float shot;
    float damage;
    float death;
    float escape;
    float ally;
    float align;
    float edge;
} RewardParts;

typedef struct {
    int score[NUM_SHIPS]; /* upstream points */
    int kills[NUM_SHIPS];
    int hits[NUM_SHIPS];
    int shots[NUM_SHIPS];
    float damage_taken[NUM_SHIPS];
    int died[NUM_SHIPS];
    int alive_ticks[NUM_SHIPS];
    int edge_ticks[NUM_SHIPS];  /* alive ticks with the ship's centre inside EDGE_BAND */
    int aimed_shots[NUM_SHIPS]; /* shots whose fire solution predicted a hit at launch */
    float episode_return[NUM_SHIPS];
    RewardParts parts[NUM_SHIPS];
    int spawned;
    int escapes;
    int dropped_spawns; /* spawns skipped because every enemy slot was taken */
    int dropped_shots;  /* shots skipped because every missile slot was taken */
} EpisodeStats;

typedef struct {
    Pcg32 rng;
    uint64_t seed;
    Ship ships[NUM_SHIPS];
    Enemy enemies[MAX_ENEMIES]; /* kept in spawn order, like upstream's list */
    int n_enemies;
    Missile missiles[MAX_MISSILES]; /* kept in launch order */
    int n_missiles;
    int tick;
    int decision;
    int spawn_timer;
    int max_ticks; /* truncation limit; counted in ticks so any action repeat gets 120 s */
    int action_repeat;
    uint32_t next_enemy_id;
    uint32_t next_missile_id;
    ShipEvents events[NUM_SHIPS];
    float potential[NUM_SHIPS]; /* shaping potential after the last env_rewards call */
    EpisodeStats stats;
    int terminated;
    int truncated;
} Env;

void env_reset(Env *env, uint64_t seed);

/* Append an enemy of the given type at a top-left position; NULL if every slot is taken. */
Enemy *env_add_enemy(Env *env, int type, float x, float y);

/* Advance one physics tick, in upstream frame order, with each ship holding its action. */
void env_tick(Env *env, const ShipAction actions[NUM_SHIPS]);

/*
 * One policy decision: clear the per-decision events, then hold the actions for action_repeat ticks.
 * Sets terminated when every ship is dead (checked first), otherwise truncated at max_ticks.
 * A finished episode does not advance until env_reset.
 */
void env_step(Env *env, const ShipAction actions[NUM_SHIPS]);

/*
 * Ego-centric observation for one ship, OBS_DIM floats in [-1, 1]. Relative values use centres;
 * entity slots hold the nearest entities first and are zero-padded. Reads state only.
 */
void env_observe(const Env *env, int ship, float *obs);

/*
 * Consistency checks for stress runs: counts within capacity, finite positions inside the arena,
 * health/ammo/cooldown ranges, shots == SHIP_AMMO - ammo, no dropped spawns or shots, and the
 * accounting identity spawned == kills + escapes + live enemies in the arena. Returns 0, or -1 with
 * a reason in why.
 */
int env_check_invariants(const Env *env, char *why, size_t why_size);

/*
 * Fire solution for a missile launched now from `ship` at live enemy `enemy`, replaying the exact
 * tick rules for that missile and enemy alone (other missiles ignored, ship assumed still).
 * Returns the horizontal offset between their centres at contact, or at the closest pass if it
 * misses (0 = dead centre); sets *hit (if not NULL). FIRE_NO_SOLUTION for a dead ship or explosion.
 */
#define FIRE_NO_SOLUTION 1e9f
float env_fire_solution(const Env *env, int ship, int enemy, int *hit);

/* Shot quality falls linearly from 1 (dead centre) to 0 at this horizontal offset. */
#define SHOT_RAMP 70.0f

/* Best shot quality for `ship` over all live enemies, in [0, 1]; sets *would_hit if any shot would hit. */
float env_shot_quality(const Env *env, int ship, int *would_hit);

/*
 * Shaping potential: the best shot quality over the enemies in the ship's observation slots, so it
 * equals the largest per-slot shot-quality input. 0 for a dead ship or one without ammo.
 */
float env_potential(const Env *env, int ship);

/* FNV-1a over gameplay state and RNG position, for replay and determinism checks. */
uint64_t env_hash(const Env *env);

/* Section 3.2 defaults: score + survival with a per-shot cost; ally sharing off. */
RewardConfig reward_config_default(void);

/*
 * Per-ship reward for the last decision: a linear combination of that ship's events plus
 * potential-based alignment shaping align * (gamma * Phi(s') - Phi(s)), with Phi = env_potential
 * and Phi(s') = 0 once the episode has ended (any ending, the 120 s limit included), and the edge
 * penalty. Call once after every env_step.
 * Also adds each component to stats.parts and the total to stats.episode_return.
 */
void env_rewards(Env *env, const RewardConfig *cfg, float rewards[NUM_SHIPS]);

#endif
