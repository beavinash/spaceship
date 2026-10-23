/* The worked examples in REWARD.md, pinned so the document cannot drift from the code. */
#include "check.h"
#include "env.h"

static void quiet(Env *env) {
    env_reset(env, 1u);
    env->spawn_timer = 1 << 30;
}

/* Example 1: ship 0 kills a type-3 enemy straight above it with exactly four missiles. */
void test_reward_doc_single_ship_type3_kill(void) {
    Env env;
    quiet(&env);
    env.ships[1].x = 800.0f; /* ships 1 and 2 stand clear, with no shot at this enemy */
    env.ships[2].x = 850.0f;
    Enemy *e = env_add_enemy(&env, 2, 385.0f, 100.0f); /* type 3: 40 HP, 8 points, centre x 410 */
    e->vx = 0.0f;
    RewardConfig cfg = reward_config_default();
    float r[NUM_SHIPS], first = 0.0f, kill_step = 0.0f;
    int kill_decision = -1, hit_decisions[4], n_hits = 0;
    for (int d = 0; d < 400 && (env.n_enemies > 0 || env.n_missiles > 0); d++) {
        ShipAction a[NUM_SHIPS] = {{H_STAY, V_STAY, env.stats.shots[0] < 4}, {H_STAY, V_STAY, 0}, {H_STAY, V_STAY, 0}};
        env_step(&env, a);
        env_rewards(&env, &cfg, r);
        if (d == 0) {
            first = r[0];
        }
        if (env.events[0].hits && n_hits < 4) {
            hit_decisions[n_hits++] = d;
        }
        if (env.events[0].kills) {
            kill_decision = d;
            kill_step = r[0];
        }
    }
    const RewardParts *p = &env.stats.parts[0];
    CHECK_NEAR(first, -0.01 + 0.5 * 0.99 * 1.0, 1e-5); /* shot cost + potential 0 -> 1 */
    CHECK(n_hits == 4 && hit_decisions[0] == 49 && hit_decisions[3] == 68);
    CHECK(kill_decision == 68);
    CHECK_NEAR(kill_step, 1.0 + 0.02 - 0.5 * 1.0, 1e-4); /* kill + hit, potential 1 -> 0: +0.52 */
    CHECK_NEAR(p->kill, 1.0, 1e-6);
    CHECK_NEAR(p->hit, 0.08, 1e-6);
    CHECK_NEAR(p->shot, -0.04, 1e-6);
    CHECK_NEAR(p->damage, -0.11, 1e-5); /* the wreck drifts down onto the ship: 11 x 1 HP */
    CHECK_NEAR(p->align, 0.495 - 67 * 0.005 - 0.5, 1e-4); /* = -0.34 */
    CHECK_NEAR(env.stats.episode_return[0], 0.59, 1e-4);
    CHECK(env.stats.score[0] == 8 && env.stats.episode_return[1] == 0.0f && env.stats.episode_return[2] == 0.0f);
}

/* Example 2: ships 0 and 1 share a type-2 target while an enemy escapes; ship 2 only watches. */
void test_reward_doc_three_ships_shared_target(void) {
    Env env;
    quiet(&env);
    env.ships[0].x = 400.0f; /* missile centre x 410 */
    env.ships[1].x = 440.0f; /* missile centre x 450 */
    env.ships[2].x = 800.0f;
    Enemy *e = env_add_enemy(&env, 1, 405.0f, 100.0f); /* type 2: 30 HP, 4 points, centre x 430 */
    e->vx = 0.0f;
    Enemy *runner = env_add_enemy(&env, 0, 700.0f, 900.0f); /* about to leave through the bottom */
    runner->vx = 0.0f;
    RewardConfig cfg = reward_config_default();
    float r[NUM_SHIPS], first = 0.0f;
    for (int d = 0; d < 120 && (env.n_missiles > 0 || d < 10); d++) {
        ShipAction a[NUM_SHIPS] = {{H_STAY, V_STAY, env.stats.shots[0] < 2}, {H_STAY, V_STAY, env.stats.shots[1] < 2},
                                   {H_STAY, V_STAY, 0}};
        env_step(&env, a);
        env_rewards(&env, &cfg, r);
        if (d == 0) {
            first = r[0];
        }
    }
    float q = 1.0f - 20.0f / SHOT_RAMP; /* both ships are 20 px off centre */
    CHECK_NEAR(first, -0.01 + 0.5 * 0.99 * q, 1e-5); /* = +0.3436 */
    CHECK(env.stats.escapes == 1);
    CHECK(env.stats.kills[0] == 1 && env.stats.kills[1] == 0); /* same-tick volley: ship 0's missile lands first */
    CHECK(env.stats.hits[0] == 2 && env.stats.hits[1] == 1);
    CHECK_NEAR(env.stats.episode_return[0], 0.2771, 1e-4);
    CHECK_NEAR(env.stats.episode_return[1], -0.2429, 1e-4);
    CHECK_NEAR(env.stats.episode_return[2], -0.05, 1e-6);
}
