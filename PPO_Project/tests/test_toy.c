#include "check.h"
#include "toy.h"

void test_toy_rules(void) {
    ToyEnv e;
    toy_reset(&e, 3u);
    e.px = 0.0f;
    e.py = 0.0f;
    e.tx = 0.3f;
    e.ty = -0.1f;
    int terminated = 0, truncated = 0;
    const int right_up_hold[3] = {2, 0, 0};
    float r = toy_step(&e, right_up_hold, &terminated, &truncated);
    CHECK_NEAR(r, -0.01 + (sqrt(0.1) - 0.2), 1e-6); /* step cost plus distance closed */
    CHECK_NEAR(e.px, 0.1, 1e-6);
    CHECK_NEAR(e.py, -0.1, 1e-6);
    const int fire_far[3] = {1, 1, 1};
    r = toy_step(&e, fire_far, &terminated, &truncated);
    CHECK_NEAR(r, -0.06, 1e-6); /* 0.2 away: a wasted shot */
    CHECK(!terminated);
    const int right[3] = {2, 1, 0};
    toy_step(&e, right, &terminated, &truncated);
    r = toy_step(&e, fire_far, &terminated, &truncated);
    CHECK_NEAR(r, 0.99, 1e-6);
    CHECK(terminated && !truncated);

    float obs[TOY_OBS];
    toy_reset(&e, 4u);
    toy_observe(&e, obs);
    CHECK_NEAR(obs[2], e.tx - e.px, 1e-6);
    for (int t = 0; t < TOY_MAX_STEPS; t++) {
        const int stay[3] = {1, 1, 0};
        toy_step(&e, stay, &terminated, &truncated);
    }
    CHECK(truncated && !terminated);
}

/*
 * Learner sanity: within 150 updates (seed 1), at least 90% of the last 100 episodes must end in a
 * hit with a mean return of at least 1.2. Walking to the target without firing scores about 0.5,
 * so passing requires learning both movement and the timed fire head.
 */
void test_toy_ppo_learns_target_reaching(void) {
    ToyResult res;
    double random_return = toy_random_return(1u, 400);
    double learned = toy_train(1u, 150, &res);
    printf("  toy: random %.3f, PPO %.3f, hit rate %.2f after %d updates, %d episodes\n", random_return, learned,
           res.hit_rate, res.updates, res.episodes);
    CHECK(random_return < 0.0);
    CHECK(res.hit_rate >= 0.9);
    CHECK(learned >= 1.2);
    CHECK(res.updates == 150);
}
