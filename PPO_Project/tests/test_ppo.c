#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "dist.h"
#include "ppo.h"

void test_ppo_rollout_masks_steps_after_death(void) {
    Rollout r;
    CHECK(rollout_init(&r, 8, 2, 3) == 0);
    const float obs[3] = {0.1f, 0.2f, 0.3f};
    const int action[ACTION_HEADS] = {2, 0, 1};
    /* Env 0: alive for steps 0-2, dies on step 2, waits out steps 3-5, a new episode from step 6. */
    for (int t = 0; t < 8; t++) {
        int active = t < 3 || t >= 6;
        rollout_record(&r, t, 0, active, obs, action, -1.5f, 0.75f);
        rollout_outcome(&r, t, 0, active ? 0.5f : 9.0f, t == 2, 0.0f);
        rollout_record(&r, t, 1, 1, obs, action, -1.0f, 0.25f);
        rollout_outcome(&r, t, 1, 1.0f, 0, 0.0f);
    }
    int ok_active = 1, ok_done = 1, ok_zero = 1;
    for (int t = 0; t < 8; t++) {
        int i = rollout_index(&r, t, 0);
        int expect_active = t < 3 || t >= 6;
        ok_active &= r.active[i] == expect_active;
        ok_done &= r.dones[i] == (t >= 2 && t <= 5);
        if (!expect_active) {
            ok_zero &= r.rewards[i] == 0.0f && r.values[i] == 0.0f && r.logp[i] == 0.0f;
            ok_zero &= r.obs[(size_t)i * 3] == 0.0f && r.bootstrap[i] == 0.0f;
        }
    }
    CHECK(ok_active);
    CHECK(ok_done);
    CHECK(ok_zero);
    int i = rollout_index(&r, 1, 0);
    CHECK(r.actions[i * ACTION_HEADS + 0] == 2 && r.actions[i * ACTION_HEADS + 2] == 1);
    CHECK_NEAR(r.obs[(size_t)i * 3 + 2], 0.3f, 0.0);
    CHECK_NEAR(r.logp[i], -1.5, 0.0);
    CHECK_NEAR(r.values[i], 0.75, 0.0);
    CHECK_NEAR(r.rewards[i], 0.5, 0.0);
    CHECK(rollout_active_count(&r) == 5 + 8);
    rollout_free(&r);
}

void test_ppo_gae_handles_termination_truncation_masks(void) {
    Rollout r;
    CHECK(rollout_init(&r, 4, 2, 1) == 0);
    const float obs[1] = {0.0f};
    const int action[ACTION_HEADS] = {1, 1, 0};
    /* Env 0: time-limit ending after step 1 (bootstrap 5), continuing at the rollout edge (V = 10). */
    const float v0[4] = {1, 2, 3, 4}, r0[4] = {1, 0, 2, 1};
    /* Env 1: dies after step 1, masked for steps 2-3. */
    const float v1[4] = {0.5f, 1.5f, 0, 0}, r1[4] = {1, -2, 0, 0};
    for (int t = 0; t < 4; t++) {
        rollout_record(&r, t, 0, 1, obs, action, 0.0f, v0[t]);
        rollout_outcome(&r, t, 0, r0[t], t == 1, t == 1 ? 5.0f : 0.0f);
        rollout_record(&r, t, 1, t < 2, obs, action, 0.0f, v1[t]);
        rollout_outcome(&r, t, 1, r1[t], t == 1, 0.0f);
    }
    const float last_values[2] = {10.0f, 99.0f};
    rollout_gae(&r, last_values, 0.9f, 0.8f);
    /* Hand-computed with gamma 0.9, lambda 0.8 (gamma * lambda = 0.72). */
    const double adv0[4] = {3.6, 2.5, 6.92, 6.0};
    const double adv1[4] = {-0.67, -3.5, 0.0, 0.0};
    for (int t = 0; t < 4; t++) {
        CHECK_NEAR(r.advantages[rollout_index(&r, t, 0)], adv0[t], 1e-5);
        CHECK_NEAR(r.returns[rollout_index(&r, t, 0)], adv0[t] + v0[t], 1e-5);
        CHECK_NEAR(r.advantages[rollout_index(&r, t, 1)], adv1[t], 1e-5);
    }
    CHECK_NEAR(r.returns[rollout_index(&r, 0, 1)], -0.17, 1e-5);
    CHECK_NEAR(r.returns[rollout_index(&r, 3, 1)], 0.0, 0.0);
    rollout_free(&r);
}
