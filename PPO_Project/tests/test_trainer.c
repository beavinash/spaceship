#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "trainer.h"

static TrainConfig small_config(void) {
    TrainConfig cfg = train_config_default();
    cfg.envs = 4;
    cfg.steps = 16;
    cfg.hidden = 32;
    cfg.seed = 7u;
    return cfg;
}

void test_trainer_collects_per_ship_rollouts(void) {
    TrainConfig cfg = small_config();
    Trainer tr;
    CHECK(trainer_init(&tr, &cfg) == 0);
    CHECK(memcmp(tr.agents[0].actor.params, tr.agents[1].actor.params, sizeof(float) * 16) != 0);

    /* Ship 0's own network strongly prefers "left"; no other ship's network does. */
    float *bias = tr.agents[0].actor.layers[2].b;
    bias[0] += 50.0f;
    /* Ship 1 in arena 0 dies on the first decision. */
    Env *e0 = &tr.envs[0];
    e0->ships[1].health = 0.5f;
    Enemy *e = env_add_enemy(e0, 2, e0->ships[1].x, e0->ships[1].y);
    e->vx = e->vy = e->accel = 0.0f;
    for (int k = 0; k < NUM_SHIPS; k++) {
        env_observe(e0, k, tr.obs[k]); /* refresh after editing the arena */
    }
    float first_obs[NUM_SHIPS][OBS_DIM];
    for (int k = 0; k < NUM_SHIPS; k++) {
        memcpy(first_obs[k], tr.obs[k], sizeof first_obs[k]);
    }

    trainer_collect(&tr);
    CHECK(tr.decisions == 16 * 4);
    for (int k = 0; k < NUM_SHIPS; k++) {
        CHECK(tr.rollouts[k].steps == 16 && tr.rollouts[k].envs == 4);
        CHECK(memcmp(tr.rollouts[k].obs, first_obs[k], sizeof first_obs[k]) == 0);
    }
    int ship0_left = 1, ship2_left = 1;
    for (int i = 0; i < 16 * 4; i++) {
        ship0_left &= tr.rollouts[0].actions[i * ACTION_HEADS] == H_LEFT;
        ship2_left &= tr.rollouts[2].actions[i * ACTION_HEADS] == H_LEFT;
    }
    CHECK(ship0_left);
    CHECK(!ship2_left);

    const Rollout *r1 = &tr.rollouts[1];
    CHECK(r1->active[rollout_index(r1, 0, 0)] == 1 && r1->dones[rollout_index(r1, 0, 0)] == 1);
    CHECK_NEAR(r1->rewards[rollout_index(r1, 0, 0)], -2.0f - 0.01f * 0.5f, 1e-5);
    int masked = 1;
    for (int t = 1; t < 16; t++) {
        masked &= r1->active[rollout_index(r1, t, 0)] == 0;
    }
    CHECK(masked);
    CHECK(rollout_active_count(r1) == 64 - 15);
    CHECK(rollout_active_count(&tr.rollouts[0]) == 64);

    /* GAE ran: active advantages are finite and masked ones are zero. */
    int gae_ok = 1;
    for (int i = 0; i < 64; i++) {
        gae_ok &= r1->active[i] ? isfinite(r1->advantages[i]) : r1->advantages[i] == 0.0f;
    }
    CHECK(gae_ok);
    trainer_free(&tr);
}
