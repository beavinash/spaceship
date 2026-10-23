#include <string.h>

#include "check.h"
#include "dist.h"
#include "rng.h"

static const float LOGITS[ACTION_LOGITS] = {1.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, 2.0f, 0.0f};

void test_dist_factorized_logprob_entropy_and_gradients(void) {
    CHECK(ACTION_HEADS == 3 && ACTION_LOGITS == 8);
    CHECK(HEAD_SIZES[0] == 3 && HEAD_SIZES[1] == 3 && HEAD_SIZES[2] == 2);
    const int action[ACTION_HEADS] = {0, 2, 1};
    /* Hand-computed: log softmax per head, summed over heads. */
    CHECK_NEAR(dist_log_prob(LOGITS, action), -0.407606 - 1.098612 - 2.126928, 1e-4);
    CHECK_NEAR(dist_entropy(LOGITS), 0.832397 + 1.098612 + 0.365334, 1e-4);

    float logits[ACTION_LOGITS];
    float g_logp[ACTION_LOGITS], g_ent[ACTION_LOGITS];
    memcpy(logits, LOGITS, sizeof logits);
    dist_log_prob_grad(logits, action, g_logp);
    dist_entropy_grad(logits, g_ent);
    const float eps = 1e-2f;
    double worst = 0.0;
    for (int k = 0; k < ACTION_LOGITS; k++) {
        float keep = logits[k];
        logits[k] = keep + eps;
        double lp_up = dist_log_prob(logits, action), h_up = dist_entropy(logits);
        logits[k] = keep - eps;
        double lp_down = dist_log_prob(logits, action), h_down = dist_entropy(logits);
        logits[k] = keep;
        worst = fmax(worst, fabs((lp_up - lp_down) / (2.0 * eps) - g_logp[k]));
        worst = fmax(worst, fabs((h_up - h_down) / (2.0 * eps) - g_ent[k]));
    }
    CHECK(worst < 1e-3);
    /* d logp / d logits is one-hot minus softmax per head. */
    CHECK_NEAR(g_logp[0], 1.0 - 0.665241, 1e-4);
    CHECK_NEAR(g_logp[7], 1.0 - 0.119203, 1e-4);

    int greedy[ACTION_HEADS];
    dist_argmax(LOGITS, greedy);
    CHECK(greedy[0] == 0 && greedy[1] == 0 && greedy[2] == 0); /* ties go to the lowest index */

    Pcg32 rng;
    pcg32_seed(&rng, 3u, 4u);
    int counts[ACTION_LOGITS] = {0};
    for (int n = 0; n < 20000; n++) {
        int a[ACTION_HEADS];
        dist_sample(LOGITS, &rng, a);
        counts[a[0]]++;
        counts[3 + a[1]]++;
        counts[6 + a[2]]++;
    }
    CHECK_NEAR(counts[0] / 20000.0, 0.665241, 0.015);
    CHECK_NEAR(counts[2] / 20000.0, 0.090031, 0.015);
    CHECK_NEAR(counts[4] / 20000.0, 1.0 / 3.0, 0.015);
    CHECK_NEAR(counts[6] / 20000.0, 0.880797, 0.015);

    /* Large logits stay finite. */
    const float big[ACTION_LOGITS] = {1000.0f, 0.0f, -1000.0f, 0.0f, 0.0f, 0.0f, 90.0f, -90.0f};
    const int pick[ACTION_HEADS] = {2, 1, 1};
    CHECK(isfinite(dist_log_prob(big, pick)) && isfinite(dist_entropy(big)));
}
