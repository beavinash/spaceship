#include <stdint.h>

#include "check.h"
#include "rng.h"

/* First outputs of the reference pcg32-demo for pcg32_srandom(42, 54). */
void test_pcg32_reference_sequence(void) {
    Pcg32 rng;
    pcg32_seed(&rng, 42u, 54u);
    const uint32_t expected[6] = {0xa15c02b7u, 0x7b47f409u, 0xba1d3330u,
                                  0x83d2f293u, 0xbfa4784bu, 0xcbed606eu};
    for (int i = 0; i < 6; i++) {
        CHECK(pcg32_next(&rng) == expected[i]);
    }
}

void test_pcg32_bounded_and_unit_float(void) {
    Pcg32 rng;
    pcg32_seed(&rng, 7u, 1u);
    int counts[3] = {0, 0, 0};
    int in_range = 1;
    for (int i = 0; i < 30000; i++) {
        uint32_t v = pcg32_bounded(&rng, 3);
        in_range &= v < 3;
        counts[v < 3 ? v : 0]++;
    }
    CHECK(in_range);
    for (int k = 0; k < 3; k++) {
        CHECK(counts[k] > 9500 && counts[k] < 10500);
    }

    double sum = 0.0;
    int unit = 1;
    for (int i = 0; i < 10000; i++) {
        float f = pcg32_float(&rng);
        unit &= f >= 0.0f && f < 1.0f;
        sum += f;
    }
    CHECK(unit);
    CHECK_NEAR(sum / 10000.0, 0.5, 0.02);

    Pcg32 a, b;
    pcg32_seed(&a, 99u, 3u);
    pcg32_seed(&b, 99u, 3u);
    int same = 1;
    for (int i = 0; i < 100; i++) {
        same &= pcg32_next(&a) == pcg32_next(&b);
    }
    CHECK(same);
}
