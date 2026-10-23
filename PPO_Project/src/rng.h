#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* PCG32 (XSH-RR) from O'Neill's pcg-c-basic: small, fast, seedable streams. */
typedef struct {
    uint64_t state;
    uint64_t inc;
} Pcg32;

static inline uint32_t pcg32_next(Pcg32 *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

static inline void pcg32_seed(Pcg32 *rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0u;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_next(rng);
    rng->state += initstate;
    pcg32_next(rng);
}

/* Unbiased integer in [0, bound) by rejection. */
static inline uint32_t pcg32_bounded(Pcg32 *rng, uint32_t bound) {
    uint32_t threshold = (uint32_t)(-bound) % bound;
    for (;;) {
        uint32_t r = pcg32_next(rng);
        if (r >= threshold) {
            return r % bound;
        }
    }
}

/* Float in [0, 1) with 24 random bits. */
static inline float pcg32_float(Pcg32 *rng) {
    return (float)(pcg32_next(rng) >> 8) * (1.0f / 16777216.0f);
}

#endif
