#ifndef HASH_H
#define HASH_H

#include <stddef.h>
#include <stdint.h>

/* 64-bit FNV-1a, for checksums and state fingerprints (not cryptographic). */
#define FNV_OFFSET 14695981039346656037ULL

static inline uint64_t fnv1a(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

#endif
