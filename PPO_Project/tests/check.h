#ifndef CHECK_H
#define CHECK_H

#include <math.h>
#include <stdio.h>
#include <string.h>

extern int check_total;
extern int check_failed;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        check_total++;                                                               \
        if (!(cond)) {                                                               \
            check_failed++;                                                          \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
        }                                                                            \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                          \
    do {                                                                               \
        double check_a_ = (double)(a);                                                 \
        double check_b_ = (double)(b);                                                 \
        check_total++;                                                                 \
        if (!(fabs(check_a_ - check_b_) <= (double)(tol))) {                           \
            check_failed++;                                                            \
            fprintf(stderr, "%s:%d: CHECK_NEAR(%s, %s) failed: %.9g vs %.9g\n",        \
                    __FILE__, __LINE__, #a, #b, check_a_, check_b_);                   \
        }                                                                              \
    } while (0)

/* 1 if the file at path contains needle (first 1 MB). */
static inline int file_contains(const char *path, const char *needle) {
    static char buf[1 << 20];
    FILE *f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

#endif
