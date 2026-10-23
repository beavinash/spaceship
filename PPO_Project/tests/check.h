#ifndef CHECK_H
#define CHECK_H

#include <math.h>
#include <stdio.h>

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

#endif
