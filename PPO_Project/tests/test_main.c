#include <stdio.h>
#include <string.h>

#include "check.h"

int check_total = 0;
int check_failed = 0;

#define TEST(name) void test_##name(void);
#include "tests.def"
#undef TEST

typedef struct {
    const char *name;
    void (*fn)(void);
} TestCase;

static const TestCase cases[] = {
#define TEST(name) {#name, test_##name},
#include "tests.def"
#undef TEST
};

int main(int argc, char **argv) {
    const char *filter = argc > 1 ? argv[1] : NULL;
    int ran = 0;
    int failed_tests = 0;

    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        if (filter && !strstr(cases[i].name, filter)) {
            continue;
        }
        int before = check_failed;
        cases[i].fn();
        ran++;
        int ok = check_failed == before;
        failed_tests += !ok;
        printf("%s %s\n", ok ? "PASS" : "FAIL", cases[i].name);
    }

    printf("%d tests, %d checks, %d failed checks, %d failed tests\n", ran, check_total, check_failed,
           failed_tests);
    return (failed_tests || ran == 0) ? 1 : 0;
}
