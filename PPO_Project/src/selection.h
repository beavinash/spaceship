#ifndef SELECTION_H
#define SELECTION_H

#include <stdint.h>

/* A checkpoint scored on the development scenarios (sampled PPO, seeded per scenario). */
typedef struct {
    char path[512];
    uint32_t update;
    double mean_score;  /* mean team score */
    double mean_deaths; /* mean ships lost per episode */
} Candidate;

/* Best candidate: highest mean team score, then fewer deaths, then earlier update, then listing order. -1 if n == 0. */
int select_best(const Candidate *c, int n);

/*
 * Scores every .bin in run_dir/checkpoints (and final.bin unless its update was already saved) on the dev
 * split, writes run_dir/selection.json, and copies the winner to run_dir/selected.bin.
 * Fills out[0..*n) in update order; returns the winner's index or -1.
 */
int select_run(const char *run_dir, Candidate *out, int cap, int *n);

#endif
