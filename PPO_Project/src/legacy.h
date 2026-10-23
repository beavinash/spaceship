#ifndef LEGACY_H
#define LEGACY_H

#include "env.h"

/*
 * Frozen port of upstream main's learned policy (Headers/ml_utilities.h, self_control.h, init.c):
 * per-ship 40 x 40 grids of softmax probabilities saved in Game_Data/trained{0,1,2}.json, indexed
 * by the ship-minus-target displacement. Inference only; upstream's online updates are not run.
 */
#define LEGACY_GRID 40
#define LEGACY_CELL 50.0f

typedef struct {
    float shoot[LEGACY_GRID][LEGACY_GRID]; /* probability of "shoot" (entry 0) */
    float move_h[LEGACY_GRID][LEGACY_GRID][3];
    float move_v[LEGACY_GRID][LEGACY_GRID][3];
} LegacyTables;

/* 1 when built with cJSON (needed to read the upstream files). */
int legacy_available(void);

/* Tables from one upstream file, starting from upstream's uniform defaults; NULL if unreadable. */
LegacyTables *legacy_load(const char *path);
void legacy_free(LegacyTables *t);

/* Upstream Normalize: grid cell of a displacement in pixels. */
void legacy_cell(float dx, float dy, int *x, int *y);

/*
 * Upstream Find_Best_Probability_Target then Help_Ally, ship by ship in index order: each alive ship
 * takes the live enemy maximising shoot probability x points / distance among enemies not already
 * locked by a lower-index ship; with none free it borrows the first locked target of ship i+1, i+2
 * (mod 3), which this frame can only be a lower-index ship's. targets[] holds enemy indices or -1.
 */
void legacy_targets(const Env *env, LegacyTables *const tables[NUM_SHIPS], int targets[NUM_SHIPS]);

/* Upstream Compute_Action with its three 1..100 draws (vertical, horizontal, shoot, in that order). */
ShipAction legacy_compute_action(const float h[3], const float v[3], float shoot, int vert_draw, int hori_draw,
                                 int shoot_draw);

#endif
