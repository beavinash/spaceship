#ifndef TRAIN_H
#define TRAIN_H

#include "trainer.h"

typedef struct {
    TrainConfig train;
    long long total_decisions; /* arena decisions; training stops once reached */
    double max_seconds;        /* wall-clock cap; training stops after the update that crosses it */
    int checkpoint_every;      /* updates between snapshots in run_dir/checkpoints */
    const char *run_dir;
} RunConfig;

RunConfig run_config_default(void);

/*
 * Train every ship's agent. Writes run_dir/manifest.json (git commit, configuration, status),
 * run_dir/metrics.csv with one row per update, run_dir/checkpoints/update_NNNNNN.bin every
 * checkpoint_every updates, and run_dir/final.bin. Episode columns average the last
 * EPISODE_WINDOW finished episodes and are empty until one finishes.
 * Returns 0, or -1 on an I/O error or a non-finite loss.
 */
int train_run(const RunConfig *rc);

#endif
