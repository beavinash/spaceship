#define _POSIX_C_SOURCE 200809L

#include "selection.h"

#include <dirent.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eval.h"
#include "nn.h"
#include "trainer.h"

int select_best(const Candidate *c, int n) {
    int best = n > 0 ? 0 : -1;
    for (int i = 1; i < n; i++) {
        const Candidate *a = &c[i], *b = &c[best];
        if (a->mean_score != b->mean_score) {
            best = a->mean_score > b->mean_score ? i : best;
        } else if (a->mean_deaths != b->mean_deaths) {
            best = a->mean_deaths < b->mean_deaths ? i : best;
        } else if (a->update < b->update) {
            best = i;
        }
    }
    return best;
}

static int compare_paths(const void *a, const void *b) {
    return strcmp(((const Candidate *)a)->path, ((const Candidate *)b)->path);
}

/* The update counter in a checkpoint header; 0 if unreadable. */
static uint32_t checkpoint_update(const char *path) {
    int n_nets, sizes[4];
    uint32_t update = 0;
    return checkpoint_shape(path, &n_nets, sizes, &update) == 0 ? update : 0;
}

static int copy_file(const char *from, const char *to) {
    FILE *in = fopen(from, "rb");
    FILE *out = in ? fopen(to, "wb") : NULL;
    int ok = in && out;
    char buf[65536];
    while (ok) {
        size_t got = fread(buf, 1, sizeof buf, in);
        if (got > 0 && fwrite(buf, 1, got, out) != got) {
            ok = 0;
        }
        if (got < sizeof buf) { /* end of file or a read error */
            ok = ok && !ferror(in);
            break;
        }
    }
    if (in) {
        fclose(in);
    }
    if (out && fclose(out) != 0) {
        ok = 0;
    }
    return ok ? 0 : -1;
}

static int score_candidate(Candidate *c) {
    Policy p;
    if (policy_ppo(&p, c->path, EVAL_SAMPLE_DEFAULT) != 0) {
        return -1;
    }
    static EpisodeResult res[100];
    int n = 0;
    int rc = eval_run(&p, "dev", 0, res, &n);
    policy_free(&p);
    if (rc != 0) {
        return -1;
    }
    c->mean_score = summarize(res, n, offsetof(EpisodeResult, team_score)).mean;
    c->mean_deaths = summarize(res, n, offsetof(EpisodeResult, deaths)).mean;
    return 0;
}

/* Every checkpoint of a run in update order: each .bin in checkpoints, then final.bin unless its update is listed. */
static int list_checkpoints(const char *run_dir, Candidate *out, int cap) {
    int n = 0;
    char dir[4096];
    snprintf(dir, sizeof dir, "%s/checkpoints", run_dir);
    DIR *d = opendir(dir);
    if (d) {
        struct dirent *ent;
        while ((ent = readdir(d)) != NULL && n < cap) {
            size_t len = strlen(ent->d_name);
            if (len > 4 && !strcmp(ent->d_name + len - 4, ".bin")) {
                snprintf(out[n].path, sizeof out[n].path, "%s/%s", dir, ent->d_name);
                n++;
            }
        }
        closedir(d);
    }
    qsort(out, (size_t)n, sizeof *out, compare_paths);
    for (int i = 0; i < n; i++) {
        out[i].update = checkpoint_update(out[i].path);
    }
    char final_path[512];
    snprintf(final_path, sizeof final_path, "%s/final.bin", run_dir);
    uint32_t final_update = checkpoint_update(final_path);
    int seen = final_update == 0;
    for (int i = 0; i < n; i++) {
        seen |= out[i].update == final_update;
    }
    if (!seen && n < cap) {
        snprintf(out[n].path, sizeof out[n].path, "%s", final_path);
        out[n].update = final_update;
        n++;
    }
    return n;
}

int latest_checkpoint(const char *run_dir, char *path, size_t size, uint32_t *update) {
    static Candidate all[1024];
    int n = list_checkpoints(run_dir, all, 1024);
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (all[i].update > 0 && (best < 0 || all[i].update >= all[best].update)) {
            best = i;
        }
    }
    if (best < 0) {
        return -1;
    }
    snprintf(path, size, "%s", all[best].path);
    *update = all[best].update;
    return 0;
}

int select_run(const char *run_dir, Candidate *out, int cap, int *n) {
    *n = list_checkpoints(run_dir, out, cap);
    for (int i = 0; i < *n; i++) {
        if (score_candidate(&out[i]) != 0) {
            fprintf(stderr, "select: cannot evaluate %s\n", out[i].path);
            return -1;
        }
    }
    int best = select_best(out, *n);
    if (best < 0) {
        fprintf(stderr, "select: no checkpoints under %s\n", run_dir);
        return -1;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/selected.bin", run_dir);
    if (copy_file(out[best].path, path) != 0) {
        return -1;
    }
    snprintf(path, sizeof path, "%s/selection.json", run_dir);
    FILE *f = fopen(path, "w");
    if (!f) {
        return -1;
    }
    fprintf(f, "{\n  \"split\": \"dev\",\n  \"rule\": \"max mean team score, then fewer deaths, then earlier update\",\n");
    fprintf(f, "  \"selected\": {\"path\": \"%s\", \"update\": %u, \"mean_team_score\": %.4f, \"mean_deaths\": %.4f},\n",
            out[best].path, out[best].update, out[best].mean_score, out[best].mean_deaths);
    fputs("  \"candidates\": [\n", f);
    for (int i = 0; i < *n; i++) {
        fprintf(f, "    {\"path\": \"%s\", \"update\": %u, \"mean_team_score\": %.4f, \"mean_deaths\": %.4f}%s\n",
                out[i].path, out[i].update, out[i].mean_score, out[i].mean_deaths, i + 1 < *n ? "," : "");
    }
    fputs("  ]\n}\n", f);
    return fclose(f) == 0 ? best : -1;
}
