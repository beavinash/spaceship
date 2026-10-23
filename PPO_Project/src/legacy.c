#include "legacy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_CJSON
#include <cjson/cJSON.h>
#endif

void legacy_cell(float dx, float dy, int *x, int *y) {
    const float half = LEGACY_GRID / 2 - 1;
    float gx = dx / LEGACY_CELL + half;
    float gy = dy / LEGACY_CELL + half;
    gx = gx < 0.0f ? 0.0f : (gx >= LEGACY_GRID ? LEGACY_GRID - 1 : gx);
    gy = gy < 0.0f ? 0.0f : (gy >= LEGACY_GRID ? LEGACY_GRID - 1 : gy);
    *x = (int)gx;
    *y = (int)gy;
}

ShipAction legacy_compute_action(const float h[3], const float v[3], float shoot, int vert_draw, int hori_draw,
                                 int shoot_draw) {
    int left = (int)(100 * h[0]);
    int stay_h = left + (int)(100 * h[1]);
    int up = (int)(100 * v[0]);
    int stay_v = up + (int)(100 * v[1]);
    int to_shoot = (int)(100 * shoot);
    ShipAction a = {H_STAY, V_STAY, 0};
    if (hori_draw <= left) {
        a.h = H_LEFT;
    } else if (hori_draw > stay_h) {
        a.h = H_RIGHT;
    }
    if (vert_draw <= up) {
        a.v = V_UP;
    } else if (vert_draw > stay_v) {
        a.v = V_DOWN;
    }
    a.fire = shoot_draw <= to_shoot;
    return a;
}

void legacy_free(LegacyTables *t) {
    free(t);
}

#ifdef HAVE_CJSON

int legacy_available(void) {
    return 1;
}

/* Upstream Read_Data: entry i of a section belongs to cell (x, y) and option i % options. */
static void load_section(const cJSON *root, const char *name, int options, LegacyTables *t) {
    const cJSON *array = cJSON_GetObjectItemCaseSensitive(root, name);
    int count = cJSON_GetArraySize(array);
    for (int i = 0; i < count; i++) {
        const cJSON *item = cJSON_GetArrayItem(array, i);
        const cJSON *xi = cJSON_GetObjectItemCaseSensitive(item, "x");
        const cJSON *yi = cJSON_GetObjectItemCaseSensitive(item, "y");
        const cJSON *p = cJSON_GetObjectItemCaseSensitive(item, "probability");
        if (!cJSON_IsNumber(xi) || !cJSON_IsNumber(yi) || !cJSON_IsNumber(p)) {
            continue;
        }
        double xv = xi->valuedouble, yv = yi->valuedouble;
        if (!isfinite(xv) || !isfinite(yv) || xv < 0 || xv >= LEGACY_GRID || yv < 0 || yv >= LEGACY_GRID ||
            floor(xv) != xv || floor(yv) != yv) {
            continue;
        }
        int x = (int)xv, y = (int)yv, k = i % options;
        float prob = (float)p->valuedouble; /* upstream keeps probabilities as float */
        if (options == 2) {
            if (k == 0) {
                t->shoot[x][y] = prob;
            }
        } else if (name[5] == 'h') {
            t->move_h[x][y][k] = prob;
        } else {
            t->move_v[x][y][k] = prob;
        }
    }
}

LegacyTables *legacy_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "legacy: cannot open %s\n", path);
        return NULL;
    }
    long size = fseek(f, 0, SEEK_END) == 0 ? ftell(f) : -1;
    if (size <= 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        fprintf(stderr, "legacy: cannot size %s\n", path);
        return NULL;
    }
    char *text = malloc((size_t)size + 1);
    size_t got = text ? fread(text, 1, (size_t)size, f) : 0;
    fclose(f);
    if (!text || got != (size_t)size) {
        free(text);
        return NULL;
    }
    text[got] = '\0';
    cJSON *root = cJSON_Parse(text);
    free(text);
    LegacyTables *t = calloc(1, sizeof *t);
    if (!root || !t) {
        fprintf(stderr, "legacy: %s is not valid JSON\n", path);
        cJSON_Delete(root);
        free(t);
        return NULL;
    }
    for (int x = 0; x < LEGACY_GRID; x++) {
        for (int y = 0; y < LEGACY_GRID; y++) { /* upstream Init_Reg_Data: probability 1 / options */
            t->shoot[x][y] = 0.5f;
            for (int k = 0; k < 3; k++) {
                t->move_h[x][y][k] = 1.0f / 3.0f;
                t->move_v[x][y][k] = 1.0f / 3.0f;
            }
        }
    }
    load_section(root, "shoot", 2, t);
    load_section(root, "move_h", 3, t);
    load_section(root, "move_v", 3, t);
    cJSON_Delete(root);
    return t;
}

#else

int legacy_available(void) {
    return 0;
}

LegacyTables *legacy_load(const char *path) {
    fprintf(stderr, "legacy: built without cJSON; cannot read %s\n", path);
    return NULL;
}

#endif
