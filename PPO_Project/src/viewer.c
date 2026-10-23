/*
 * raylib viewer: watch three ships play one scenario with upstream's original artwork.
 * Keys: Space pause, F 1x/4x speed, R restart, N next scenario, Esc quit.
 * --screenshot PATH --frames N renders N frames in a hidden window, saves PATH, and checks it.
 */
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eval.h"
#include "nn.h"

typedef struct {
    Texture2D ship;
    Texture2D enemy[3];
    Texture2D explosion;
    Texture2D missile;
    Font font;
    int have_font;
} Art;

static Texture2D load_sprite(const char *dir, const char *name, int w, int h) {
    char path[4096];
    snprintf(path, sizeof path, "%s/Pictures/%s", dir, name);
    Image img = LoadImage(path);
    if (!img.data) {
        fprintf(stderr, "viewer: cannot load %s\n", path);
        exit(1);
    }
    ImageResize(&img, w, h); /* upstream resizes every sprite to its collision diameter */
    Texture2D t = LoadTextureFromImage(img);
    UnloadImage(img);
    return t;
}

static Art load_art(const char *dir) {
    Art art;
    int ship = (int)(2 * SHIP_RADIUS), enemy = (int)(2 * ENEMY_RADIUS), blast = (int)(2 * EXPLOSION_RADIUS);
    art.ship = load_sprite(dir, "spaceship.png", ship, ship);
    art.enemy[0] = load_sprite(dir, "aliencraft1.png", enemy, enemy);
    art.enemy[1] = load_sprite(dir, "aliencraft2.png", enemy, enemy);
    art.enemy[2] = load_sprite(dir, "aliencraft3.png", enemy, enemy);
    art.explosion = load_sprite(dir, "explosion.png", blast, blast);
    art.missile = load_sprite(dir, "ammo1.png", (int)(2 * MISSILE_RADIUS), (int)(3 * MISSILE_RADIUS));
    char path[4096];
    snprintf(path, sizeof path, "%s/Fonts/NotoSans-VariableFont.ttf", dir);
    art.font = LoadFontEx(path, 24, NULL, 0);
    art.have_font = art.font.texture.id != 0;
    if (!art.have_font) {
        art.font = GetFontDefault();
    }
    return art;
}

static void draw_scene(const Env *env, const Art *art) {
    ClearBackground(BLACK);
    for (int k = 0; k < env->n_enemies; k++) {
        const Enemy *e = &env->enemies[k];
        DrawTexture(e->dead ? art->explosion : art->enemy[e->type], (int)e->x, (int)e->y, WHITE);
    }
    for (int m = 0; m < env->n_missiles; m++) {
        DrawTexture(art->missile, (int)env->missiles[m].x, (int)env->missiles[m].y, WHITE);
    }
    for (int i = 0; i < NUM_SHIPS; i++) {
        const Ship *s = &env->ships[i];
        if (!s->alive) {
            continue;
        }
        DrawTexture(art->ship, (int)s->x, (int)s->y, WHITE);
        char label[8];
        snprintf(label, sizeof label, "%d", i);
        DrawTextEx(art->font, label, (Vector2){s->x + SHIP_RADIUS - 5, s->y + 2 * SHIP_RADIUS}, 20, 1, WHITE);
    }
}

typedef struct {
    int seed;
    int frames;
    int sample;
    const char *screenshot;
    const char *assets;
    const char *policy;
    const char *checkpoint;
    const char *game_data;
} ViewerArgs;

static int parse_args(int argc, char **argv, ViewerArgs *a) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--sample")) {
            a->sample = 1;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", argv[i]);
            return -1;
        }
        const char *flag = argv[i], *v = argv[++i];
        if (!strcmp(flag, "--seed")) {
            a->seed = atoi(v);
        } else if (!strcmp(flag, "--frames")) {
            a->frames = atoi(v);
        } else if (!strcmp(flag, "--screenshot")) {
            a->screenshot = v;
        } else if (!strcmp(flag, "--assets")) {
            a->assets = v;
        } else if (!strcmp(flag, "--policy")) {
            a->policy = v;
        } else if (!strcmp(flag, "--checkpoint")) {
            a->checkpoint = v;
        } else if (!strcmp(flag, "--game-data")) {
            a->game_data = v;
        } else {
            fprintf(stderr, "unknown option %s\n", flag);
            return -1;
        }
    }
    return 0;
}

/* Saves the frame and checks it shows the game: ships, enemies, missiles, and a non-blank image. */
static int save_and_check(RenderTexture2D target, const Env *env, const char *path) {
    Image img = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&img);
    int ok = ExportImage(img, path);
    Color *px = LoadImageColors(img);
    long lit = 0, total = (long)img.width * img.height;
    for (long i = 0; i < total; i++) {
        lit += px[i].r > 16 || px[i].g > 16 || px[i].b > 16;
    }
    UnloadImageColors(px);
    int ships = 0;
    for (int k = 0; k < NUM_SHIPS; k++) {
        ships += env->ships[k].alive;
    }
    printf("screenshot: %s %dx%d, %ld lit pixels, %d ships, %d enemies, %d missiles\n", path, img.width, img.height,
           lit, ships, env->n_enemies, env->n_missiles);
    UnloadImage(img);
    /* a blank frame lights nothing; even a sparse scene of a few sprites lights well over 0.2% */
    return ok && lit > total / 500 && ships > 0 && env->n_enemies > 0 && env->n_missiles > 0 ? 0 : 1;
}

/* The policy named on the command line; returns 0 or prints why it cannot be built. */
static int make_policy(const ViewerArgs *a, Policy *p) {
    if (!strcmp(a->policy, "random")) {
        policy_random(p);
        return 0;
    }
    if (!strcmp(a->policy, "legacy")) {
        if (policy_legacy(p, a->game_data) != 0) {
            fprintf(stderr, "viewer: cannot load legacy tables from %s\n", a->game_data);
            return -1;
        }
        return 0;
    }
    if (!strcmp(a->policy, "ppo")) {
        int rc = a->checkpoint ? policy_ppo(p, a->checkpoint, a->sample) : CKPT_IO;
        if (rc != 0) {
            fprintf(stderr, "viewer: %s: %s\n", a->checkpoint ? a->checkpoint : "--checkpoint missing",
                    checkpoint_error(rc));
            return -1;
        }
        return 0;
    }
    fprintf(stderr, "viewer: unknown policy %s\n", a->policy);
    return -1;
}

int main(int argc, char **argv) {
    ViewerArgs args = {.seed = 1000, .assets = "..", .policy = "random", .game_data = "../Game_Data"};
    if (parse_args(argc, argv, &args) != 0) {
        fputs("usage: viewer [--policy random|ppo|legacy] [--checkpoint PATH] [--sample] [--game-data DIR]\n"
              "              [--seed N] [--assets DIR] [--screenshot PATH --frames N]\n",
              stderr);
        return 2;
    }
    Policy policy;
    if (make_policy(&args, &policy) != 0) {
        return 1;
    }
    SetTraceLogLevel(LOG_WARNING);
    if (args.screenshot) {
        SetConfigFlags(FLAG_WINDOW_HIDDEN);
    }
    InitWindow((int)ARENA_SIZE, (int)ARENA_SIZE, "Spaceship PPO");
    SetTargetFPS(TICK_HZ / policy.action_repeat);
    Art art = load_art(args.assets);
    RenderTexture2D target = LoadRenderTexture((int)ARENA_SIZE, (int)ARENA_SIZE);

    Env env;
    Pcg32 rng;
    env_reset(&env, (uint64_t)args.seed);
    env.action_repeat = policy.action_repeat;
    pcg32_seed(&rng, (uint64_t)args.seed, 0xe7a1u); /* the same action stream as eval */
    int frame = 0, status = 0;
    while (!WindowShouldClose()) {
        if (!env.terminated && !env.truncated) {
            ShipAction a[NUM_SHIPS];
            policy_act(&policy, &env, &rng, a);
            env_step(&env, a);
        }
        BeginTextureMode(target);
        draw_scene(&env, &art);
        EndTextureMode();
        BeginDrawing();
        DrawTextureRec(target.texture, (Rectangle){0, 0, ARENA_SIZE, -ARENA_SIZE}, (Vector2){0, 0}, WHITE);
        EndDrawing();
        if (args.screenshot && ++frame >= args.frames) {
            status = save_and_check(target, &env, args.screenshot);
            break;
        }
    }
    UnloadRenderTexture(target);
    CloseWindow();
    policy_free(&policy);
    return status;
}
