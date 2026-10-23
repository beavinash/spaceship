/*
 * raylib viewer: watch three ships play one scenario with upstream's original artwork.
 * Keys: Space pause, F 1x/4x speed, R restart, N next scenario, Esc quit.
 * --run-dir DIR follows a training run: each new episode plays the run's newest checkpoint,
 * and finished episodes roll on to the next scenario by themselves.
 * --screenshot PATH --frames N renders N frames in a hidden window, saves PATH, and checks it.
 */
#include <raylib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eval.h"
#include "nn.h"
#include "selection.h"

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
        DrawTextEx(art->font, label, (Vector2){s->x + SHIP_RADIUS - 5, s->y - 22}, 20, 1, WHITE);
    }
}

typedef struct {
    char lines[8][96];
    int count;
} Hud;

static void hud_add(Hud *h, const char *text) {
    if (h->count < 8) {
        snprintf(h->lines[h->count++], sizeof h->lines[0], "%s", text);
    }
}

/* HUD text: one line per ship, then time, team score, escapes, and the policy shown. */
static Hud make_hud(const Env *env, const char *policy_label, int seed) {
    Hud h = {.count = 0};
    char line[96];
    int team = 0;
    for (int i = 0; i < NUM_SHIPS; i++) {
        const Ship *s = &env->ships[i];
        team += env->stats.score[i];
        if (s->alive) {
            snprintf(line, sizeof line, "Ship %d  score %d  kills %d  health %d  ammo %d", i, env->stats.score[i],
                     env->stats.kills[i], (int)s->health, s->ammo);
        } else {
            snprintf(line, sizeof line, "Ship %d  destroyed  score %d  kills %d", i, env->stats.score[i],
                     env->stats.kills[i]);
        }
        hud_add(&h, line);
    }
    snprintf(line, sizeof line, "Team score %d   escapes %d", team, env->stats.escapes);
    hud_add(&h, line);
    snprintf(line, sizeof line, "Time %.1f / %d s   seed %d   %s", (double)env->tick / TICK_HZ, EPISODE_SECONDS, seed,
             policy_label);
    hud_add(&h, line);
    return h;
}

static void draw_hud(const Hud *h, const Art *art, int paused, int speed, int over, int follow) {
    for (int k = 0; k < h->count; k++) {
        DrawTextEx(art->font, h->lines[k], (Vector2){12, 10 + 24.0f * k}, 20, 1, RAYWHITE);
    }
    const char *status = over ? "Episode over: R restart, N next scenario" : paused ? "Paused (Space)" : NULL;
    if (over && follow) {
        status = "Episode over: next scenario with the newest checkpoint...";
    }
    if (status) {
        DrawTextEx(art->font, status, (Vector2){12, 10 + 24.0f * h->count}, 20, 1, YELLOW);
    }
    char keys[96];
    snprintf(keys, sizeof keys, "Space pause   F speed %dx   R restart   N next   Esc quit", speed);
    DrawTextEx(art->font, keys, (Vector2){12, ARENA_SIZE - 28}, 18, 1, GRAY);
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
    const char *run_dir;
} ViewerArgs;

static int parse_args(int argc, char **argv, ViewerArgs *a) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--greedy")) {
            a->sample = 0;
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
        } else if (!strcmp(flag, "--run-dir")) {
            a->run_dir = v;
            a->policy = "ppo";
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

/* Follow mode: switch to the run's newest checkpoint if it is newer than the one playing. */
static int follow_latest(const ViewerArgs *a, Policy *p, uint32_t *loaded) {
    char path[512];
    uint32_t update;
    if (latest_checkpoint(a->run_dir, path, sizeof path, &update) != 0) {
        return *loaded ? 0 : -1;
    }
    if (update == *loaded) {
        return 0;
    }
    Policy next;
    int rc = policy_ppo(&next, path, a->sample);
    if (rc != 0) {
        fprintf(stderr, "viewer: %s: %s\n", path, checkpoint_error(rc));
        return *loaded ? 0 : -1; /* keep playing the previous checkpoint */
    }
    if (*loaded) {
        policy_free(p);
    }
    *p = next;
    *loaded = update;
    return 0;
}

int main(int argc, char **argv) {
    ViewerArgs args = {.seed = 1000, .sample = EVAL_SAMPLE_DEFAULT, .assets = "..", .policy = "random",
                       .game_data = "../Game_Data"};
    if (parse_args(argc, argv, &args) != 0) {
        fputs("usage: viewer [--policy random|ppo|legacy] [--checkpoint PATH] [--greedy] [--game-data DIR]\n"
              "              [--run-dir DIR (follow a training run's newest checkpoint)]\n"
              "              [--seed N] [--assets DIR] [--screenshot PATH --frames N]\n",
              stderr);
        return 2;
    }
    Policy policy;
    policy_random(&policy); /* replaced below by the requested policy */
    uint32_t loaded = 0;
    if (args.run_dir ? follow_latest(&args, &policy, &loaded) != 0 : make_policy(&args, &policy) != 0) {
        if (args.run_dir) {
            fprintf(stderr, "viewer: no checkpoint under %s yet\n", args.run_dir);
        }
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
    int seed = args.seed, paused = 0, speed = 1, restart = 1, over_frames = 0;
    int frame = 0, status = 0;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_N) || (args.run_dir && over_frames > 2 * TICK_HZ / policy.action_repeat)) {
            seed++;
            restart = 1;
        }
        if (IsKeyPressed(KEY_R) || restart) {
            if (args.run_dir) {
                follow_latest(&args, &policy, &loaded);
            }
            over_frames = 0;
            env_reset(&env, (uint64_t)seed);
            env.action_repeat = policy.action_repeat;
            pcg32_seed(&rng, (uint64_t)seed, 0xe7a1u); /* the same action stream as eval */
            restart = 0;
        }
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_F)) {
            speed = speed == 1 ? 4 : 1;
        }
        int over = env.terminated || env.truncated;
        for (int d = 0; d < speed && !paused && !over; d++) {
            ShipAction a[NUM_SHIPS];
            policy_act(&policy, &env, &rng, a);
            env_step(&env, a);
            over = env.terminated || env.truncated;
        }
        over_frames = over ? over_frames + 1 : 0;
        char label[64];
        if (policy.kind == POLICY_PPO && loaded) {
            snprintf(label, sizeof label, "ppo update %u (%s, following run)", loaded, policy.sample ? "sampled" : "greedy");
        } else if (policy.kind == POLICY_PPO) {
            snprintf(label, sizeof label, "ppo (%s)", policy.sample ? "sampled" : "greedy");
        } else {
            snprintf(label, sizeof label, "%s", policy.name);
        }
        Hud hud = make_hud(&env, label, seed);
        BeginTextureMode(target);
        draw_scene(&env, &art);
        draw_hud(&hud, &art, paused, speed, over, args.run_dir != NULL);
        EndTextureMode();
        BeginDrawing();
        DrawTextureRec(target.texture, (Rectangle){0, 0, ARENA_SIZE, -ARENA_SIZE}, (Vector2){0, 0}, WHITE);
        EndDrawing();
        if (args.screenshot && ++frame >= args.frames) {
            status = save_and_check(target, &env, args.screenshot);
            for (int k = 0; k < hud.count; k++) {
                printf("hud: %s\n", hud.lines[k]);
            }
            break;
        }
    }
    UnloadRenderTexture(target);
    CloseWindow();
    policy_free(&policy);
    return status;
}
