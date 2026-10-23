# Spaceship PPO (C)

Reinforcement learning for the upstream three-ship Spaceship game (`Drjakll/spaceship` `main`, `e683c7d`), all in C11. It has three parts:

- a headless, seeded port of the game rules;
- a per-ship reward function;
- PPO with one neural-network actor and critic per ship.

Upstream's own learner fits per-cell logistic-regression tables to labels it assigns itself (`Headers/ml_utilities.h`). This project replaces that with return-maximising PPO, and keeps upstream's tables as a baseline.

Everything here lives in `PPO_Project/`. The game, artwork and `Game_Data/` in the repository root are only read.

## Build and check

The core needs a C11 compiler, `make`, libm and pthreads. The viewer needs raylib, and the legacy baseline needs cJSON (`brew install raylib pkg-config cjson`). Run every command from `PPO_Project/`.

```sh
make test          # unit tests: simulation, fire solution, rewards, networks, PPO maths, toy learner, eval
make sanitize      # tests + 1M random decisions under ASan/UBSan with invariant checks
make analyze       # clang static analyzer, fails on any finding
make smoke         # two tiny training updates through the real command
make viewer-check  # renders a frame headlessly and checks ships, enemies, missiles, HUD
```

## Train, select, evaluate, watch

```sh
./build/ppo train --run-dir runs/seed1 --seed 1 --total-decisions 100000000 --max-seconds 1200   # 20 min
./build/ppo select --run-dir runs/seed1
./build/ppo eval --policy ppo --checkpoint runs/seed1/selected.bin --output runs/seed1/dev.json
./build/ppo eval --policy random --output runs/dev-random.json
./build/ppo eval --policy legacy --output runs/dev-legacy.json
make viewer && ./build/viewer --policy ppo --checkpoint runs/seed1/selected.bin
```

`scripts/container_train.sh RUN_DIR SEED DECISIONS SECONDS` runs the whole pipeline for one seed: build, tests, smoke, train, select, and a `.tar.gz` of the run.

- **Run directory.** `train` writes `manifest.json` (git commit, full configuration, final status), `metrics.csv` (one row per update), `checkpoints/update_NNNNNN.bin` and `final.bin`. `select` scores every checkpoint on the dev scenarios and adds `selection.json` and `selected.bin`.
- **Selection rule.** Highest mean team score, then fewer deaths, then the earlier update.
- **Evaluation.** PPO is evaluated by sampling from the policy with a per-scenario seed, so results are reproducible. `--greedy` takes the most likely action instead, but that proved unstable for these policies.
- **Held-out split.** Dev scenarios are 1000–1019. Scenarios 10000–10099 (`--split test`) are reserved for one final run and need `--final`. Training uses seeds ≥ 100000.
- **Viewer keys.** Space pause, F 1×/4× speed, R restart, N next scenario, Esc quit.

## The environment

The port follows upstream's code rule by rule, at a fixed 1/120 s tick. Section 3.1 of [PRD.md](PRD.md) has the full table, and unit tests pin each rule. There are two deliberate, user-required changes:

| | Upstream | Here |
|---|---|---|
| Ship health | 1000 | **200** |
| Ammo | unlimited | **100 missiles per ship per episode, no refill** |

Other choices:

- **Decisions.** Each decision holds an action for 4 ticks (30 decisions/s).
- **Episodes.** An episode ends when all ships are dead, or is truncated at 120 s.
- **Death.** A dead ship's later samples are masked out of training until its arena resets.

Upstream quirks are kept on purpose:

- **Shared enemy cooldown.** An enemy's contact cooldown is shared by all ships, so the lowest-index ship in range takes the hit.
- **Multi-hit missiles.** One missile damages every live enemy it overlaps.
- **Zigzag rule.** The `floor(x) mod 350` reversal rule makes enemies jitter inside the window and stick to the right wall.
- **Launch point.** Missiles launch from the ship's top-left corner.

**Actions** (per ship, factorized like upstream): horizontal {left, stay, right} × vertical {up, stay, down} × fire {no, yes}.

**Fire solution.** For each live enemy, `env_fire_solution` replays the exact tick rules for a missile fired now: missile and enemy acceleration, zigzag and wall clamp. It reports whether the missile would hit, and the horizontal offset at contact or closest pass. Tests fire real missiles and confirm all 1,152 grid predictions. Shot quality is `max(0, 1 − offset/70)`.

**Observation** (per ship, 148 floats in [−1, 1], relative to the ship's centre):

| Block | Size | Fields |
|---|---|---|
| self | 8 | x, y, health, weapon cooldown, ammo, alive, time left, would-hit-now |
| allies | 2 × 6 | alive, dx, dy, health, weapon cooldown, ammo |
| enemies and explosions | 8 × 12 | nearest first: present, dx, dy, vx, vy, health, explosion, type one-hot (3), can-hurt-now, shot quality |
| missiles | 8 × 4 | nearest first: present, dx, dy, mine |

**Reward** (per ship, per decision; every coefficient has a `train` flag):

```text
r = 0.125 · points of own kills (2/4/8)      --kill-points
  + 0.02  · own missile hits                 --hit
  − 0.01  · own missiles fired               --shot
  − 0.01  · health lost                      --damage
  − 2     · died this decision               --death
  − 0.05  · live enemies escaping (alive)    --escape
  + 0     · ally kill points · 0.125         --ally-share
  + 0.5   · (γ·Φ(s′) − Φ(s))                 --align   (Φ = best shot quality with ammo; 0 when dead)
  − 0.003 · centre within 60 px of a side wall or the bottom   --edge
```

- **Alignment shaping** is potential-based: it rewards moving into a firing solution without changing which policy is optimal. Its γ is tied to PPO's γ.
- **Edge penalty.** Wall camping costs about −10.8 per episode at 100% edge time.
- **Separate score.** Upstream's game score (2/4/8 points per kill) is reported separately from this shaped return.

## PPO

Each ship is an independent learner with its own actor, critic and Adam state. The three updates run on separate threads, with results identical to running them serially.

**Networks.** The actor is 148 → 128 → 128 (tanh) → 8 logits; the critic is 148 → 128 → 128 → 1 value.

**Rollouts.** 16 arenas × 128 decisions. GAE uses γ 0.99 and λ 0.99; λ 0.95 kept only ~3% of the credit for a hit landing 1–2 s after the shot. Time-limit endings bootstrap from V(final observation); death and defeat do not bootstrap.

**Update.** Clipped surrogate (0.2), value loss 0.5, entropy 0.01. 4 epochs × 4 minibatches, with advantages normalised per minibatch. Adam at 3e-4, annealed linearly; gradient-norm clip 0.5.

**Learner check.** The same code solves a toy target-reaching task, with a 100% hit rate after 150 updates (`make test`).

**Metrics.** `metrics.csv` and eval reports include team score, kills, accuracy, `aimed_share` (share of shots fired while the fire solution predicted a hit), `edge_share` (alive time in the edge band), deaths and escapes.

## Files

| Path | Contents |
|---|---|
| `src/env.[ch]` | game rules, fire solution, rewards, observations, invariants |
| `src/nn.[ch]`, `src/hash.h` | dense layers, tanh MLP, Adam, checksummed checkpoints |
| `src/dist.[ch]` | factorized categorical policy |
| `src/ppo.[ch]` | rollout buffer, GAE, clipped loss, minibatch update |
| `src/trainer.[ch]`, `src/train.[ch]` | per-ship collection over arenas; the training run |
| `src/eval.[ch]`, `src/selection.[ch]` | baselines, scenario evaluation, checkpoint selection |
| `src/legacy.[ch]` | upstream's frozen tables and target locking |
| `src/main.c`, `src/viewer.c` | `ppo` command; raylib viewer |
| `scripts/container_train.sh` | one-seed pipeline: build, test, train, select, archive |
| `tests/` | unit and integration tests, including the toy learner |
