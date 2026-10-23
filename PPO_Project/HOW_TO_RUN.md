# How to run PPO_Project

PPO reinforcement learning for the three-ship Spaceship game, written in C11. It has three parts:

- a headless copy of the game rules;
- a per-ship reward function;
- one neural-network actor and critic per ship, trained with PPO.

Two rule changes from upstream are deliberate: ships have 200 health, and each ship gets 100 missiles per episode.

## 1. Requirements

- **Always needed:** a C11 compiler and `make` (macOS: Xcode Command Line Tools).
- **Optional:** the game viewer needs raylib, and the upstream-tables baseline (`--policy legacy`) needs cJSON. On macOS: `brew install raylib pkg-config cjson`.

Run everything from `PPO_Project/`:

```sh
cd PPO_Project
mkdir -p runs
```

`runs/` holds training runs and reports; git ignores it.

## 2. Build and test

```sh
make test     # 47 unit and integration tests
make smoke    # two tiny training updates through the real command
```

## 3. Watch the trained ships play

`checkpoints/ppo_seed101.bin` is the committed checkpoint from a 10-minute training run.

```sh
make viewer
./build/viewer --policy ppo --checkpoint checkpoints/ppo_seed101.bin
```

Keys: Space pause, F 1×/4× speed, R restart, N next scenario, Esc quit. `--policy random` or `--policy legacy` shows the baselines instead.

## 4. Evaluate against the baselines

Each command plays the 20 development scenarios (seeds 1000–1019) and writes a JSON report.

```sh
./build/ppo eval --policy ppo --checkpoint checkpoints/ppo_seed101.bin --output runs/dev-ppo.json
./build/ppo eval --policy random --output runs/dev-random.json
./build/ppo eval --policy legacy --output runs/dev-legacy.json
```

Each run prints a summary line, for example `ppo on dev (20 scenarios): team score 119.30 +/- 10.29, deaths 0.00 +/- 0.00`.

The 100 held-out test scenarios (`--split test`) are reserved for one final check and need `--final`.

## 5. Train your own

```sh
./build/ppo train --run-dir runs/my-run --seed 1 --total-decisions 100000000 --max-seconds 600
./build/ppo select --run-dir runs/my-run
./build/ppo eval --policy ppo --checkpoint runs/my-run/selected.bin --output runs/my-run/dev.json
```

**Training** stops after `--max-seconds`; 600 s is about 4M decisions on an Apple M3 Pro. It writes `metrics.csv` (one row per update), `manifest.json`, and periodic checkpoints. Every reward coefficient and PPO setting is a flag; `./build/ppo` with no arguments prints them.

**Selection.** `select` scores every checkpoint on the dev scenarios and copies the best one to `selected.bin`.

**Watching live.** To watch a run while it trains, use a second terminal. The viewer loads the newest checkpoint at the start of each episode:

```sh
./build/viewer --run-dir runs/my-run
```

## 6. Results of the committed checkpoint

These are from the 100 held-out test scenarios, with actions sampled from the policy. Values are mean ± 95% CI.

| | PPO | Random | Upstream tables |
|---|---|---|---|
| Team score | 117.9 ± 3.6 | 103.7 ± 4.6 | 103.8 ± 4.4 |
| Ships lost per episode | 0.00 | 1.40 | 0.65 |
| Accuracy | 0.288 | 0.256 | 0.269 |

This is a single 10-minute training run. Design details and full evidence are in `PRD.md` and `WALKTHROUGH.md`.
