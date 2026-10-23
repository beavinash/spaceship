# Walkthrough — PPO training in C for the upstream Spaceship game

## 1. Delivery summary

Upstream Spaceship's three-ship game now trains with PPO and neural networks, all in C11, inside `PPO_Project/`. The sprint delivered:

- a seeded, headless port of the game with the required rule changes (200 ship health; 100 missiles per ship per episode, no refill);
- an exact fire-solution predictor;
- a per-ship reward with potential-based alignment shaping and an edge penalty;
- hand-written dense/MLP/Adam/checkpoint code and a factorized 3×3×2 policy;
- the PPO learner: rollouts, GAE, clipped loss, threaded per-ship updates;
- `train` / `eval` / `select` / `stress` commands;
- random and frozen-upstream ("legacy") baselines;
- a raylib viewer that can follow a training run's newest checkpoint;
- a one-seed pipeline script;
- one kept checkpoint.

**Model trained.** One run: seed 101, 600 s, 3,796,992 decisions. Its selected checkpoint was evaluated once on the 100 held-out test scenarios, scoring 117.9 ± 3.6 team points with 0 ship deaths. Random play scored 103.7 ± 4.6 and upstream's tables 103.8 ± 4.4.

**Scope is partial.** Tasks T042 and T046 (three 20-minute seeds, AC-018) were stopped by the user ("don't run anymore seeds as it's enough for now") and remain open. Every other backlog task is checked.

**Snapshot.**

- Review base: `e683c7de0594e6108a1189ed00593b6115732f1a` (upstream `main`).
- Head: `cff2ff668835919749d6591a9e84256a8ac57623` on `feature/ppo-training`.
- Scope: `PPO_Project/`.
- State: committed and clean; `git ls-files --others --exclude-standard -- PPO_Project` and `git status --short PPO_Project` are empty.
- Nothing outside `PPO_Project/` differs from the base. The repository root has untracked `artifacts/` and `build/` directories from the earlier `feature/cooperative-ppo` branch; they are outside this scope and untouched.

This report was written after the evidence below was captured.

## 2. Changes in the diff

`git diff --numstat e683c7d -- PPO_Project`: 45 files, 7,315 insertions, 0 deletions. Every file is new, so each row's changed lines are the whole file.

| Changed file and diff line(s) | Observed change | Relevant task |
| --- | --- | --- |
| `PPO_Project/src/rng.h` 1–44 (new) | PCG32 generator, bounded ints, unit floats | T001 |
| `PPO_Project/src/env.h` 1–255, `PPO_Project/src/env.c` 1–625 (new) | Game rules at 1/120 s ticks (movement, cooldown, ammo, missiles, spawns, zigzag, hits, explosions, contact, death, escapes); episode stepping; rewards incl. alignment shaping and edge penalty; 148-float observations; fire solution and shot quality; invariants; state hash | T002–T017, T044, T047–T051 |
| `PPO_Project/src/hash.h` 1–19 (new) | Shared FNV-1a helper | refactor |
| `PPO_Project/src/nn.h` 1–96, `PPO_Project/src/nn.c` 1–441 (new) | Dense layers, tanh MLP, orthogonal init, Adam, gradient clipping, checksummed checkpoints, header shape reader | T018, T019, T021, T022 |
| `PPO_Project/src/dist.h` 1–33, `PPO_Project/src/dist.c` 1–99 (new) | Factorized categorical log-prob, entropy, gradients, sampling, argmax | T020 |
| `PPO_Project/src/ppo.h` 1–120, `PPO_Project/src/ppo.c` 1–340 (new) | Rollout buffer with death masks, GAE with time-limit bootstrap, clipped PPO loss, minibatch update, Agent, defaults (λ 0.99) | T023–T026, T052 |
| `PPO_Project/src/trainer.h` 1–79, `PPO_Project/src/trainer.c` 1–199 (new) | Per-ship collection over arenas, threaded per-ship updates, episode records | T028, T031 |
| `PPO_Project/src/train.h` 1–25, `PPO_Project/src/train.c` 1–235 (new) | Training run: manifest, metrics CSV, periodic/final checkpoints, LR annealing | T029, T030, T051 |
| `PPO_Project/src/eval.h` 1–96, `PPO_Project/src/eval.c` 1–352 (new) | Dev/test splits, random/PPO/legacy policies, JSON reports with mean/std/CI | T032, T033, T035, T051, T052 |
| `PPO_Project/src/selection.h` 1–28, `PPO_Project/src/selection.c` 1–168 (new) | Checkpoint selection on dev, newest-checkpoint lookup | T036, T054 |
| `PPO_Project/src/legacy.h` 1–42, `PPO_Project/src/legacy.c` 1–173 (new) | Upstream `Game_Data` table loader (optional cJSON), grid lookup, target locking, `Compute_Action` | T034, T035 |
| `PPO_Project/src/main.c` 1–267 (new) | `ppo train / eval / select / stress` CLI | T029, T032–T036, T040 |
| `PPO_Project/src/viewer.c` 1–341 (new) | raylib viewer with original artwork, HUD, keys, screenshot check, `--run-dir` follow mode | T037–T039, T054 |
| `PPO_Project/tests/*.c`, `check.h`, `tests.def`, `toy.[ch]` (all new; e.g. `test_env.c` 1–842, `test_nn.c` 1–275, `test_ppo.c` 1–252) | 47 test cases plus the toy learner task | T001–T055 |
| `PPO_Project/Makefile` 1–81 (new) | `test`, `smoke`, `sanitize`, `analyze`, `viewer`, `viewer-check`; optional cJSON/raylib detection | T001, T029, T037, T040 |
| `PPO_Project/scripts/container_train.sh` 1–39 (new) | Build → test → smoke → train → select → archive for one seed | T045 |
| `PPO_Project/checkpoints/index.json` 1–24 (new) | Index for the kept checkpoint (seed, update, decisions, dev metrics, sha256) | T055 |
| `PPO_Project/checkpoints/ppo_seed101.bin` (new, binary, 868,096 bytes) | Selected checkpoint of the seed-101 run; no text line numbers | T055 |
| `PPO_Project/README.md` 1–150, `PPO_Project/PRD.md` 1–461, `PPO_Project/.gitignore` 1–5 (new) | Usage, design, results; PRD with backlog and evidence; ignores `build/`, `runs/`, archives | T041, PRD |

## 3. Verification evidence

Full suite, run fresh after `make clean` from `PPO_Project/`:

```text
make test -> exit 0
47 tests, 589 checks, 0 failed checks, 0 failed tests

make sanitize -> exit 0   (ASan + UBSan over the tests, then 1M random decisions with invariant checks)
47 tests, 589 checks, 0 failed checks, 0 failed tests
stress: 1000000 decisions, 279 episodes, peak 16/64 enemies, 28/64 missiles, 0 violations

make analyze -> exit 0   (clang static analyzer over src/, tests/, viewer)
analyze: 0 findings

make smoke -> exit 0   (two training updates through the real command; header plus rows written)

make viewer-check -> exit 0
screenshot: build/viewer-check.png 1000x1000, 23782 lit pixels, 3 ships, 9 enemies, 17 missiles
hud: Team score 14   escapes 4
```

Other fresh checks:

```text
build without cJSON: make BUILD=build-nocjson CJSON_LIBS= build-nocjson/test_runner -> exit 0
  legacy: built without cJSON; skipped
47 tests, 572 checks, 0 failed checks, 0 failed tests

README command check (every command in README sh blocks, viewer lines in screenshot mode) -> exit 0
11 README commands ran, 1 skipped

scripts/container_train.sh runs/container-check 7 4096 30 --checkpoint-every 1 -> exit 0
47 tests, 589 checks, 0 failed checks, 0 failed tests
smoke: ok
"status":"completed"
archive: runs/container-check.tar.gz (3.1M)

marker scan (find-markers.sh --base e683c7d over the 45 changed paths)
BINARY PPO_Project/checkpoints/ppo_seed101.bin
none found
summary: 45 path(s), 45 scanned, 0 hit(s), 1 binary, 0 deleted, 0 missing
```

Dependency audit: the core is C11 + libc + libm + pthreads. The optional Homebrew dependencies are raylib 6.2 (viewer) and cJSON 1.7.19 (legacy baseline). No vulnerability scanner was run.

**Model training** (separate from the software tests, run once during the sprint, not re-run here): `ppo train --run-dir runs/pilot2 --seed 101 --max-seconds 600 --checkpoint-every 150`. It stopped at the time limit after 1,854 updates (3,796,992 decisions). `ppo select` chose update 1854.

**Model evaluations** (sampled actions, seeded per scenario). The dev and test results were re-run after the final cleanup commit, and the summary and per-seed rows were byte-identical:

```text
dev  (1000-1019): ppo 119.30 +/- 10.29 deaths 0.00 | random 100.70 +/- 10.57 deaths 1.30 | legacy 102.60 +/- 9.97 deaths 0.75
test (10000-10099, --final):
ppo on test (100 scenarios): team score 117.90 +/- 3.62, deaths 0.00 +/- 0.00
random on test (100 scenarios): team score 103.68 +/- 4.65, deaths 1.40 +/- 0.16
legacy on test (100 scenarios): team score 103.78 +/- 4.42, deaths 0.65 +/- 0.12
test accuracy 0.288 / 0.256 / 0.269; aimed-shot share 0.243 / 0.229 / 0.228; edge share 0.033 / 0.113 / 0.042
MQ-1 PASS  MQ-2 PASS  MQ-3 PASS  MQ-4 PASS
```

## 4. Acceptance criteria

| Criterion | Status: met / unmet / unverified | Diff or execution evidence |
| --- | --- | --- |
| AC-001 strict C11 build; core needs only libc, libm, pthreads | met | `make test` exit 0 with `-Werror`; build without cJSON passes 47 tests |
| AC-002 each section 3.1 rule, incl. 200 health and 100 missiles | met | `env_reset…`, `env_movement…`, `env_weapon_cooldown…`, `env_ammo_is_100…`, `env_missile_flight…`, `env_spawns…`, `env_enemy_zigzag…`, `env_missile_damages…`, `env_kill_becomes_explosion…`, `env_contact_damage…`, `env_counts_live_enemies_escaping…` pass |
| AC-003 deterministic replay | met | `env_replay_is_deterministic` passes (10,000 ticks) |
| AC-004 terminated vs truncated; dead ships leave physics | met | `env_episode_ends_terminated_or_truncated`, `env_ship_dies_at_zero_health` pass |
| AC-005 per-ship reward is the configured combination | met | `env_rewards_follow_configured_coefficients` passes |
| AC-006 148 floats, bounded, ordered, zero-padded, no RNG | met | `env_observation_layout_and_bounds`, `env_observation_fire_solution_features` pass |
| AC-007 gradient checks | met | `nn_dense…`, `nn_mlp…`, `dist_factorized…` finite-difference checks pass |
| AC-008 Adam reference; checkpoint round trip and rejection | met | `nn_adam_and_gradient_clipping`, `nn_checkpoint_roundtrip_and_rejection` pass |
| AC-009 GAE, clipped-loss gradients, epoch coverage | met | `ppo_gae…`, `ppo_clipped_loss_gradients`, `ppo_minibatch_epochs…` pass |
| AC-010 toy task learned within a documented budget | met | `toy_ppo_learns_target_reaching` passes (hit rate ≥ 0.9, return ≥ 1.2 in 150 updates) |
| AC-011 one network per ship; counts and death masks | met | `trainer_collects_per_ship_rollouts` passes |
| AC-012 manifest, metrics, checkpoints; threaded = serial | met | `train_run_writes_metrics_csv`, `…manifest_and_checkpoints`, `trainer_threaded_update_matches_serial`; `make smoke` exit 0 |
| AC-013 JSON reports; dev/test split enforcement | met | `eval_random_policy_dev_report`, `eval_ppo_checkpoint_report_matches_schema`, `eval_defaults_to_sampled_actions` pass |
| AC-014 legacy lookups and target locking | met | `legacy_tables_reproduce_upstream_lookups`, `legacy_target_locking_matches_upstream` pass with cJSON |
| AC-015 viewer with artwork, policies, screenshot, follow mode | met | `make viewer-check` exit 0. The README check ran `./build/viewer --run-dir runs/seed1` and `--policy ppo --checkpoint checkpoints/ppo_seed101.bin` in screenshot mode, exit 0. Interactive keys are not automatically verified (section 5) |
| AC-016 sanitizers, 1M-decision stress, analyzer clean | met | `make sanitize` and `make analyze` exit 0, output above |
| AC-017 README commands run as written | met | `11 README commands ran, 1 skipped`; the skipped 20-minute training command is a limitation |
| AC-018 seeds 1–3 trained 20 min each | unmet | T042/T046 stopped at the user's request; one 10-minute run (seed 101) exists instead |
| AC-019 test report per selected checkpoint with MQ-1..MQ-4 | met | One kept checkpoint; test reports for it and both baselines; MQ-1..MQ-4 all PASS |
| AC-020 one-seed pipeline script on this Mac | met | `scripts/container_train.sh` exit 0, archive written |
| AC-021 fire solution matches real missiles | met | `env_fire_solution_matches_real_missiles` passes (1,152 grid cases) |
| AC-022 potential-based shaping; edge band only | met | `env_alignment_shaping_is_potential_based` (incl. γ = 1 telescoping), `env_edge_penalty_only_inside_band` pass |
| AC-023 edge and aimed-shot shares in CSV/JSON | met | `env_counts_edge_ticks_and_aimed_shots`; CSV header and JSON key assertions pass |
| AC-024 10-minute pilot measured against random | met | Pilot report recorded under T053; dev metrics above |
| AC-025 git holds code, docs, selected checkpoint only | met | `git ls-files PPO_Project` has `checkpoints/index.json` and `checkpoints/ppo_seed101.bin`; no `runs/` or `build/` paths |

## 5. Limitations

- **AC-018 unmet.** Seeds 1–3 (T042, T046) were not trained at the user's request. All results come from one training seed (101) and one 10-minute run, so seed-to-seed variation is unknown. MQ-1's original "at least 2 of 3 training seeds" clause cannot be assessed; MQ-1 was judged for the single kept checkpoint.
- **Small aiming gain.** The improvement in aimed shots is small: test aimed-shot share 0.243 ± 0.011 vs random 0.229 ± 0.012, intervals touching. Accuracy (0.288 vs 0.256) and score show a clearer gain.
- **Escapes.** About 127 of the 160 enemies spawned per episode still escape; 300 missiles per episode cannot stop them all.
- **Pilot manifest stamp.** The kept checkpoint's run manifest records `c2928fc…-dirty`: the binary was built from the T052 working tree just before that commit (`5513b14`). `checkpoints/index.json` states this.
- **Training command not re-run.** The README's 20-minute training command was skipped by the README check. The same command shape ran in the 10-minute pilot with a different seed and duration.
- **Viewer keys untested.** Interactive viewer keys (Space, F, R, N, Esc) and live window behaviour were not verified automatically; only screenshot mode is checked.
- **Linux untested.** The Linux/gcc build path was never exercised, because Modal was dropped. The container script ran only on macOS with Apple clang.
- **Toy threshold changed.** The learner-sanity task changed after its first version failed: the never-fire local optimum led to adding progress shaping and a hit-rate criterion (recorded under T027).
- **Prediction scope.** The fire solution, alignment shaping and aimed-shot metric ignore other missiles and assume the ship holds still until it fires.
- **Legacy timing.** The legacy baseline decides every tick, as upstream does; PPO and random decide every 4 ticks. Their shaped returns are therefore not comparable; team score and the other game metrics are.
- **Conditional legacy test.** `legacy_tables_reproduce_upstream_lookups` prints `legacy: built without cJSON; skipped` and passes without checking when built without cJSON. With cJSON (this Mac) it runs fully.
- **No vulnerability scan.** Dependency audit is limited: the core has no third-party code, and the Homebrew raylib 6.2 and cJSON 1.7.19 were not scanned.
- **Upstream quirks.** Kept intentionally and documented: shared enemy cooldown, multi-hit missiles, zigzag jitter and right-wall sticking, top-left missile launch.
