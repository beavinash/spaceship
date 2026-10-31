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

---

# Addendum — reward audit and REWARD.md (T056–T059)

## 1. Delivery summary

The user asked for "an another document on reward function and how it works in this spaceship for 3 … detailed with calculations and fix any bugs that can arise while you are investigating deeply." This round delivered:

- the audit;
- fixes for two learning-signal defects (T056, T057);
- per-component reward columns in `ppo eval` reports (T059);
- `REWARD.md` (T058), whose worked examples are pinned by `tests/test_reward_doc.c`.

Scope is complete for this request.

The fixes change only the training signal, not what ships observe or do. The committed checkpoint's dev play metrics were re-checked after each change and are identical (team score 119.30).

**Snapshot.**

- Range: `5b8670c` (HOW_TO_RUN commit) → `78d361d`, scope `PPO_Project/`.
- State: committed and clean. `README.md` is untracked by design.

## 2. Changes in the diff

| Changed file and diff line(s) | Observed change | Relevant task |
| --- | --- | --- |
| `src/env.c` new 355–356; new 411–421; old 445 and 447 removed, new 461; new 635–649 | Φ′ = 0 once the episode ends; `observed_enemies` shared by `env_observe` and the new `env_potential`; `env_rewards` uses `env_potential` | T056, T057 |
| `src/env.h` new 241–246, 255–257 | `env_potential` declaration; reward comment updated | T056, T057 |
| `src/trainer.c` old 144–149 → new 144–147, new 147 | No time-limit bootstrap for the game (zero bootstrap, comment) | T057 |
| `src/eval.c` new 211–220, 318–326; `src/eval.h` new 69–71 | Nine `reward_*` components per ship in `EpisodeResult` and the JSON summary | T059 |
| `src/ppo.c` 183 | λ comment corrected (~5% credit at λ 0.95, not ~3%) | T058 |
| `tests/test_env.c` (+73/−3), `tests/test_trainer.c` (+20), `tests/test_eval.c` (+26), `tests/test_reward_doc.c` (new, 80), `tests/tests.def` (+6) | Observed-slot potential, time-limit terminal, component sums, and the two pinned worked examples | T056–T059 |
| `REWARD.md` (new, 293) | The reward document | T058 |
| `PRD.md` (+33/−3) | Amendment 4, AC-026–AC-028, tasks with evidence | all |

## 3. Verification evidence

```text
make test        -> exit 0   53 tests, 636 checks, 0 failed checks, 0 failed tests
make sanitize    -> exit 0   53 tests, 636 checks, 0 failed checks, 0 failed tests
                             stress: 1000000 decisions, 279 episodes, peak 16/64 enemies, 28/64 missiles, 0 violations
make analyze     -> exit 0   analyze: 0 findings
make smoke, make viewer-check -> exit 0
make test T=reward -> exit 0  4 tests, 51 checks, 0 failed checks, 0 failed tests
REWARD.md section 12 commands: all exit 0; section 9's 42 numbers match fresh ppo eval reports
marker scan (13 changed paths since 5b8670c): none found
```

Model evaluation (dev scenarios, same checkpoint) after the fixes:

```text
ppo on dev (20 scenarios): team score 119.30 +/- 10.29, deaths 0.00 +/- 0.00   (identical play metrics; shaped return -3.758 -> -3.446)
```

## 4. Acceptance criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| AC-026 potential uses only observed slots and equals the best slot input | met | `env_potential_uses_observed_slots` passes |
| AC-027 the 120 s limit is terminal for learning | met | `env_time_limit_is_terminal_for_shaping`, `trainer_time_limit_has_no_bootstrap` pass; the γ = 1 telescoping check expects 0 |
| AC-028 REWARD.md with calculations that trace to code, tests, or commands | met, with an exception | Examples pinned by tests; section 9 regenerated by `ppo eval`. Figures marked *(audit)* come from a probe not kept in the repo (section 5) |

## 5. Limitations

- **AC-028 exception.** The *(audit)* figures in REWARD.md cannot be regenerated with repository commands: slot crowding (5.2 of 8 slots), hidden best shots (about a third), edge-check agreement (99.3–99.9%), and the ~650 px hidden-target distance. The probe was a throwaway.
- **No retraining.** The fixes affect training only, and nothing was retrained (the user paused training), so their effect on learning is unmeasured. The committed checkpoint predates them.
- **Slot crowding (root cause) unfixed.** Explosions fill 5.2 of the 8 enemy slots. Fixing that changes observations and needs retraining.
- **Latent suicide incentive.** If γ is raised toward 1, the escape tax (−6.3 per episode while alive) exceeds death plus damage (−4). At γ = 0.99 the discounted cost is only about −0.18 (REWARD.md 11.2).
- **Slow sanitizer run.** One `make sanitize` run took about 8 minutes while the machine's load average was about 12. Re-timed individually, the heaviest sanitized tests take 5–7 s and the 1M-decision stress run 1 s.

# Addendum 2 — ammo and fire-bomb reward (T060–T064)

## 1. Delivery summary

The user reported "right now it's not avoiding the fire bomb as well as ammo empty within 27 seconds". Shown the proposed code, they replied "make the above changes and let's train for 10-15 minutes and show me the spaceship playing the game." This round delivered:

- **Shot cost.** The default per-missile cost is now −0.05, up from −0.01 (T060).
- **Fire-bomb term.** A `bomb` reward term charges an extra −0.04 per HP lost to explosions (`--bomb`, T061).
- **New metrics.** `ammo_out_seconds` and `bomb_damage` appear in every eval report (T062).
- **REWARD.md** is updated (T063).
- **Retrain.** One 15-minute retrain on seed 101, compared with the kept checkpoint on dev (T064). The viewer showed the result.

Scope is complete for this request. On dev, the retrained ships keep missiles until 81 s on average instead of 27 s. They take 2.6 HP from explosions per kill instead of 6.8, and the team score is 253.0 instead of 119.3. They also take more body hits and lose 0.20 ships per episode (section 5).

**Snapshot.**

- Range: base `b90aa38` (the user's own commit adding the root `artifacts/` and `build/` folders, outside this scope and untouched) → head `6a538bc`, scope `PPO_Project/`.
- State: committed and clean. `README.md` is untracked by design, and `runs/` (including the new checkpoint) is ignored.

## 2. Changes in the diff

| Changed file and diff line(s) | Observed change | Relevant task |
| --- | --- | --- |
| `src/env.c` old 339 → new 346 | Default `.shot = -0.05f` with its rationale comment | T060 |
| `src/env.c` new 248–251, 354, 375, 379, 390 | Explosion contact adds to `bomb_health_lost` (and `stats.bomb_damage`); default `.bomb = -0.04f`; `part.bomb` computed, added to `r`, and accumulated | T061, T062 |
| `src/env.c` new 75–77 | `stats.ammo_out_tick` recorded when a ship fires its last missile | T062 |
| `src/env.h` new 125, 141, 157, 166, 171 | `ShipEvents.bomb_health_lost`, `RewardConfig.bomb`, `RewardParts.bomb`, `EpisodeStats.bomb_damage`, `EpisodeStats.ammo_out_tick` | T061, T062 |
| `src/eval.c` new 207–208, 223, 310, 315, 332; old 353–354 → new 359–360; `src/eval.h` new 58, 63; old 71 → new 73 | `ammo_out_seconds`, `bomb_damage`, `reward_bomb` per episode and in the JSON summary; the report's reward config includes `bomb` | T061, T062 |
| `src/main.c` old 16 → new 16, new 86–87; `src/train.c` old 81 → new 81, old 89–91 → new 89–91 | `--bomb` flag; `bomb` recorded in the run manifest | T061 |
| `tests/test_env.c` 420, 433, 439, 444, 450 (old → new); new 913–986 | Coefficient expectations for −0.05; `frozen_explosion`, `env_explosion_damage_costs_extra`, `env_tracks_ammo_out_and_bomb_damage` | T060–T062 |
| `tests/test_eval.c` old 185–186 → new 185–187, old 193–194 → new 194–195, new 200–242 | Ten components in the sum check (tolerance 1e-3), `reward_bomb` key, report `bomb` config, `eval_reports_ammo_out_and_bomb_damage` | T061, T062 |
| `tests/test_reward_doc.c` 37, 43, new 45, 46 → 47, 73 → 74, 77–78 → 78–79; `tests/test_train.c` old 102 → new 102–103; `tests/tests.def` new 50–51, 54 | Worked examples re-pinned (example 1 return −0.01 with bomb −0.44); manifest expects shot −0.05 and bomb −0.04; three tests registered | T060–T062 |
| `REWARD.md` (+107/−35 across 37 hunks, e.g. new 21–23, 64–72, 141–155, 206–245, 265–291) | Formula, per-enemy table, aimed vs blind shot economics, examples 1–3, section 9 regenerated, new section 10.3 with the retrain result | T063, T064 |
| `PRD.md` new 14–18, 108, 110, 170–171, 184, 484–508, 516; `HOW_TO_RUN.md` 28 | Amendment 5, reward spec, AC-029–AC-031, T060–T064 with evidence, Q3; stale test count removed | all |

## 3. Verification evidence

Code is identical at `80db9e4` and `6a538bc`; T064 changed only `PRD.md` and `REWARD.md`.

```text
make test          (at 6a538bc) -> exit 0   56 tests, 667 checks, 0 failed checks, 0 failed tests
make sanitize      (at 80db9e4) -> exit 0   56 tests, 667 checks, 0 failed checks, 0 failed tests
                                            stress: 1000000 decisions, 279 episodes, peak 16/64 enemies, 28/64 missiles, 0 violations
make analyze       (at 80db9e4) -> exit 0   analyze: 0 findings
make smoke         (at 80db9e4) -> exit 0
make viewer-check  (at 80db9e4) -> exit 0   screenshot: build/viewer-check.png 1000x1000, 23782 lit pixels, 3 ships, 9 enemies, 17 missiles
marker scan (14 changed paths since b90aa38): none found
```

Model evaluation (dev scenarios 1000–1019, sampled actions; separate from the software tests):

```text
ppo train --run-dir runs/bomb101 --seed 101 --total-decisions 50000000 --max-seconds 900 --checkpoint-every 150
  -> exit 0, "status": "time_limit", 3247 updates, 6649856 decisions, git_commit 0e368c1 (clean)
ppo select --run-dir runs/bomb101
  -> selected runs/bomb101/final.bin (update 3247): dev team score 253.00, deaths 0.20, from 22 checkpoints
ppo eval (kept checkpoints/ppo_seed101.bin) -> team score 119.30 +/- 10.29, deaths 0.00 +/- 0.00
ppo eval (runs/bomb101/selected.bin)        -> team score 253.00 +/- 9.80, deaths 0.20 +/- 0.18
ppo eval (update_002100.bin, ~585 s)        -> team score 174.00 +/- 5.89, deaths 0.00 +/- 0.00
ppo eval --policy random                    -> team score 100.70 +/- 10.57, deaths 1.30 +/- 0.32

                          kept    selected
ammo_out_seconds          27.0    81.0
aimed_shot_share          0.243   0.809
accuracy                  0.283   0.729
kills                     26.1    55.0
bomb_damage (team HP)     176.3   145.7
damage_taken (team HP)    216.9   394.8
escapes                   126.6   97.8
return_mean (new reward)  -9.80   -4.63

viewer, seed 1000, HUD at 30 s: kept "ammo 0" for all three ships; selected "ammo 75", "ammo 58", "ammo 66"
```

## 4. Acceptance criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| AC-029 shot −0.05 by default; explosion HP costs an extra −0.04 (`--bomb`), logged as a tenth component | met | `env_rewards_follow_configured_coefficients`, `env_explosion_damage_costs_extra`, `eval_reports_reward_components` pass; a `--bomb -0.1` run writes `"bomb": -0.1` to its manifest |
| AC-030 eval reports explosion damage and ammo-out time | met | `env_tracks_ammo_out_and_bomb_damage`, `eval_reports_ammo_out_and_bomb_damage` pass (replay matches within 1e-9) |
| AC-031 one 15-minute retrain on seed 101, selected on dev, compared with the kept checkpoint | met | Section 3 model evaluation |
| AC-028 REWARD.md traces every number to code, tests, or commands | met, with an exception | Examples pinned by tests; section 9 and 10.3 come from `ppo eval`; the 63% / 18% aimed and blind hit rates are *(audit)* figures (section 5) |
| AC-005 components sum to the return | met | Ten components within 1e-3 (measured ≤ 1.1e-4) on all 20 dev episodes |
| AC-016 sanitizers and static analysis clean | met | `make sanitize`, `make analyze` above |

## 5. Limitations

- **One seed, optimistic selection.** 253.0 is the best of 22 dev scores from a single run. Neighbouring checkpoints scored 176.2 and 190.2, and the held-out test scenarios were not run for the new checkpoint.
- **More body hits, some losses.** Body-contact damage rose from 40.6 to 249.1 team HP, and 0.20 ships are lost per episode (4 of 60 ship-episodes, all after about 87 s). The kept checkpoint lost none. Body contact still costs only −0.01 per HP.
- **A ship went silent mid-training.** At update 2,100 ship 0 fired 2.4 shots per episode; the final checkpoint recovered (93 shots). The higher shot cost makes a "never fire" local optimum possible for an independent learner.
- **Not equal budgets.** The new run trained 900 s against the kept checkpoint's 600 s. At about 585 s (update 2,100) it already scored 174.0 with no losses, but with ship 0 idle.
- **New checkpoint not committed.** It lives in `runs/bomb101/` (ignored). `HOW_TO_RUN.md`, `checkpoints/index.json`, and the local README still describe the kept checkpoint.
- **AC-028 exception.** The 63% / 18% hit rates for aimed vs blind shots, and the per-ship figures above, come from scratch probes not kept in the repo.
- **Loosened tolerance.** The eval component-sum tolerance went from 1e-4 to 1e-3, because float32 sums now drift up to 1.1e-4 at the larger returns. Leaving out any non-zero component still fails the check.
- **Viewer self-check.** `viewer --screenshot` exits 1 when no missiles are on screen (the kept checkpoint at 30 s, the new one at 90 s); the images were still saved and inspected.
