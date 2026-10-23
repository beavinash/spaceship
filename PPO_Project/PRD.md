# PRD — PPO training in C for the upstream Spaceship game

Status: Approved
Approval evidence: On 2026-10-22 the user replied "PRD approved as drafted, including brew install of raylib/pkg-config/cjson and the 60-min-per-seed training budget. Build the whole sprint backlog." Earlier answers in chat chose everything in C (no Python, no PufferLib), three ships with separate networks, a score-plus-survival reward, and a raylib viewer.
Amendment (2026-10-22, user-required): "make sure we have 200 as health for the ships and we do not have infinite ammo. these constraints are required." Ammo choice: "100 per ship, no refill" (its option text included a small per-shot cost). Compute: "We can use my modal.com account for 5-10 minutes of training once the code and smoke test is done", then "let's use and train 2 ships or seeds and one on my mac. first make sure it runs properly on local before modal.com run". These amendments are folded into sections 2–6 below.
Authoritative backlog: this document, section 5

## 1. Problem and users

The user wants the latest upstream Spaceship game (`Drjakll/spaceship`, `main` at `e683c7d`) trained with real reinforcement learning. Upstream's three-ship game (`init.c`, `Headers/`) learns through per-cell multinomial logistic regression (`Headers/ml_utilities.h`). Labels are hand-assigned after the fact: a missile hit marks the earlier key press as "do", a miss as "don't", and a nearby enemy marks the current move as "avoid". This is supervised imitation of self-labelled snapshots, not return maximization. The learned tables live in `Game_Data/trained{0,1,2}.json`. Upstream's separate `AI_Project/` is a single-ship tabular Q-learner on an independent game, so it does not cover the three-ship game.

Desired outcome: a self-contained `PPO_Project/` folder, all in C11. It holds a headless, seeded port of the upstream three-ship game, an explicit reward function, vectorized rollouts, and a neural-network actor-critic trained with PPO. It also has reproducible evaluation against baselines and a raylib viewer that uses the original artwork.

Research question: can independent PPO learners (one network per ship) score more upstream points than random play or upstream's frozen logistic-regression tables, without dying more often?

## 2. Goals and scope

Goals:

- Train three ships with PPO and neural networks in C, with no CUDA or PufferLib. Seed 1 trains on this Mac (Apple M3 Pro, 12 cores, 36 GB); seeds 2–3 train on Modal CPU containers, only after the full pipeline works locally.
- Apply two required changes to upstream's rules: ship health is 200 (upstream 1000), and ammo is finite at 100 missiles per ship per episode, with no refill (upstream unlimited).
- Make the reward, observations, actions, rollouts, and PPO maths explicit and unit-tested.
- Compare trained checkpoints against baselines on held-out seeds, and watch them in a viewer.

Must-have scope (this sprint):

- Headless C simulation reproducing upstream `main` game rules at a fixed timestep.
- Per-ship reward function with configurable coefficients and logged components.
- Fixed-size ego-centric observations, factorized actions that match upstream's left/stay/right × up/stay/down × shoot/no-shoot structure.
- C neural networks: dense layers, tanh MLP actor and critic, factorized categorical policy, Adam, checkpoints.
- PPO: vectorized rollouts, GAE, clipped surrogate, value loss, entropy bonus, minibatch epochs, metrics logging.
- `train`, `eval`, and `viewer` commands; random and legacy-table baselines; checkpoint selection.

Later (deferred):

- Scripted heuristic baseline, running reward normalization, training resume with optimizer state.
- Accelerate/BLAS matrix kernels, recurrent policies, shared or centralized critics (MAPPO), learned enemies or self-play.

Non-goals:

- Changing upstream's `init.c`, `Headers/`, `AI_Project/`, or `Game_Data/`; they stay as they are.
- Python in the simulation, learner, or evaluation; PyTorch, PufferLib, GPU training. The only Python is the Modal launcher, because Modal's SDK requires it. The earlier `feature/cooperative-ppo` branch and `~/Developer/PRD.md` are left untouched.
- Fixing upstream gameplay quirks (see section 6, Q5).

## 3. Requirements and constraints

Platform: macOS on Apple Silicon. The simulation, networks, PPO, and evaluation need only a C11 compiler, `make`, libc, libm, and pthreads. The viewer needs raylib; the legacy baseline needs cJSON (section 6, Q1). Everything builds and runs from `PPO_Project/`, which references the root `Pictures/`, `Fonts/`, and `Game_Data/` read-only.

### 3.1 Environment (port of upstream `main`)

Physics advances in fixed ticks of 1/120 s, the upstream target frame rate. Each policy decision repeats for 4 ticks, so there are 30 decisions per second. Upstream cooldowns convert to integer tick counts so float drift cannot shift them. Positions are top-left corners, with centre = position + radius, as upstream.

| Rule | Upstream value carried over |
| --- | --- |
| Arena | 1000 × 1000; ship x/y clamped to [0, 950] |
| Ships | 3; start at (400 + 50i, 700); **200 health (user-required; upstream 1000)**; 400 px/s; radius 25; diagonal moves normalized to 400 px/s |
| Weapon | 0.25 s cooldown (30 ticks); missile spawns at ship position, vy 30 px/s upward, +300 px/s² acceleration, radius 10, 10 damage; removed above y = −10 |
| Ammo | **100 missiles per ship per episode, no refill (user-required; upstream unlimited)**; firing with no ammo does nothing and leaves the cooldown untouched |
| Enemy spawn | every 0.75 s (90 ticks); type uniform over 3; x uniform integer [0, 800]; y = −25; vx 200, vy 10 |
| Enemy types | health 20 / 30 / 40; contact damage 5 / 7 / 10; downward acceleration 90 / 72 / 54; points 2 / 4 / 8; attack cooldown 1.0 / 1.2 / 1.4 s |
| Zigzag | vy += a·dt; y += vy·dt; reverse vx when 0 < (int)floor(x) mod 350 < 10; x += vx·dt; clamp x to [0, 950] |
| Missile hit | centre distance < 10 + enemy radius on a live enemy: −10 health to **every** overlapping live enemy, missile removed |
| Kill | health ≤ 0 → explosion: radius 50, damage 1, cooldown 0.15 s (18 ticks), vx 1, vy 100, acceleration −1; shooter's score += points |
| Contact | centre distance < 25 + enemy radius and the enemy's cooldown has elapsed: ship loses that enemy's damage; the cooldown is per enemy, shared by all ships, checked in ship index order |
| Death | health ≤ 0 → ship leaves physics: no moves, shots, or damage |
| Removal | enemies or explosions with y > 1000 are removed; a live enemy removed this way counts as an escape |

Tick order follows upstream's frame: spawn → contact damage per alive ship → ship moves and shots → missile hits → enemy moves → cooldown decrement → missile moves → escape removal. Upstream removes escaped enemies on the next list pass; the port removes them in the same tick, which is the one intended timing difference. Upstream's unused `Enemy_Learn_To_Avoid` is not ported.

Episodes start from a seeded reset. They **terminate** when every ship is dead and are **truncated** at 120 simulated seconds (3,600 decisions). Upstream has no time limit (section 6, Q4). A ship's death is terminal for that ship's trajectory. Its later samples in the same episode are masked out until the arena resets.

### 3.2 Actions, observations, reward

**Action per ship** (MultiDiscrete [3, 3, 2], upstream's factorization): horizontal {left, stay, right}, vertical {up, stay, down}, fire {no, yes}. The log-probability is the sum of the three heads.

**Observation per ship** (proposed, 139 float32 values in [−1, 1], computed from centres without consuming random numbers):

| Block | Count | Fields |
| --- | --- | --- |
| Self | 7 | x/1000, y/1000, health/200, weapon cooldown/30, ammo/100, alive, time-remaining fraction |
| Allies | 2 × 6 | alive, dx/1000, dy/1000, health/200, weapon cooldown/30, ammo/100 |
| Enemies and explosions | 8 × 11 | present, dx/1000, dy/1000, vx/800, vy/1000, health/40, explosion bit, 3 type bits, attack-ready bit |
| Missiles | 8 × 4 | present, dx/1000, dy/1000, fired-by-self bit |

Entity slots hold the nearest entities by centre distance, ordered nearest first, zero-padded, with relative values clamped to [−1, 1].

**Reward per ship per decision** (score + survival; coefficients are configurable, these are defaults):

```text
r_i = 0.125 * points_of_kills_by_i          # +0.25 / +0.5 / +1.0 for type 1 / 2 / 3
    + 0.02  * missile_hits_by_i             # shaping: kills take 2-4 hits
    - 0.01  * missiles_fired_by_i           # ammo is finite: discourage wasted shots
    - 0.01  * health_lost_by_i              # clamped to remaining health
    - 2.0   * died_i                        # once, on the death decision
    - 0.05  * escapes   (if ship i alive)   # live enemies leaving the bottom
    + 0.0   * ally_kill_points * 0.125      # team-share knob, off by default
```

Rewards are not clipped. Each component is accumulated and logged per episode. The upstream game score (points per ship and team total) is tracked separately from the shaped return.

### 3.3 Networks and PPO

- One actor and one critic per ship, 3 ships → 6 MLPs. Each MLP is 139 → 128 → 128 tanh; the actor ends in 8 logits (3 + 3 + 2) and the critic in 1 value. Orthogonal init uses gain √2 for hidden layers, 0.01 for the policy output, and 1 for the value output.
- PPO defaults: 16 arenas × 128 decisions per rollout (2,048 samples per ship network), γ 0.99, GAE λ 0.95, clip 0.2, 4 epochs, 4 minibatches, Adam lr 3e-4 annealed linearly, ε 1e-5, value coefficient 0.5, entropy coefficient 0.01, global grad-norm clip 0.5, per-minibatch advantage normalization.
- Truncation bootstraps from V(final observation). Termination and death do not bootstrap.
- Seeds: every arena and each action sampler use their own seeded PCG32 stream, so a run is reproducible on the same build and platform.
- Training stops at `--total-decisions` or `--max-seconds`, whichever comes first. The run directory holds `manifest.json` (git commit, config, seed), `metrics.csv` (one row per update), and checkpoints.
- Metrics per update: decisions per second, mean episode return per ship, team score, kills, hits, damage taken, deaths, escapes, policy loss, value loss, entropy, approximate KL, clip fraction, explained variance, learning rate.

### 3.4 Baselines and evaluation protocol

- Policies: `random` (uniform per head); `legacy` (upstream `Game_Data/trained{0,1,2}.json` tables, frozen, using upstream target locking and grid lookup); `ppo` (checkpoint, argmax per head by default, `--sample` optional).
- Seed splits: training uses seeds 1, 2, 3 (one independent run each). Checkpoint selection uses development scenarios 1000–1019. The held-out test scenarios 10000–10099 run once, at the end.
- Report per policy: per-seed rows and mean, standard deviation, and 95% CI for team score, per-ship score, kills, hits, accuracy, damage taken, deaths, ships alive at the end, survival time, and escapes. Reports also carry the binary hash, checkpoint hash, and config.
- Model-quality targets (hypotheses, not promises):
  - **MQ-1:** on test scenarios, a selected PPO checkpoint's mean team score exceeds both `random` and `legacy` by more than the sum of the two 95% CI half-widths, for at least 2 of 3 training seeds.
  - **MQ-2:** for the same checkpoints, mean deaths per episode are at most `random`'s.
- Compute: seed 1 trains on this Mac in the background. Seeds 2–3 train on Modal in two parallel CPU containers, each capped at 10 min including build and tests, and only after the local pipeline (tests, smoke, seed 1) works. All three seeds share one `--total-decisions` budget sized from measured throughput so a Modal container finishes in about 8 min. The approved 60 min is the hard cap for the local run. A seed that hits its time cap before the decision budget is reported as truncated. The Modal launcher creates no persistent resources (no Volumes or secrets) and returns the run directory as an archive.

## 4. Acceptance criteria and validation

Software correctness:

| ID | Observable acceptance criterion | How to verify |
| --- | --- | --- |
| AC-001 | `PPO_Project` builds with `-std=c11 -Wall -Wextra -Wpedantic -Werror`; the core needs only libc, libm, and pthreads | `make -C PPO_Project test` |
| AC-002 | The simulation reproduces each section 3.1 rule at fixed ticks, including 200 health and the 100-missile limit | per-rule unit tests with hand-computed fixtures |
| AC-003 | Identical seed and actions give identical trajectories; different seeds differ | state-hash replay test over 10,000 ticks |
| AC-004 | Episodes report terminated vs truncated correctly; dead ships leave physics | boundary tests |
| AC-005 | Per-ship reward equals the configured linear combination; components sum to the total | scripted-event reward fixture |
| AC-006 | Observations have a fixed length, stay within [−1, 1], order slots by distance, zero-pad, and consume no RNG | observation tests |
| AC-007 | Analytic gradients of dense layers, tanh, factorized log-prob, and entropy match finite differences | gradient-check tests (tolerance stated in tests) |
| AC-008 | Adam matches reference values; checkpoints round-trip exactly; corrupt or mismatched files are rejected | optimizer and checkpoint tests |
| AC-009 | GAE handles termination, truncation bootstrap, and masks; clipped-loss gradients are correct; every active sample is visited once per epoch | PPO maths fixtures |
| AC-010 | PPO learns a toy target-reaching task to a documented threshold within a documented update budget (seed 1) | learner sanity test |
| AC-011 | Rollouts use one network per ship across N arenas; sample counts and death masks are correct | rollout tests |
| AC-012 | `train` writes a manifest, per-update metrics, and loadable checkpoints; threaded and serial updates match | smoke run plus tests |
| AC-013 | `eval` writes JSON with per-seed rows and summary statistics for `random` and `ppo`; dev and test splits are enforced | eval tests |
| AC-014 | The `legacy` policy reproduces upstream table lookups and target locking from `Game_Data` | legacy fixture tests |
| AC-015 | The raylib viewer shows three ships with the original artwork, driven by `ppo`, `random`, or `legacy`, and can save a screenshot | `viewer --screenshot` run plus image inspection |
| AC-016 | ASan/UBSan runs of all tests plus 1M random decisions are clean; the clang static analyzer reports nothing | `make sanitize`, `make analyze` |
| AC-017 | README commands for build, train, eval, and view run as written | executing each command |
| AC-020 | One container script builds, tests, smoke-trains, trains, and archives a seed; it runs on this Mac first, then unchanged inside Modal | local script run, then a Modal run |

Model quality (stochastic; measured separately from correctness):

| ID | Observable acceptance criterion | How to verify |
| --- | --- | --- |
| AC-018 | Seed 1 (this Mac) and seeds 2–3 (Modal, ≤10 min each) finish training on the shared decision budget with complete metrics and a selected checkpoint | run directories |
| AC-019 | One test-split report per selected checkpoint against both baselines, with MQ-1 and MQ-2 recorded as pass or fail | `eval --split test` reports |

Commands (created by T001/T040): `make -C PPO_Project test`, `make -C PPO_Project sanitize`, `make -C PPO_Project analyze`. Dependency audit: the core has no third-party code; the Homebrew packages raylib and cJSON are recorded by version in the README.

## 5. Sprint backlog

Sprint: PPO in C for upstream Spaceship
Review base: e683c7de0594e6108a1189ed00593b6115732f1a
Sprint branch: feature/ppo-training (from upstream `main` e683c7d)
Review scope: PPO_Project/

- [x] T001 | P0 | depends: none | AC-001 | Create `PPO_Project` with a strict-warning C11 Makefile running a seeded PCG32 reference test; done when: `make -C PPO_Project test` passes the PCG32 sequence test.
  - red: `make -C PPO_Project test` → exit 2; `tests/test_rng.c:4:10: fatal error: 'rng.h' file not found`
  - green: `make -C PPO_Project test` → exit 0; `2 tests, 13 checks, 0 failed checks, 0 failed tests`
  - regression: `make -C PPO_Project test` → exit 0; `2 tests, 13 checks, 0 failed checks, 0 failed tests` (first task; no earlier tests)
  - scans: `clang --analyze` over src/tests → `clang-analyzer: 0 warnings over 2 files`; dependency audit: unavailable (no third-party code or manifest in the core)
- [x] T002 | P0 | depends: T001 | AC-002 | Implement arena reset from upstream ship constants with the required health/ammo; done when: a test asserts three ships at (400+50i, 700) with 200 health, 100 missiles, 400 px/s speed, ready weapons.
  - red: `make -C PPO_Project test` → exit 2; `tests/test_env.c:4:10: fatal error: 'env.h' file not found`
  - green: `make -C PPO_Project test T=env_reset` → exit 0; `PASS env_reset_matches_upstream_ships`
  - regression: `make -C PPO_Project test` → exit 0; `3 tests, 40 checks, 0 failed checks, 0 failed tests`
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [x] T003 | P0 | depends: T002 | AC-002 | Apply factorized movement actions with upstream diagonal normalization plus clamping; done when: one-tick displacement tests match upstream `Control_Spaceship` arithmetic.
  - red: `make -C PPO_Project test T=movement` → exit 2; `tests/test_env.c:39:5: error: call to undeclared function 'env_tick'`
  - green: `make -C PPO_Project test T=movement` → exit 0; `PASS env_movement_matches_upstream_control`
  - regression: `make -C PPO_Project test` → exit 0; `4 tests, 49 checks, 0 failed checks, 0 failed tests`
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [x] T004 | P0 | depends: T003 | AC-002 | Fire missiles through a 30-tick weapon cooldown; done when: holding fire for 90 ticks spawns exactly three missiles.
  - red: `make -C PPO_Project test T=weapon` → exit 2; `tests/test_env.c:73: CHECK(env.n_missiles == 3) failed`
  - green: `make -C PPO_Project test T=weapon` → exit 0; `PASS env_weapon_cooldown_is_30_ticks`
  - regression: `make -C PPO_Project test` → exit 0; `5 tests, 64 checks, 0 failed checks, 0 failed tests`
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [x] T044 | P0 | depends: T004 | AC-002 | Limit each ship to 100 missiles per episode without refill; done when: holding fire for 3,300 ticks spawns exactly 100 missiles.
  - red: `make -C PPO_Project test T=ammo` → exit 2; `tests/test_env.c:101: CHECK(env.stats.shots[0] == 100) failed`
  - green: `make -C PPO_Project test T=ammo` → exit 0; `PASS env_ammo_is_100_without_refill`
  - regression: `make -C PPO_Project test` → exit 0; `6 tests, 68 checks, 0 failed checks, 0 failed tests`
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [x] T005 | P0 | depends: T044 | AC-002 | Advance missiles along the upstream accelerating upward path; done when: tick-by-tick y matches a closed-form fixture until removal above y = −10.
  - red: `make -C PPO_Project test T=missile_flight` → exit 2; `tests/test_env.c:125: CHECK_NEAR(env.missiles[0].y, y) failed: 700 vs -10.6875`
  - green: `make -C PPO_Project test T=missile_flight` → exit 0; `PASS env_missile_flight_matches_closed_form`
  - regression: `make -C PPO_Project test` → exit 0; `7 tests, 74 checks, 0 failed checks, 0 failed tests` (T004's launch assertion now checks the post-move state, as upstream moves missiles in the launch frame)
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [x] T006 | P0 | depends: T002 | AC-002 | Spawn seeded enemies every 90 ticks with upstream per-type stats; done when: a test asserts spawn ticks, x within [0, 800], per-type stats.
  - red: `make -C PPO_Project test T=spawns` → exit 2; `tests/test_env.c:138: CHECK(env.stats.spawned == 1) failed`
  - green: `make -C PPO_Project test T=spawns` → exit 0; `PASS env_spawns_seeded_enemies_every_90_ticks`
  - regression: `make -C PPO_Project test` → exit 0; `8 tests, 86 checks, 0 failed checks, 0 failed tests`
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [x] T007 | P0 | depends: T006 | AC-002 | Move enemies with the upstream zigzag reversal rule; done when: a fixture reproduces the reversal tick for a known start x.
  - red: `make -C PPO_Project test T=zigzag` → exit 2; `tests/test_env.c:181:16: error: call to undeclared function 'env_add_enemy'`
  - green: `make -C PPO_Project test T=zigzag` → exit 0; `PASS env_enemy_zigzag_reverses_in_window` (two wall expectations in the fixture were corrected to the upstream rule: right-wall sticking and in-window jitter)
  - regression: `make -C PPO_Project test` → exit 0; `9 tests, 100 checks, 0 failed checks, 0 failed tests`
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [x] T008 | P0 | depends: T005, T007 | AC-002 | Apply missile damage to every overlapping live enemy; done when: tests show 10 damage per overlapped enemy with the missile removed.
  - red: `make -C PPO_Project test T=overlapping` → exit 2; `tests/test_env.c:230: CHECK_NEAR(env.enemies[0].health, 30.0f) failed: 40 vs 30`
  - green: `make -C PPO_Project test T=overlapping` → exit 0; `PASS env_missile_damages_every_overlapping_live_enemy`
  - regression: `make -C PPO_Project test` → exit 0; `10 tests, 110 checks, 0 failed checks, 0 failed tests`
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [x] T009 | P0 | depends: T008 | AC-002 | Convert destroyed enemies into upstream explosions credited to the shooter; done when: a kill test asserts radius 50, damage 1, 18-tick cooldown, owner score + points.
  - red: `make -C PPO_Project test T=explosion` → exit 2; `tests/test_env.c:257: CHECK(e->dead == 1) failed`
  - green: `make -C PPO_Project test T=explosion` → exit 0; `PASS env_kill_becomes_explosion_credited_to_shooter`
  - regression: `make -C PPO_Project test` → exit 0; `11 tests, 125 checks, 0 failed checks, 0 failed tests`
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [x] T010 | P0 | depends: T009 | AC-002 | Apply contact damage under per-enemy attack cooldowns; done when: overlapping a type-1 enemy for 240 ticks deals exactly 10 damage.
  - red: `make -C PPO_Project test T=contact` → exit 2; `tests/test_env.c:287: CHECK_NEAR(env.ships[0].health, 190.0f) failed: 200 vs 190`
  - green: `make -C PPO_Project test T=contact` → exit 0; `PASS env_contact_damage_respects_enemy_cooldown`
  - regression: `make -C PPO_Project test` → exit 0; `12 tests, 132 checks, 0 failed checks, 0 failed tests`
  - scans: `clang --analyze` → `clang-analyzer: 0 warnings over 4 files`; dependency audit: unavailable (no third-party code in the core)
- [ ] T011 | P0 | depends: T010 | AC-004 | Remove ships from physics once health reaches zero; done when: a dead ship ignores actions, cannot fire, takes no damage.
- [ ] T012 | P0 | depends: T007 | AC-002 | Count live enemies escaping below the arena; done when: the counter rises once per live enemy crossing y > 1000, never for explosions.
- [ ] T013 | P0 | depends: T011 | AC-004 | Step policy decisions with a 4-tick action repeat; done when: one decision advances exactly 4 ticks with the action held.
- [ ] T014 | P0 | depends: T013 | AC-004 | End episodes on team death or the 120 s limit; done when: tests distinguish terminated from truncated flags.
- [ ] T015 | P0 | depends: T014, T012 | AC-005 | Compute per-ship rewards from configurable event coefficients; done when: a scripted-event fixture yields hand-computed rewards with matching component totals.
- [ ] T016 | P0 | depends: T014 | AC-006 | Encode ego-centric observations with nearest-first entity slots; done when: tests check length 139, [−1, 1] bounds, ordering, zero padding.
- [ ] T017 | P0 | depends: T016 | AC-003 | Verify deterministic replay of seeded episodes; done when: identical seeds plus actions give identical state hashes over 10,000 ticks.
- [ ] T018 | P0 | depends: T001 | AC-007 | Implement batched dense-layer forward/backward passes; done when: finite-difference gradient checks pass.
- [ ] T019 | P0 | depends: T018 | AC-007 | Build orthogonally initialized tanh actor/critic MLPs; done when: tests check output shapes, init gains, deterministic forward passes.
- [ ] T020 | P0 | depends: T019 | AC-007 | Compute factorized 3×3×2 categorical log-probabilities with entropy; done when: hand-computed fixtures plus gradient checks pass.
- [ ] T021 | P0 | depends: T018 | AC-008 | Implement Adam with global gradient-norm clipping; done when: one step on fixed gradients matches reference values.
- [ ] T022 | P0 | depends: T019 | AC-008 | Save versioned checkpoints with dimension headers plus checksums; done when: round trips are exact while corrupt or mismatched files are rejected.
- [ ] T023 | P0 | depends: T020 | AC-009 | Store per-ship rollout samples with activity masks; done when: a buffer test shows post-death steps masked.
- [ ] T024 | P0 | depends: T023 | AC-009 | Compute GAE with termination, truncation bootstrap, masks; done when: outputs match a hand-computed fixture.
- [ ] T025 | P0 | depends: T020 | AC-009 | Compute clipped PPO loss gradients for policy, value, entropy terms; done when: finite-difference checks pass with the expected clip fraction.
- [ ] T026 | P0 | depends: T024, T025, T021 | AC-009 | Run shuffled minibatch epochs with per-minibatch advantage normalization; done when: each active sample is visited exactly once per epoch.
- [ ] T027 | P0 | depends: T026 | AC-010 | Train PPO on a toy target-reaching environment; done when: seed 1 exceeds the documented return threshold within the documented update budget.
- [ ] T028 | P0 | depends: T027, T017 | AC-011 | Collect vectorized arena rollouts with one network per ship; done when: tests confirm per-ship networks, sample counts, death masks.
- [ ] T029 | P0 | depends: T028 | AC-012 | Add a `train` command logging per-update CSV metrics; done when: a 2-update smoke run writes every documented column.
- [ ] T030 | P1 | depends: T029, T022 | AC-012 | Write run manifests with periodic checkpoints; done when: a smoke run directory holds a manifest plus loadable checkpoints.
- [ ] T031 | P1 | depends: T030 | AC-012 | Update the three ship networks on parallel pthreads; done when: threaded results match serial results bit for bit for a fixed seed.
- [ ] T032 | P0 | depends: T017 | AC-013 | Add an `eval` command reporting per-seed metrics for the random policy; done when: a dev-split JSON report lists 20 seed rows with mean, std, 95% CI.
- [ ] T033 | P0 | depends: T032, T022 | AC-013 | Evaluate PPO checkpoints in the `eval` command; done when: a checkpoint report matches the random report schema.
- [ ] T034 | P1 | depends: T032 | AC-014 | Load upstream `Game_Data` tables as a frozen legacy policy; done when: tests reproduce upstream probability lookups for fixed displacements.
- [ ] T035 | P1 | depends: T034 | AC-014 | Port upstream target locking for the legacy policy; done when: a fixture matches upstream target choice, including ally fallback.
- [ ] T036 | P1 | depends: T033 | AC-013 | Select the best checkpoint on development seeds; done when: selection picks the highest mean team score with deterministic tie-breaking.
- [ ] T037 | P1 | depends: T017 | AC-015 | Render the arena with original artwork in raylib; done when: `viewer --screenshot` saves a PNG showing ships, enemies, missiles.
- [ ] T038 | P1 | depends: T037, T033 | AC-015 | Drive viewer ships from checkpoints or baselines; done when: `viewer --policy ppo --checkpoint PATH --screenshot` exits 0 with an image.
- [ ] T039 | P2 | depends: T038 | AC-015 | Show per-ship score, health, kills in a HUD with pause/speed keys; done when: a screenshot shows the HUD text.
- [ ] T040 | P1 | depends: T017 | AC-016 | Add sanitizer plus static-analysis Makefile targets; done when: both targets report zero findings over tests plus 1M random decisions.
- [ ] T041 | P1 | depends: T033 | AC-017 | Document build, training, evaluation, viewing in the README; done when: every README command runs as written.
- [ ] T042 | P1 | depends: T031, T036 | AC-018 | Train seed 1 on this Mac under the shared decision budget; done when: its run directory holds full metrics plus a selected checkpoint.
- [ ] T045 | P1 | depends: T042 | AC-020 | Add a container training script that runs locally before Modal; done when: a local run builds, tests, smoke-trains, trains, archives one seed.
- [ ] T046 | P1 | depends: T045 | AC-018 | Train seeds 2–3 in parallel Modal CPU containers capped at 10 min each; done when: both returned archives hold full metrics plus a selected checkpoint.
- [ ] T043 | P1 | depends: T046, T035 | AC-019 | Evaluate selected checkpoints once on test seeds against baselines; done when: the report records MQ-1 plus MQ-2 as pass or fail.

## 6. Open questions

Q1 and Q2 were confirmed explicitly; the other items were approved as part of the drafted PRD.

- **Q1 — Homebrew packages.** Resolved: approved. Install `raylib`, `pkg-config`, and `cjson` with `brew install raylib pkg-config cjson`; upstream's root game already needs raylib and cJSON. Impact: needed only for the viewer and the legacy baseline. Blocks T034–T039; nothing else.
- **Q2 — Compute budget.** Resolved, then amended by the user: seed 1 on this Mac (60 min hard cap), seeds 2–3 on Modal at ≤10 min each, Modal only after local runs work. Assumption: all seeds share one decision budget sized to the Modal window. Blocks T042, T046.
- **Q3 — Reward coefficients.** Defaults as in section 3.2: hit shaping +0.02, shot cost −0.01, death −2, escape −0.05 per alive ship, ally share 0. All are CLI-configurable and do not block.
- **Q4 — Episode length.** Episodes are truncated at 120 simulated seconds; upstream runs until every ship dies. Does not block.
- **Q5 — Upstream quirks ported as-is.** Enemy attack cooldowns are shared across ships, one missile can damage several overlapping enemies, the zigzag modulo rule lets enemies stick at walls, and missiles launch from the ship's top-left corner. Fixing these would change the game, so they stay and are documented. Does not block.
- **Q6 — Timestep.** Fixed 1/120 s ticks with a 4-tick action repeat (30 decisions/s); cooldowns become integer ticks. Does not block.
- **Q7 — Layout.** All new code lives in `PPO_Project/` on `feature/ppo-training`. Local `main` was fast-forwarded to upstream `e683c7d`; nothing has been pushed. Assets are read from the root `Pictures/` and `Fonts/` rather than copied. Does not block.
- **Q8 — Model shape.** 8 nearest enemy/explosion slots, 8 nearest missile slots, a 139-float observation, 2 × 128 tanh hidden layers. Does not block.
- **Q9 — Evaluation mode.** Greedy (argmax) actions are the primary evaluation; sampled actions are optional. Does not block.
- **Q10 — Leftovers.** The untracked root `build/` and `artifacts/` directories from the old branch stay where they are. Does not block.
- **Q11 — Required rule changes.** Resolved by the user: 200 ship health, 100 missiles per ship per episode with no refill. These intentionally differ from upstream and are labelled as such in code and docs.
