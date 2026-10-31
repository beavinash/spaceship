# The reward function: how the three ships are paid

This document explains how `PPO_Project` turns what happens in the game into the numbers each ship learns from. It covers every term, the arithmetic, worked examples, what three independent learners do with it, and the reward bugs found and fixed while writing it.

Where the numbers come from:

- **Code** ([`src/env.c`](src/env.c) `env_rewards`, `reward_config_default`): all coefficients and game constants.
- **Tests** ([`tests/test_reward_doc.c`](tests/test_reward_doc.c), [`tests/test_env.c`](tests/test_env.c)): the worked examples. Reproduce with `make test T=reward`.
- **`ppo eval` reports** on the 20 dev scenarios (section 9): episode totals. Commands are in section 12.
- **A one-off audit probe** that is not kept in the repo; those figures are marked *(audit)*.

---

## 1. The formula

Every **decision**, which lasts 4 physics ticks (1/30 s), each ship `i` receives its own reward:

```text
r_i =  0.125 · (upstream points of the enemies ship i killed)      kill      2 / 4 / 8 points → +0.25 / +0.5 / +1.0
     + 0.02  · (hits by ship i's missiles)                         hit
     − 0.05  · (missiles ship i launched)                          shot
     − 0.01  · (health ship i lost)                                damage    clamped to the health it had left
     − 0.04  · (health ship i lost to explosions)                  bomb      on top of damage: −0.05 per explosion HP in total
     − 2     · (1 if ship i died during this decision)             death
     − 0.05  · (live enemies that escaped while ship i was alive)  escape
     + 0     · 0.125 · (points of kills by the other two ships)    ally      team-share knob, off by default
     + 0.5   · (0.99 · Φ_i(s') − Φ_i(s))                           align     potential-based shaping, section 4
     − 0.003 · (1 if ship i's centre is within 60 px of a side wall or the bottom)   edge
```

- **Per ship.** There is no single team reward: each ship's learner sees only its own `r_i` (`ally` is 0 by default).
- **When it is computed.** `env_rewards` is called once after every `env_step`. The events (kills, hits, shots, damage, death, escapes) are counted during that decision's 4 ticks and cleared at the start of the next one.
- **Tunable.** Every coefficient has a `train` flag: `--kill-points --hit --shot --damage --death --escape --ally-share --align --edge --bomb`. The shaping discount follows `--gamma`.
- **Logged.** Each component is accumulated per ship (`stats.parts`), and `ppo eval` reports it as `reward_kill … reward_bomb`. A test checks that the ten components add up to the return.
- **Changed on 2026-10-30 (amendment 5).** The shot cost went from −0.01 to −0.05, and the `bomb` term is new. Section 10.3 explains why.

## 2. One decision, step by step

One `env_step` runs these steps, in upstream's frame order:

1. **Tick × 4.** Each tick: spawn → contact damage (ships in index order) → ships move and fire → missile hits → enemies move → cooldowns → missiles move → escapes removed. Each event lands in the owning ship's `events` entry:
   - **kill:** points to the shooter;
   - **hit:** +1 to the missile's owner for every live enemy it touches;
   - **shot:** counted when a missile actually launches, so pressing fire during cooldown or with no ammo costs nothing;
   - **damage and death:** charged to the ship hit; damage from an explosion also counts as `bomb`;
   - **escape:** charged to every ship alive at that tick.
2. **`env_rewards`** turns the events into `r_i`. It computes the new potential Φ_i(s′), pays `0.5 · (0.99 · Φ′ − Φ)`, then stores Φ′ for the next decision.
3. **The trainer** stores `r_i` in ship `i`'s own rollout. A ship that was already dead at the start of the decision is masked out: its reward is forced to 0 and never trained on.

## 3. What one enemy is worth

Missiles deal 10 damage. The table assumes every missile hits, which is the best case:

| Enemy | Health | Hits to kill | Points | Kill reward | Hit rewards | Shot costs | **Net for the kill** | **Net per missile** |
|---|---|---|---|---|---|---|---|---|
| Type 1 | 20 | 2 | 2 | +0.25 | +0.04 | −0.10 | **+0.19** | +0.095 |
| Type 2 | 30 | 3 | 4 | +0.50 | +0.06 | −0.15 | **+0.41** | +0.137 |
| Type 3 | 40 | 4 | 8 | +1.00 | +0.08 | −0.20 | **+0.88** | +0.220 |

A type-3 kill is worth more than twice as much per missile as a type-1 kill. Upstream pays 2 points per hit for it instead of 1, and the fixed shot cost weighs more on cheap kills.

Ammo is finite: 100 missiles per ship, one every 0.25 s at most. So *which* enemy a missile goes to matters.

**Aimed vs blind shots.** Not every missile lands. For the kept checkpoint on the dev scenarios, a missile that lands is worth about **+0.196**: (reward_kill + reward_hit) ÷ hits per ship = (4.971 + 0.566) ÷ 28.3 (section 9). Shots fired while the fire solution predicts a hit land 63% of the time; blind shots land 18% *(audit)*:

| Shot | Lands | Expected value at −0.01 (before) | At −0.05 (now) |
|---|---|---|---|
| Aimed (a hit is predicted) | 63% | 0.63 × 0.196 − 0.01 = **+0.113** | 0.63 × 0.196 − 0.05 = **+0.073** |
| Blind | 18% | 0.18 × 0.196 − 0.01 = **+0.025** | 0.18 × 0.196 − 0.05 = **−0.015** |

At −0.01 every shot paid, so the best habit was to fire whenever the weapon was ready, and all 100 missiles were gone by about 27 s. At −0.05 only aimed shots pay.

**What it costs to get hurt:**

| Source | Damage per touch | Cooldown | Reward per touch |
|---|---|---|---|
| Type 1 contact | 5 | 1.0 s | −0.05 |
| Type 2 contact | 7 | 1.2 s | −0.07 |
| Type 3 contact | 10 | 1.4 s | −0.10 |
| Explosion (wreck of a kill) | 1 | 0.15 s | −0.01 damage − 0.04 bomb = −0.05, i.e. about −0.33 per second inside it |

- **Fire bombs.** Before amendment 5 an explosion touch cost −0.01 (−0.067 per second), and the kept checkpoint learned to dodge enemy bodies but not wrecks. Per episode it takes 41 HP from bodies but 176 HP from explosions, as much as random play takes (180 HP, section 9). Pinned by `env_explosion_damage_costs_extra`.
- **Dying.** A ship that loses all 200 health pays −0.01 × 200 = **−2** in damage over its life (up to −10 if it all came from explosions), plus **−2** at death: **−4** or more in total.
- **Damage order.** Upstream checks contact damage in ship order with one cooldown per enemy, so when an enemy overlaps several ships, the lowest-index ship takes the hit.

**Escapes.** 160 enemies spawn in 120 s (one every 0.75 s). The trained checkpoint still lets about 127 escape. That costs each living ship 127 × −0.05 ≈ **−6.3 per episode**, the largest single term.

**The edge band.** −0.003 per decision × 30 decisions/s = **−0.09 per second**. A ship that never leaves the band pays −0.003 × 3,600 = **−10.8 per episode**. In the earlier sweep runs without this term, ships sat on the bottom wall 86–97% of the time. The trained ships now spend 2.8% of their alive time anywhere in the band.

## 4. Alignment shaping in depth

Shaping exists because the real payoff for aiming arrives late. A missile takes about 1.6 s (≈50 decisions) to reach an enemy near the top, and a kill needs 2–4 hits. Shaping pays for *getting into a firing position* immediately, and takes the payment back later, so it cannot change which policy is best.

### 4.1 From geometry to Φ

1. **Fire solution.** For each enemy in the ship's 8 observed enemy slots, `env_fire_solution` replays the exact tick rules for a missile fired now: missile acceleration, the enemy's fall, zigzag reversals and wall clamps. The `offset` is the horizontal distance between the centres at contact, or at the closest pass for a miss. Tests fire real missiles and agree on all 1,152 grid cases.
2. **Shot quality.** `q = max(0, 1 − offset / 70)`.

   | offset (px) | 0 | 10 | 20 | 35 (edge of a hit) | 50 | ≥ 70 |
   |---|---|---|---|---|---|---|
   | q | 1.000 | 0.857 | 0.714 | 0.500 | 0.286 | 0 |

3. **Potential.** `Φ_i(s)` is the best `q` over the ship's 8 observed enemy slots. It is exactly the largest per-slot shot-quality input the network sees. It is 0 when the ship is dead, out of ammo, or the episode has ended.

### 4.2 What the shaping term pays

`F = 0.5 · (0.99 · Φ(s′) − Φ(s))`, with Φ between 0 and 1, so F lies between −0.5 and +0.495.

| Situation | Φ(s) → Φ(s′) | F |
|---|---|---|
| Line up a dead-centre shot | 0 → 1 | 0.5 × 0.99 × 1 − 0 = **+0.495** |
| Stay lined up for one decision | 1 → 1 | 0.5 × (0.99 − 1) = **−0.005** |
| The target dies, moves away, or the last missile is fired | 1 → 0 | 0.5 × (0 − 1) = **−0.500** |
| The ship dies, or the 120 s limit ends the episode | Φ → 0 | **−0.5 · Φ** |

### 4.3 Why it cannot change the best policy

From any state s_k, the discounted shaping still to come is:

```text
Σ_t 0.99^(t−k) · 0.5 · (0.99 · Φ_{t+1} − Φ_t)  =  0.5 · (0.99^(T−k) · Φ_T − Φ_k)  =  −0.5 · Φ(s_k)      (Φ_T = 0 at every ending)
```

So the shaped value is the plain value shifted by a fixed amount per state, `V′(s) = V(s) − 0.5 · Φ(s)`. A shift like this can't make any policy better or worse (Ng, Harada and Russell, 1999). Shaping only changes *when* the reward arrives.

### 4.4 Why undiscounted totals show a small negative number

Added up *without* discounting over an episode that starts and ends at Φ = 0, the shaping does not cancel: `Σ F = −0.005 · Σ_t Φ_t`. This "leak" is the price of γ < 1, not a bug.

On the dev scenarios, the trained ships' shaping totals **−0.625 per episode**. That implies Σ Φ ≈ 125 over 3,600 decisions, an average Φ of about 0.035.

## 5. Worked example 1: one ship kills a type-3 enemy

Setup (pinned by `reward_doc_single_ship_type3_kill`):

- **Ship 0** sits still at x = 400, so its missiles' centre x is 410.
- **The target** is a type-3 enemy at x = 385, centre x 410: dead centre. It falls straight down (vx = 0).
- **Ship 0** fires exactly four missiles, then holds still.
- **Ships 1 and 2** stand at x = 800 and 850 with no shot, so their rewards stay 0.

| Decision (tick) | What happens | kill | hit | shot | damage | bomb | shaping | **r₀** |
|---|---|---|---|---|---|---|---|---|
| 0 (4) | Fires missile 1; Φ: 0 → 1 | | | −0.05 | | | +0.495 | **+0.445** |
| 1–67 | Stays lined up (Φ = 1); fires missiles 2–4 at decisions 7, 15, 22 (−0.05 each) | | | | | | −0.005 each | −0.005 or −0.055 |
| 49, 55, 62 (200, 224, 252) | Missiles 1–3 land (−10 HP each) | | +0.02 | | | | −0.005 | **+0.015** each |
| 68 (276) | Missile 4 lands: kill (8 points); Φ: 1 → 0 | +1.00 | +0.02 | | | | −0.500 | **+0.520** |
| 171–216 | The wreck drifts down onto the still ship: 11 touches of 1 HP | | | | −0.01 each | −0.04 each | | −0.050 each |

Ship 0's totals: kill +1.00, hit +0.08, shot −0.20, damage −0.11, bomb −0.44, shaping −0.34 (+0.495 − 67 × 0.005 − 0.500). Episode return: **−0.01**.

Three things to notice:

- **Shaping moves money earlier.** The kill step pays +0.52 instead of +1.02: half of the type-3 bonus was already paid at decision 0, when the ship lined up. Discounted, the shaping still sums to exactly 0 (section 4.3).
- **Holding a sure shot leaks a little.** 67 aligned decisions × −0.005 = −0.335.
- **Standing under the wreck costs more than the kill earns.** The explosion falls at about 100 px/s onto a ship that doesn't move: 11 HP × −0.05 = −0.55, so the return drops from +0.54 to −0.01. Before amendment 5 the same 11 HP cost only −0.11. Stepping aside after the kill keeps the +0.54.

## 6. Worked example 2: three ships, one shared target, one escape

Setup (pinned by `reward_doc_three_ships_shared_target`):

- **Ships 0 and 1** stand at x = 400 and 440, so their missiles' centres are at 410 and 450.
- **The target** is a type-2 enemy (30 HP, 4 points) whose centre is at 430: 20 px from both ships.
- **Ships 0 and 1** each fire two missiles, in the same ticks.
- **A type-1 enemy** at the bottom is about to escape.
- **Ship 2** (x = 800) does nothing.

Shot quality for both shooters is `1 − 20/70 = 0.714`.

| Decision (tick) | Ship 0 | Ship 1 | Ship 2 |
|---|---|---|---|
| 0 (4): both fire; Φ 0 → 0.714 | −0.05 + 0.5 × 0.99 × 0.714 = **+0.3036** | **+0.3036** | 0 |
| 7 (32): both fire again | −0.05 − 0.0036 = **−0.0536** | **−0.0536** | 0 |
| 41 (168): the bottom enemy escapes | −0.05 − 0.0036 = **−0.0536** | **−0.0536** | **−0.05** |
| 48 (196): both first missiles hit (30 → 10 HP) | +0.02 − 0.0036 = **+0.0164** | **+0.0164** | 0 |
| 54 (220): both second missiles arrive on the same tick | kill +0.5, hit +0.02, Φ 0.714 → 0: −0.3571 → **+0.1629** | no hit (its missile meets the new explosion), Φ → 0: **−0.3571** | 0 |
| **Episode return** | **+0.1971** | **−0.3229** | **−0.0500** |

What this shows about three learners:

- **Kill credit goes to whoever lands the killing hit.** Ship 1 did a third of the damage and ended at −0.32. Ship 0 did two thirds and ended at +0.20. When missiles arrive on the same tick, the lower-index ship's missile is checked first, an upstream quirk. `--ally-share 0.5` would give every ship half the value of its allies' kills (+0.25 each here).
- **Escapes tax every living ship.** Ship 2 paid −0.05 for an enemy it never touched, so the team paid −0.15 for one escape.
- **Shaping is per ship.** Both shooters were lined up, so both got the +0.354, and both gave it back when the target died, even though only ship 0 earned the kill.

## 7. Worked example 3: one busy decision

Pinned by `env_rewards_follow_configured_coefficients`. One decision with these events:

| | Events this decision | Arithmetic | **r** |
|---|---|---|---|
| Ship 0 | kill worth 8 points, 3 hits, 2 shots, 7 HP lost, 2 escapes | 0.125 × 8 + 0.02 × 3 − 0.05 × 2 − 0.01 × 7 − 0.05 × 2 = 1.00 + 0.06 − 0.10 − 0.07 − 0.10 | **+0.79** |
| Ship 1 | lost its last 3 HP and died | −0.01 × 3 − 2 | **−2.03** |
| Ship 2 | nothing | | **0** |

With `--ally-share 0.5`, ship 0 also gets 0.5 × 0.125 × 4 = +0.25 for an ally's 4-point kill, giving **+1.04**. No shaping, edge, or bomb terms appear here: there are no enemies or explosions, and every ship is away from the walls.

## 8. How the reward reaches PPO

- **Rollouts.** Each ship has its own rollout of 16 arenas × 128 decisions. Its actor (148 → 128 → 128 → 8 logits) and critic (148 → 128 → 128 → 1) see only its own observation and reward.
- **Advantages.** GAE uses γ = 0.99 and λ = 0.99. In example 1 a hit lands 49 decisions after its shot and keeps (0.99 × 0.99)^49 ≈ 37% of its credit. With λ = 0.95 it would keep (0.99 × 0.95)^49 ≈ 5%, which is why λ was raised.
- **Endings.** Death, team defeat, and the 120 s limit are all terminal: no bootstrap, Φ′ = 0 (section 10.2).
- **After death.** A dead ship's later samples are masked until its arena resets. Kills by its missiles still count toward the game score, but not toward its learning (the audit saw none in 50 ship deaths).
- **No clipping.** Rewards are not clipped or rescaled. Advantages are normalised per minibatch.

## 9. A whole episode, ship by ship

Dev scenarios 1000–1019, mean per ship per 120 s episode, from `ppo eval` (`reward_*` keys), priced with the amendment-5 reward. The PPO checkpoint is the kept one, trained before amendment 5, so this table shows its old habits at the new prices. The game rows don't depend on the reward.

| Component | PPO checkpoint | Random | Upstream tables* |
|---|---|---|---|
| kill | **+4.971** | +4.196 | +4.275 |
| hit | +0.566 | +0.489 | +0.524 |
| shot | −5.000 | −5.000 | −5.000 |
| damage | **−0.723** | −1.700 | −1.548 |
| bomb | −2.351 | −2.397 | −2.365 |
| death | **0.000** | −0.867 | −0.500 |
| escape | −6.330 | −5.813 | −6.300 |
| ally | 0 | 0 | 0 |
| align (shaping) | −0.625 | −0.649 | −2.936 |
| edge | **−0.305** | −1.114 | −1.992 |
| **return** | **−9.796** | −12.854 | −15.843 |
| team score (game points) | **119.3** | 100.7 | 102.6 |
| ships alive at the end | **3.0** | 1.7 | 2.25 |
| score by ship (0 / 1 / 2) | 38.6 / 47.2 / 33.5 | 36.4 / 32.9 / 31.4 | 20.9 / 40.8 / 40.9 |
| ammo runs out at (s, mean per ship) | 27.0 | 27.6 | 34.3 |
| explosion damage (team HP) | 176.3 | 179.8 | 177.4 |

\*Upstream's tables decide every tick (14,400 decisions instead of 3,600). The per-decision terms (shaping leak, edge) therefore pile up four times faster, so their return isn't comparable. Their game metrics are.

Cross-checks:

- **Kill:** PPO's kill term is 4.971 × 3 ships ÷ 0.125 = 119.3 points, exactly the team score.
- **Damage:** the 216.9 HP team damage ÷ 3 × 0.01 = 0.723.
- **Escape:** 126.6 escapes × 0.05 = 6.33.
- **Shot:** 100 missiles × 0.05 = 5.000. Every policy fires all of its ammo.
- **Bomb:** the 176.3 HP of explosion damage ÷ 3 × 0.04 = 2.351.

Random play's escape term (−5.81) is *smaller* than its 129.95 escapes × 0.05 = −6.50, because ships that die stop paying (section 11.2).

**Where PPO's advantage comes from:** +0.78 more kill reward, +0.98 less damage, +0.87 less death, and +0.81 less edge penalty. It gains almost nothing on explosions (+0.05) and nothing on shots. Those are the two habits amendment 5 targets (section 10.3).

## 10. Bugs found and fixed

The audit played 40 episodes each with the checkpoint and with random play, and checked the invariants. These held:

- **Components:** they sum to the return to float32 rounding: within 1e-4 then, and up to 1.1e-4 with the larger amendment-5 returns (the test allows 1e-3).
- **Dead ships:** no reward ever reached a dead ship the learner couldn't see.
- **Edge check:** the end-of-decision edge check agrees with time actually spent in the band in 99.3–99.9% of decisions *(audit)*.

Two defects in the learning signal were fixed, each test-first. Neither changes what ships see or do, so the committed checkpoint's play and every reported game score are unchanged.

### 10.1 Shaping depended on enemies the ship couldn't see (T056)

**Before:** Φ was the best shot over *all* enemies. Each ship observes only its 8 nearest enemies or explosions. On screen there are about 10.2 at a time, and explosions fill **5.2 of the 8 slots** on average *(audit)*. So in about **a third of the decisions where any shot existed**, Φ came from an enemy outside the ship's view: one roughly 650 px overhead, ranked about 10th by distance *(audit)*. The critic was being trained on reward jumps it had no way to predict.

**After:** Φ uses only the 8 observed slots, so it equals the largest shot-quality input. `env_potential_uses_observed_slots` builds a target hidden behind eight explosions and checks Φ ignores it. With one explosion removed, Φ sees the target.

**Effect:** on dev, the checkpoint's shaped return moved from −3.758 to −3.446 (both fixes together); its play is identical.

### 10.2 The time limit was treated as "the game continues" (T057)

**Before:** the observation includes time remaining, and evaluation scores exactly 120 s. Yet at the limit, training bootstrapped from V(final state) and kept Φ there, as if the game went on. With a visible clock, that teaches the critic two contradictory things about the last seconds. This is the inconsistency described by Pardo et al., *Time Limits in Reinforcement Learning* (ICML 2018).

**After:** the limit is terminal for learning: no bootstrap, and Φ′ = 0. `env_time_limit_is_terminal_for_shaping` checks that a ship still lined up at the limit pays back −0.5 × Φ. `trainer_time_limit_has_no_bootstrap` checks no sample carries a bootstrap.

### 10.3 Shots and fire bombs were too cheap (T060–T062, amendment 5)

**Before:**
- **Shots.** At −0.01 per missile even a blind shot paid (+0.025, section 3). Ships fired whenever the weapon was ready and ran dry at **27.0 s** (random play: 27.6 s; the fastest possible is 24.8 s).
- **Fire bombs.** Explosions cost the same −0.01 per HP as an enemy body. The kept checkpoint learned to avoid bodies (41 HP per episode, against random play's 330) but took **176 HP** from explosions, as much as random play.

**After:**
- **Shots.** The shot cost is −0.05, so a blind shot loses about −0.015 while an aimed one still earns about +0.07.
- **Fire bombs.** Each HP lost to an explosion costs −0.04 extra: −0.05 in total, about −0.33 per second inside one.
- **Reporting.** `ppo eval` reports both symptoms directly, as `ammo_out_seconds` and `bomb_damage`.

Tests: `env_rewards_follow_configured_coefficients`, `env_explosion_damage_costs_extra`, `env_tracks_ammo_out_and_bomb_damage`, `eval_reports_ammo_out_and_bomb_damage`. Unlike 10.1 and 10.2, this changes which policy is best, so it needs retraining.

**Retrain (T064).** One 15-minute run on seed 101 (3,247 updates, 6.6M decisions) with the new reward. The selected checkpoint is the run's final one, the best of 22 on dev. It is kept locally in `runs/bomb101/`, not committed.

| Dev, 20 scenarios | Kept checkpoint (old reward) | Retrained (amendment 5) |
|---|---|---|
| ammo runs out (s, mean per ship) | 27.0 | **81.0** |
| shots fired with a predicted hit | 24% | **81%** |
| accuracy | 0.283 | **0.729** |
| team score | 119.3 | **253.0** |
| explosion damage (team HP) | 176.3 (6.8 per kill) | **145.7 (2.6 per kill)** |
| body damage (team HP) | **40.6** | 249.1 |
| ships lost per episode | **0.00** | 0.20 |

The two targeted habits changed as intended, and the policy now aims before it fires. In exchange it hunts closer to the enemies and takes more body hits, which still cost only −0.01 per HP. The ships it lost all died in the last 35 s. This is one seed, picked as the best of 22 dev scores, and it has not been run on the held-out test scenarios.

## 11. Known issues left as they are (with numbers)

### 11.1 Explosions crowd targets out of the observation

This is the root cause behind 10.1. The fix belongs in the observation itself: for example, give live enemies above the ship priority over explosions, or reserve some slots for targets. That changes what the trained network sees, so it needs retraining, which was paused. Recommended for the next training round.

### 11.2 A latent "suicide incentive" if γ is raised toward 1

Only living ships pay the escape term. Over a whole episode, surviving costs about −6.3 in escapes, while dying at once costs −2 (death) + at least −2 (damage) = −4 or more; the bomb term only adds to that. Undiscounted, dying early could therefore look *cheaper* for a ship that can't kill much.

At the trained γ = 0.99 (horizon ≈ 100 decisions ≈ 3.3 s), the escape cost a living ship can see ahead is only:

```text
(126.6 escapes / 3,600 decisions) × 0.05 × 1 / (1 − 0.99)  ≈  0.0352 × 0.05 × 100  ≈  −0.18
```

That is far below death's −2, so it doesn't bite. If γ is ever raised toward 1, fix it first: raise `--death`, or charge escapes against the team rather than per living ship.

### 11.3 Same-tick kill credit favours lower-index ships

This is shown in example 2. It follows upstream's missile order. `--ally-share` softens it if team play matters more than individual credit.

### 11.4 The shaping leak shows up in undiscounted totals

The leak is −0.005 · Φ per decision (section 4.4). It is harmless to learning, but remember it when reading `reward_align`.

### 11.5 The fire solution assumes the ship stands still and ignores other missiles

A missile already on its way can kill the target first, as with ship 1's second missile in example 2.

### 11.6 Shaped return in `ppo eval` uses γ = 0.99

`ppo eval` uses the default γ = 0.99 for shaping. If you train with another `--gamma`, compare game metrics rather than `return_mean`.

## 12. Reproduce

```sh
make test T=reward                                   # reward tests, including examples 1-3
make test T=alignment_shaping && make test T=potential && make test T=time_limit && make test T=edge_penalty
make test T=explosion_damage && make test T=ammo_out
mkdir -p runs
./build/ppo eval --policy ppo --checkpoint checkpoints/ppo_seed101.bin --output runs/dev-ppo.json
./build/ppo eval --policy random --output runs/dev-random.json
./build/ppo eval --policy legacy --output runs/dev-legacy.json
```

The reports' `summary` holds `reward_kill … reward_bomb` and `return_mean` (per ship), plus the game metrics in section 9, including `ammo_out_seconds` and `bomb_damage`.
