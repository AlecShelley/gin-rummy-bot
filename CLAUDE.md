# Gin Rummy Bot — Handoff (v3 → v4: the road to superhuman)

> **CURRENT HANDOFF: `docs/HANDOFF_V13.md`** (repo + Downloads,
> 2026-09-09) — the champion is **v10home**; **THE HOME VARIANT IS
> THE DEFAULT GAME (user decision 2026-09-04)**: all new work
> targets v10home and home rules unless the user says otherwise.
> Binding since 2026-09-08: every promotion runs the FULL regression
> ladder (v8home/teacher/knocker), not just the predecessor gate.
> Campaign evidence: `results/v9..v12_gates.log` (registrations
> first). `docs/HANDOFF_V8.md` covers the v8home era. Repo restructured 2026-10-15 (see the V13 layout addendum): bot + standing gates at root; studies/, gates/, legacy/, examples/, docs/. This file remains
> the project history and the numbered lessons, which all still
> bind.

## Where this project is

Three generations of bots exist, each beating the last:

1. **v1/v2** (`results/final/`, `results/v2_final/`): behaviour-cloned networks
   from hand-written teachers. Historical; see `RESULTS.md` and `HANDOFF_V2.md`.
2. **v3search — the current strongest bot.** Not a network: a belief-state
   rollout searcher (`expert.py`), the poker-bot architecture. It tracks what
   the opponent's discards/pickups/declined-upcards imply about their hidden
   cards (`belief.py`), samples possible worlds from that belief, evaluates
   candidate moves by simulated playouts (`sim.py`), and deviates from a strong
   heuristic prior only when the paired-world evidence is statistically
   significant. ~2.5 s/hand. Nothing we built exploits it.
3. **Your job: v4 — a bot stronger than any human.** The roadmap is at the
   bottom. NOTE 2026-08-23: the value-network step (old #1) and correlated
   sampling (old #2) were both built and gated this session — negative and
   tie respectively (lessons #8, #9). The open frontier is structural search
   changes (candidate set, decision rule, opponent-model conditioning).

**The game is scored under FULL RULES** (user decision, 2026-08-22): lay-offs
onto the knocker's melds, undercut (+25 to the defender) applied AFTER
lay-offs, no lay-offs against gin, dead hands 0-0. Implemented exactly in
`fullrules.py` — see "Full rules" below for why this needed no engine change.
**DEAD-HAND RULE (user decision, 2026-08-30): STANDARD.** Once the stock is
down to 2 cards, the player to move may not draw or pick up — they knock/gin
or the hand is dead 0-0. Hardwired in `sim.fast_legal_actions`; the ONE
deliberate divergence from the rlcard judge (which allows pickup chains on a
dead stock), accounted for in `sim.py --verify`. Match results gated before
2026-08-30 were measured under the rlcard variant; the difference is
endgame-marginal (v6 vs teacher re-gate: results/v7_regate_deadhand.log).
Per-hand point differential is the objective (no match/box bonuses — user
explicitly deferred those; adding them is a one-line payoff change and would
shift optimal play toward more knocking, closer to human match strategy).

Run the bot:

```bash
.venv/Scripts/python.exe play_vs_bot.py v3search --hands 5     # play it
.venv/Scripts/python.exe sim.py --verify                       # kernel gates
.venv/Scripts/python.exe fullrules.py --test                   # scoring gates
```

**Position evaluator** (chess-engine-style analysis for the user's own
study; added 2026-08-23): describe a position in a small text file (hand,
opponent's known cards, pile, optional event history and opponent knock
model) and get every candidate's EV in points, the teacher prior, the
paired-significance verdict, the recommendation, and the belief's read on
the opponent's likely holdings. Backed by `RolloutExpert.analyze()` (the
refactored single source of decide(); decision parity verified by
`refactor_check.py`).

```bash
.venv/Scripts/python.exe evaluate.py --example > position.txt   # template
.venv/Scripts/python.exe evaluate.py position.txt --worlds 192
.venv/Scripts/python.exe evaluate.py position.txt --html --open  # visual report
```

`--html` (viz.py) writes a self-contained visual report: turn history,
pile/stock, opponent known+hidden cards with the belief read, hero hand
grouped by best melding, per-candidate EV bars, and the recommended move
highlighted in place. Sample positions: `sample_turn1.txt` (obvious take),
`sample_turn2.txt` (significant deviation — the engine refuses the
deadwood-optimal K♦ discard because the opponent showed two kings),
`sample_turn3.txt` (KNOCK at deadwood 9 -- re-audited 2026-08-29: the old "declines a knock" claim was doc rot; no engine config reproduces it).

**Stage-aware boost (user-found bias, 2026-08-23).** True opponent-hand
deadwood falls ~52 (opp turn 0) → ~6 (turn 10+); a flat boost=2.5 makes
turn-1 opponents ~25 pts too strong (turn-1 hand EVs read ~20 pts too
pessimistic) and late opponents too weak (`corr_sample.py --stages`).
Fix: `sample_boost="auto"` in RolloutExpert = min(3.0, 0.5 × observed
opponent turns) — the evaluator's default. NOTE: the promoted v4 match
config still uses the flat 2.5 it was gated with; re-gating v4 with "auto"
is a cheap open item (paired diffs were robust to world bias all session,
so expect ≈ no change head-to-head, but absolute leaf/EV realism improves).
Position files also accept my-draw/my-discard/my-pickup events for a full
interleaved transcript (display + hero_discarded only; opp-* events alone
shape the belief). Even b=3.0 can't reach late-game truth (sampled ~21 vs
true ~9 at turn 9+) — the belief marginals bound what adjacency boosting
can express; a learned hand-generator (old roadmap #2b) is the known next
step if that gap ever matters.

Engine-driven matches (the standard harness — seat-swapped, 95% CIs):

```python
import multiprocessing as mp, meta
with mp.Pool(12) as pool:
    m, ci, n = meta.match("v3search", "teacher", 1000, 12, pool,
                          seed=..., rules="full")
```

## Current strength (all seat-swapped, 95% CI)

**v7 — THE CURRENT CHAMPION (PROMOTED 2026-08-31,
`results/v8_gates.log`).** v6's search + particle filter with the
**opponent-action likelihood redesigned** after the hidden-5♠ incident
proved the v6 pickup likelihood INVERTED (junk-dump takes scored 0.97,
meld-extension takes 0.80, though the teacher takes both
deterministically): `pickup_mode="rule"` models the teacher's binary
take rule (take iff gain ≥ 1) with tau_p=0.1 as the rule-mismatch rate
plus a meld tier (meld takes can never score below junk dumps), and
`mh_all_takes=True` stops MH rejuvenation from forgetting old
pickup/decline evidence (the window now applies to discards only).
tau_p was calibrated for the first time (72-config grid,
`results/v7_calibrate_pickup.log`). Offline: held-out Brier beats v6
on teacher AND clone streams; worlds gate improves the search-consumed
stats (late truth overlap .560→.580, knock-Brier .239→.224);
truth-keyed read regressions (`read_regress.py`) hold-or-sharpen
(hard3's 5♠/6♠ → 0.96/0.91). Match vs v6: TIE, as lesson #9 predicts
for symmetric belief realism (selection +0.80 ± 1.62 seed 74100;
fresh confirm −0.30 ± 0.99 @ 3984 seed 74500) — promoted on posterior
quality per the HANDOFF_V7 gate protocol. Probe clean: ad_self +4.55,
ad_jam +4.54, ad_self_k10 +7.69, teacher +1.98. ~1.2× v6 cost
(0.52–0.56 s/searched-move). Registry `v7`; registry `v6` is FROZEN
at its gated config (pickup_mode=logistic, tau_p=1.0, windowed MH);
`play_vs_bot.py v7`; the evaluator inherits v7 automatically via
`particle.make_model`/DEFAULT_PF.

**v6 — prior champion (PROMOTED 2026-08-28).** `pf_expert.py`
PFPolicy: the UNCHANGED v5 search (staged 16/48/192 + deep 2-ply) with
worlds sampled from a **particle-filter posterior** (`particle.py`:
sequential PF over the opponent's hidden hand; softened-teacher
likelihoods — τ-softmax over the dw+μ·live energy with an ε-floor and a
0.02 robust likelihood clip; atomic draw/discard turns pricing
drew-and-dropped at 1/|pool|; fully-adapted APF kept-draw step;
belief-guided repair on hero-draw refutations; and the load-bearing
piece, **resample-move** — windowed-MH card-swap rejuvenation after each
ESS resample, without which the filter only TIES the old belief).
~1.33× v5 cost (0.44 s/searched-move under fan-out). All gates green
(`results/v6_gates.log`): vs **v5 +1.92 ± 0.97** (3984 fresh; selection
+2.31 ± 1.59), compute-matched control (v5 @ 16/48/256) only +0.50 ±
0.94 at the same cost — the architecture wins; **teacher +2.56 ± 1.65 —
the FIRST head-to-head win over the patient teacher** (v5 tied);
smart +8.76 ± 1.31; probe: **ALL 21 adversaries lose** (best −0.26 ±
3.24, `results/probe_v6_full_20260827.log`) — the cleanest probe of any
bot here. New oracle gate (`oracle_gate.py`): perfect opponent-hand
knowledge is worth **+15.91 ± 1.44** over v5 — v6 captured ~12% of that
bound, so most of the belief headroom is STILL OPEN. Registry `v6`;
`play_vs_bot.py v6`. Campaign record: `results/v6_gates.log`; prior-art
survey with design rules: `results/v6_prior_art.md`.

**v5 — prior champion (2026-08-27).** `adaptive.py`
AdaptiveExpert: staged world budget (16/48/192 — cheap screen, escalate
near-significant decisions, deviate only on highest-stage evidence) +
2-ply close-call rollouts (exact max over top-3 branches at the hero's
first rollout decision, final stage only), on the quirk-corrected serial
loop. Config: T=16, z=1.5, opp_model (10,1.0), boost 2.5; ~1.7× v4
wall-clock (~0.33 s/searched-move under match fan-out). All gates green
(`results/v5_q/v5_gates.log`, probe `results/probe_v5_full_20260826.log`):
vs **v4 +1.20 ± 0.93** (3984 fresh hands), vs compute-matched v4-K80
+1.61 ± 1.55 (the architecture wins, not the cycles), teacher −0.12 ± 0.96
(tie), smart +6.83 ± 1.27, **unexploited** by the 21-adversary probe
(best +0.51 ± 3.11 = the patient-teacher artifact). Registry `v5`;
`play_vs_bot.py v5`. NOTE: the equal-wall-clock-tuned variant
(16/40/128, mid_alpha .75) is a TIE vs v4 — v5's edge needs its 1.7×
budget; at fixed v4 cost, keep v4.

**Full-rules exploitability (2026-08-23, CURRENT leaf-fixed bot, fresh seed
20260823): unexploited.** Best of 21 dedicated adversaries gained **+0.59 ±
2.58** (max of 21 noisy measurements — statistically nothing); most LOSE 1–4
pts/hand trying. `results/v3_meta/exploit_v3search_full.json`,
`results/probe_v3search_full_20260823.log`. The pre-leaf-fix measurement gap
from the last handoff is closed.

**Full rules, current (rules-aware leaf), 888 hands each:**

| v3search vs | pts/hand |
|---|---:|
| teacher (its own heuristic prior) | **+1.41 ± 1.86** |
| knock-at-10 (deterministic) | **+1.97 ± 1.55** |
| smart-draw agent | **+4.17 ± 1.62** |

**Full rules, 1000-hand matrix** (run with the older biased leaf — expect the
current bot to do a bit better; re-run is cheap): v3search beat rule +13.1,
v1 +5.3, v3-clone +5.3, knock6 +3.2, v2 +2.3, knock10 +1.3.
`results/v3_meta/matrix_full.json`. Note the patient teacher extracts MORE
from weak early-knockers than v3search does (e.g. +7.4 vs smart to our +4.2)
while losing/tying head-to-head — non-transitivity worth investigating.

**Classic rules (historical, `results/v3_meta/matrix.json`):** v3search beat
every member; exploitability probes: best of 21 dedicated adversaries gained
**−0.10 ± 2.71** against it (i.e. nothing beats it), while v2 leaked
**+4.83 ± 2.76** and every fixed heuristic leaked 3–5.

**Nash of the full-rules empirical game:** 100% on the patient teacher, with
v3search a statistical co-optimum (they tie head-to-head; v3search dominates
the rest of the field). Under classic rules the Nash was a 70/27/3 mixture —
analysis showed the weights were noise at the matrix's ±1.5 CI floor.

**Estimated absolute level:** likely the strongest *documented* gin rummy bot
(published academic work — the EAAI-21 challenge line — uses simpler hand
estimation + 1-ply heuristics, no search, no exact lay-off optimization, no
exploitability measurement). Well short of what a Suphx-scale learned system
would be. Vs humans: strong club / low expert — perfect endgame arithmetic and
calibrated hand-reading, but no long-horizon draw planning.

## The architecture (file by file)

| File | What it is |
|---|---|
| `sim.py` | Fast id-space engine kernel: `min_deadwood_ids` (own meld solver), `fast_legal_actions` (judge replica), snapshot/restore, `apply_world` determinization, `playout`, `FastTeacher` (exact port of the tuned heuristic), `MixedKnock`, `set_rules()`. 536 rollouts/s/core classic. `--verify --bench --stress` gates. |
| `fullrules.py` | Exact full-rules scoring: optimal lay-offs (chained run extensions, adversarial knocker arrangement, joint defender melding+layoff optimization), undercut after lay-offs. `--test`. |
| `belief.py` | Opponent-hand belief: per-card weights from move-history events (decline α=0.25, discard β=0.35, pickup γ=0.5 — calibrated, held-out Brier **+12.5%** vs uniform), weighted world sampling, and `build_obs` (the 10×52 observation). |
| `expert.py` | `RolloutExpert` (K=48 worlds, T=16 truncation, z=1.5 teacher-prior deviation test) + `ExpertAgent` env wrapper. **This is the bot.** |
| `meta.py` | The empirical meta-game: 38-member population registry (heuristics, mixed-knock grid, network bots, search adversaries), engine matches, RM+/FP Nash solver, BR search, `probe_exploitability`, `hypothesis_test`. All rules-parameterized. |
| `teacher.py` / `tune_teacher.py` | The v2 deterministic heuristic + its tuner (frozen; `FastTeacher` in sim.py is its exact port). |
| `hist_env.py`, `v3_bot.py`, `exit_gen.py`, `exit_train.py` | The ExIt distillation pipeline (kept for reference — **it failed**, see below). |
| `gin_rummy_bot.py`, `head_to_head.py`, `benchmark.py` | v1/v2-era network wrapper + env-based harnesses. `make_env(args)` honors `args.obs` ("base"/"hist") and `args.rules` ("classic"/"full"). |
| `results/v3_meta/*.json` | matrices, nash, exploitability probes, hypothesis tests. `results/*.log` = raw evidence for every number above. |

## Home variant (user's table rules — implemented 2026-08-24)

**OPENING RULE (user decision 2026-09-01, binding — closes the old
"dealer's second option" fidelity gap):** the first player may take
the flip; if they decline, the second player may take it (playing out
of turn, then play returns to the first player); if both decline, the
first player MUST draw from stock. Passes move no cards but are
public decline evidence on the flip. Engine/bot/evaluator must all
model this.

10+10 deal, one card flipped to found the pile, knock threshold = the
flip's deadwood value. `sim.new_game_home(seed)` (transform of the 11/10
deal; `rnd._home_flip_id` lets Belief seed its replayed pile — the flip is
nobody's discard). Search adapts with ZERO retraining; the threshold is
public, so the opponent knock model becomes exact: (flip, 1.0).

- Behavior confirmed (`home_variant.py --stats`, 30k hands,
  knock-when-legal both seats): ace flip → avg 44.5 steps, 50% gin;
  ten flip → 25.7 steps, 98% knock. Monotone across flips;
  `results/home_stats.log`.
- Strength carries over (`home_variant.py --match`, 984 hands each):
  v4home vs knock-when-legal **+5.38 ± 1.60**; vs teacher +1.23 ± 1.95 —
  consistent with standard-game numbers. `results/home_match_*.log`.
- Evaluator: add `knock-max: N` to a position file (flip at the bottom of
  `pile:`).

## Full rules: why no engine change was needed

Laying off is never a strategic decision — moving a card onto the knocker's
melds strictly reduces the defender's count, so optimal defenders always lay
off maximally. Therefore full rules = a different TERMINAL PAYOFF on the same
game tree. `fullrules.defender_residual(knocker_ids, defender_ids)` computes
it exactly: the knocker picks, among their minimum-deadwood arrangements, the
one minimizing the defender's lay-off; the defender jointly optimizes own
melding + lay-offs (including keeping a suboptimal own-melding when lay-offs
compensate). Cached, ~sub-ms. `sim.set_rules("full")` switches every playout;
`args.rules="full"` switches env scorers. Worker entry points call
`set_rules` from job parameters (Windows spawn = no global inheritance).

Measured consequence: knocking is much weaker under full rules (patient
teacher vs knock-at-10 went +0.91 classic → **+4.70** full). This matches why
human experts fear the late 10-point knock.

## Hard-won lessons — do NOT relearn these (all measured)

1. **Model-free RL fails in this environment.** Six configurations (NFSP, DQN
   variants, policy gradient, warm-started NFSP) all ended worse than their
   own initialization. Cause: knocking is legal only at deadwood ≤ 10, random
   exploration reaches it ~7% of hands. Full evidence in `HANDOFF_V2.md` /
   `RESULTS.md`. Supervised learning on labels works fine.
2. **Behaviour-cloning the search fails.** 95.9% action accuracy, yet the
   clone played 4–9 pts/hand WORSE than its expert, flat across DAgger rounds;
   confidence-gating trended monotonically to "just play the teacher." The
   expert's edge is rare, context-dependent deviations that look like label
   noise to a classifier. **Value regression, not policy classification, is
   the path** (see roadmap).
3. **Never argmax over noisy rollout values.** The first expert lost to its
   own rollout policy by −16/hand: per-world payoffs swing ±50 while true
   candidate gaps are 1–3 points, so argmax selects noise (the draw decision
   degrades to a coin flip, worth −11 alone). The fix that works: truncated
   rollouts + paired worlds + **teacher-prior with a z-SE deviation test**
   (deviate only on significant paired evidence; worst case = the prior).
4. **The z-test guards against noise, NOT bias.** Under full rules the
   raw-deadwood truncation leaf was lay-off-blind (biased); systematically bad
   deviations passed the test and the expert fell to −0.73 vs its own teacher.
   Fixed by a rules-aware leaf (effective deadwood via `defender_residual`).
   Any future leaf/value change must re-run the vs-teacher gate.
5. **Winner's curse discipline.** Never report the max of many noisy
   measurements: the "+2.35 best response" was the patient teacher measured 5
   times under different names (grid duplicates), scattering −0.54…+2.35.
   Selection and reporting seeds must differ; validate any "winner" fresh.
6. **The noise floor is ±1.5 pts/hand at 1000 hands** (per-hand std ≈ 25).
   Differences inside that are not findings. Budget hands accordingly.
7. **Range-balancing (mixing knock decisions to protect the non-knock range)
   does not measurably help here.** Classic rules: hand-reading adversary's
   edge identical vs p∈{0,½,1}. Full rules: monotone in p (+1.0 / +2.4 / +3.9)
   — LESS knocking is simply safer; mixing beats always-knock but never-knock
   beats both. The mixing infrastructure (`MixedKnock`, 0/50/100 grid) remains
   for re-testing under new scoring (box bonuses would change this).
8. **An MSE-optimal value net is NOT a good search leaf (2026-08-23).** The
   full roadmap-#1 pipeline was built and gated: V(perfect-info state) ≈
   E[final full-rules payoff | teacher-vs-teacher continuation], trained on
   3M hands / 20M states, val MSE +11.4% over a linear-scalars baseline
   (corr 0.38, no overfit; run 1 with only 300k hands overfit in ONE epoch —
   states within a hand share one noisy label). Inside the search it LOSES
   to the handcrafted eff-deadwood leaf: vs v3search (1500 hands each)
   K=192/T=6 −1.65±1.37, K=48/T=16 z=1.0 −1.56±1.45, z=0.75 −2.20±1.45
   (lower z monotonically worse → the net's extra deviations are bad),
   K=96/T=12 −0.48±1.44. Two causes: (a) determinized rollouts at T=16 SEE
   the sampled stock order — tactical clairvoyance no static evaluator can
   match; (b) true candidate gaps are 1–3 pts, exactly the scale outcome
   regression blurs (the value-side analogue of lesson #2). The net DID beat
   the hand leaf at short truncation (hand leaf at K=192/T=6: −0.71 vs
   teacher — its bias explodes there), so it's only useful if a
   short-T/huge-K regime ever wins. Infrastructure kept: `valuenet.py`,
   `value_gen.py`, `value_train.py`, `value_expert.py`, `results/v4_value/`.
9. **The paired design is robust to world-model realism (2026-08-23).**
   Marginal belief sampling produces structurally meldless opponent hands —
   measured: true hidden hands avg 23.3 deadwood, sampled 41.6
   (`corr_sample.py`). The fix exists and is calibrated
   (`belief.sample_free_cards(..., boost)`, sequential meld-adjacency boost;
   boost=2.5 matches true deadwood 23.1). But it does NOT cash out:
   v3boost vs v3search +1.19±1.34 on the 1500-hand selection run, then
   **+0.13 ± 0.77 on the 4488-hand fresh-seed confirmation** — a tie (and a
   live demonstration of lesson #5). Symmetric world bias cancels in paired
   candidate differences. v3boost vs teacher +2.04±1.41 (fine, ≈ v3search's
   +1.41±1.86). Registry members `v3boost`, `v3wide` exist for re-testing.
10. **Action-value distillation v1 fails on EV accuracy (2026-08-25).** The
    full HANDOFF_V5_TRAINING plan was executed and gated: 783k analyze()
    EV-labeled decisions (K=128, full rules — note exit_gen.py never set
    rules; qgen.py does), Q-net (2048-2048-1024) regressed on paired diffs,
    offline dev-ranking gate PASSED 0.726. All three online gates failed or
    tied: standalone greedy −5.43±1.41 vs teacher, best margin-hybrid tie
    −0.54±1.28; as rollout policy at equal wall-clock (xl-net K=16 ≙ v4's
    0.103 s/move) −2.6±1.45 vs v4, and the K=16 teacher-rollout ablation
    (−0.72±1.47) shows the net's rollouts themselves cost ~1.9 pts at equal
    K — the frontier is dominated (margin→∞ IS the ablation); as candidate
    pruner −0.60±1.48 (tie). ROOT CAUSE, measured: the net's diff RMSE is
    4.6 pts against true candidate gaps of 1–3 pts — any use that lets the
    net SELECT from its own Q selects its own noise (the value-side
    winner's curse; offline ranking of SEARCH-chosen deviations is a
    different conditional and stays consistent). Full record:
    `results/v5_q/v5_gates.log`. Do not retry until a net's diff RMSE
    approaches ~2 pts; untried levers: 2–3× data (val loss still
    descending), per-candidate conditioned architecture, obs-derived
    feature augmentation, K=192 labels, knocker roll-in opponents. KEPT:
    the dataset, qgen/qtrain/q_bot, and `q_expert.QRolloutExpert` — a
    mock-parity-proven lockstep searcher that batches ALL rollout worlds
    through one forward per step (reusable for any future rollout policy).
11. **v4 world-weight staleness quirk (found 2026-08-24 proving parity).**
    expert.py:200 computes each world's weight from `opp.known_cards` read
    off the scratch, which for world k+1 is still polluted by world k's
    rollout pickups (restore comes after the weight check) → sporadic
    over-rejection of knockable-opponent worlds. v4 was gated as-is, so
    nothing is invalidated; `QRolloutExpert(mock=True)` is the corrected
    searcher (measured vs v4, K=48: see results/v5_q/v5_gates.log tail).
    If expert.py is ever fixed, v4 must be re-gated.
12. **The search is insensitive to rollout-INTERIOR play quality
    (2026-08-26, user-hypothesis test).** Belief-aware rollout gameplay
    (`belief_teacher.py`: discard safety = E[opp-held meld-mates] from the
    belief; windowed variant trades deadwood for safety) ties everywhere:
    standalone vs teacher −0.13…+0.48 (windowed sweep all noise, heavy
    safety −1.48); in-search vs v4 at EQUAL WORLDS K=48 +0.08 ± 1.55.
    Three independent interior changes now all tie (bt +0.08, quirk-fix
    −0.98, even K=16 teacher rollouts only −0.72): paired candidate
    differences cancel symmetric play improvements. Interior changes only
    matter when they break payoff symmetry across candidates — rollout
    TERMINATION behavior (the opp knock model, +3–4) and LEAF scoring
    (lesson #4). v4's crude in-world play is a robustness feature, not a
    weak link; don't spend again here. A ~+0.4 vs-knockers standalone
    trend for windowed bt (6.21 vs 5.76 on smart, 3.38 vs 3.01 on knock10,
    same seeds) is real-looking but needs ~30k hands to prove — parked.
    Kept: `belief_teacher.py` (BeliefTeacher, BTSerialExpert — the
    corrected-v4 serial loop with a pluggable belief-aware hero, machinery
    parity proven; 1.23× wall-clock at K=48).
13. **What finally beat v4 (2026-08-26/27): reallocate compute, don't
    refine play.** After lessons #10/#12, three single-hand mechanisms
    were gated separately (same selection seed): adaptive staged budget
    alone +1.11 ± 1.50; + 2-ply close-call rollouts +2.26 ± 1.50
    (selection); endgame mode (no truncation + ALL-discard candidates)
    +0.26 alone and NEGATIVE combined (−0.42) — widening the candidate
    set multiplies the noisy-argmax comparisons the top-4 pruning exists
    to control. Adaptive+deep confirmed +1.20 ± 0.93 fresh-seed and beat
    the compute-matched plain-K80 control (+1.61 ± 1.55) — but the
    equal-wall-clock-tuned version (16/40/128, α=.75) is a TIE: the win
    is *better-allocated extra compute* (escalation to 192 worlds + exact
    2-ply verification exactly on the close calls). The
    sequential-testing discipline that made it safe: early stops only
    ever CONFIRM the prior; deviating requires highest-stage evidence.
    Endgame sub-lesson: precision upgrades must not widen the comparison
    surface.

## Engineering gotchas (environment-specific)

- **rlcard engine facts** (verified): play is deterministic after the deal;
  phase lives in `game.actions[-1]` + `move_sheet[-1]`; determinize by
  swapping `opp.hand` + shuffling the rest into `stock_pile` (pop from END),
  keep `known_cards`; `player.did_populate_hand()` rebuilds meld caches the
  judge reads (scratch games strip this — they must only use
  `fast_legal_actions`); gin is mandatory when available; knock/gin discards
  never join the pile; `move_sheet` is perfect-information (filter it —
  `DrawCardMove.card` and `move_sheet[0].shuffled_deck` are hidden info).
- **Windows spawn**: workers re-import modules fresh — population registries
  must be static module-level code; runtime registry mutations never reach
  workers. Pass `rules`/config through job tuples.
- **`id(round)` is reused by the allocator across hands** — stamp rounds with
  a token object (`rnd._v3_token`) to detect new hands, never key on `id()`.
- **Do not edit any imported .py while a background eval runs** — spawned
  workers import files lazily; two runs died this way.
- **scipy is unusable on this machine** (Application Control policy blocks its
  HiGHS DLL). `meta.solve_nash` is a dependency-free RM+/FP solver, validated
  on RPS and dominant-strategy games.
- **Card.__hash__ costs two list.index calls** — never put Cards in hot-path
  sets/dicts; work in card-id space (`sim.CARD_ID`, string-keyed).
- Torch: RTX 5080 needs the cu128 wheel; CPU inference in workers with
  `torch.set_num_threads(1)`. 16 logical cores ≈ 4–7 effective under full
  fan-out; chunk background jobs small so shards land incrementally.

## The roadmap to a bot stronger than any human

In leverage order. Every step gates on: (a) beats the previous best
head-to-head (full rules, ≥1500 hands, CI excluding 0); (b) stays unexploited
by `meta.probe_exploitability` (heuristic grid + search adversaries); (c)
selection seed ≠ reporting seed.

1. **~~Learned value network in the search~~ — DONE 2026-08-23, NEGATIVE.**
   Built exactly as prescribed (with one correction: the leaf evaluates
   perfect-info determinized states, so V's input is a perfect-info encoding,
   `valuenet.leaf_features`, not the 10×52 belief obs) and gated. It loses
   to the handcrafted leaf — see lesson #8 for the numbers and the two
   causes. Do NOT retry a static outcome-regression leaf. What might still
   work: a net predicting the T=16-rollout result (distilling the search's
   own evaluation to buy K, not accuracy), or value nets as best-response
   adversaries (roadmap #4).
2. **~~Correlated belief sampling~~ — DONE 2026-08-23, TIE.** Implemented and
   calibrated (`belief.sample_free_cards(..., boost)`, boost=2.5); the
   realism bias was real and large (23.3 vs 41.6 deadwood) but head-to-head
   it's a wash: +0.13 ± 0.77 confirmation — see lesson #9. Keep boost=2.5
   available; it may matter once something else changes (opponent-model
   conditioning, knock modeling, match scoring).
3. **Opponent-model conditioning — the big win of 2026-08-23.** The
   searcher was leaking ~4 pts/hand vs weak knockers (teacher +7.43 vs
   smart; every searcher variant ~+3.4). TWO fixes were needed, and only
   TOGETHER do they work (interaction: all same-seed 45100, 1488 hands):
   (a) the opp_model must shape the rollout CONTINUATION, not just the
   world weights — the modeled opponent seat now plays
   MixedKnock(t, p) in rollouts (expert.py); alone: +4.26.
   (b) correlated worlds (boost=2.5); alone: +3.55. The meldless marginal
   sampler almost never produced knockable hands, so the knock-model had
   nothing to bite on. COMBINED (`adv10_boost`): +7.59 selection, then
   **+6.59 ± 0.76 over 3984 fresh-seed hands — confirmed.** Safety: wrong
   model vs teacher costs nothing (−0.64 ± 1.51); head-to-head vs
   v3search: +2.49 ± 1.41 and **+1.59 ± 0.87 (3984 hands)** on two
   independent fresh seeds — CI excludes 0 both times. **PROMOTED as `v4`**
   (meta registry alias `v4`; `play_vs_bot.py v4`). Gate (b) probe
   (results/probe_v4_full_20260823.log): every KNOCKING adversary loses
   heavily to v4 (−2 to −6.65); the headline "+3.75 by br_t6_p0" is the
   lesson-#5 artifact — all six p0 grid rows are the SAME patient-teacher
   policy measured six times (+1.35/+0.41/+0.62/+3.75/−0.60/+1.51, pooled
   ≈ +1.0 ± 1.2). So v4's one candidate leak is ~1 pt/hand to a PURE
   never-knocker, where its fixed knocks-at-10 assumption is maximally
   wrong — REFUTED by the direct 4000-hand measurement: **v4 vs teacher
   +0.89 ± 0.94** (results/v4_value/v4_vs_teacher_4000.log). v4 beats or
   ties every member of the field: v3search +1.59, teacher +0.89 (par or
   better), smart +6.59, all knocking adversaries lose 2–6.65 to it.
   Promotion fully closed, both gates passed. Adaptive (t, p) estimation
   remains the next refinement (it should turn the teacher par into a win
   and generalize the knock model beyond the fixed (10, 1.0) prior).
   NEXT: make the model ADAPTIVE (estimate the opponent's (t, p) online
   from observed knocks/behavior instead of assuming it), then re-probe
   exploitability of the promoted config (gate b).
   Also tried and dead this session: wider top_discards=8 (v3wide,
   −0.88 ± 1.40 vs v3search). Still untried: 2–3-ply candidate trees,
   information-set MCTS.
4. **Keep the exploitability loop as referee.** Extend `meta.py` adversaries
   as the bot improves (value-net-based best responses are the natural next
   adversary class — the 20M-state dataset and trained net in
   `results/v4_value/` are a head start). The full-rules probe of the
   current bot is DONE (2026-08-23, unexploited — see Current strength).
5. ~~Match-style scoring~~ / 6. ~~Human evaluation~~ — **REMOVED
   FOREVER (user decision, 2026-08-31)**, along with adaptive (t,p)
   opponent-knock estimation. The project goal is THE GTO BOT FOR
   SINGLE-HAND PLAY: unexploitable equilibrium play maximizing
   per-hand point differential under the established rules (full
   scoring, standard dead-hand, home variant). No opponent
   modeling/adaptation, no match/box bonuses, no human-eval track.
   Do not re-propose these.

### Suggested next session

NOTE 2026-08-27: v5 (adaptive.py) is promoted — see "Current strength".
Items below predate it. Fresh priorities: (a) scaling curve — does
adaptive+deep keep growing with budget (stages 16/48/384? kbranch 4?),
i.e., is v5 the start of an anytime-strength curve v4 never had;
(b) 3-ply on the final stage; (c) endgame mode RETRY with the candidate
set kept at top-4 (the all-discard widening is what hurt, lesson #13);
(d) ~~match-style scoring then human eval~~ (REMOVED 2026-08-31 — GTO
single-hand goal, see roadmap #5/6). The v5-net
distillation lessons (#10) stand: don't retrain nets until RMSE ~2 pts.

1. Re-run `sim.py --verify`, `fullrules.py --test` (~2 min) — confirm nothing
   rotted (last green 2026-08-23, after the boost-sampler edits).
2. Check `results/v4_value/gate_wide.log` — the v3wide (top_discards=8)
   selection run vs v3search/teacher may have finished after handoff. If
   positive, confirm on a fresh seed at ≥4000 hands before believing it
   (lesson #5; the v3boost selection run's +1.19 collapsed to +0.13).
3. ~~Adaptive opponent model~~ — REMOVED FOREVER (user decision,
   2026-08-31): no opponent modeling/adaptation; the goal is the GTO
   single-hand bot. (In the home variant the threshold is public
   anyway.)
4. 2–3-ply search-in-search: real search on the first few rollout
   decisions before handing to the teacher (~5–10× cost, no training).
5. **~~Action-value distillation~~ — DONE 2026-08-25, NEGATIVE.** Executed
   exactly per `HANDOFF_V5_TRAINING.md`; all three online gates failed or
   tied — see lesson #10 for the numbers, mechanism (net diff RMSE 4.6 pts
   vs 1–3 pt true gaps → self-selection of noise), and the untried levers.
   Infrastructure + 783k-decision dataset kept in `results/v5_q/` and
   qgen/qtrain/q_bot/q_expert. Original plan (for reference): generate
   (position → per-candidate
   EV table) labels from `RolloutExpert.analyze()` at scale and train a
   Q-net by REGRESSION on candidate values — dense signal, unlike the
   argmax-label cloning that failed (the search's edge is rare deviations
   ≈ label noise to a classifier). Gate three ways: standalone vs teacher;
   as rollout policy at EQUAL WALL-CLOCK (nets cost ~1000× a teacher call —
   it must win per-compute, not per-call); as candidate pruner. If it
   clears gates, iterate the ExIt loop, gating every round.
6. ~~Match-style scoring / human eval~~ — REMOVED FOREVER (see #5/6).
