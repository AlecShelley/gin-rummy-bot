# V6 Handoff — the four-track plan (exact endgames, Bayes-correct worlds)

Standard gin rummy, FULL RULES (lay-offs, undercut after lay-offs, dead
hands 0-0 — `fullrules.py`). Read `CLAUDE.md` first for the project-wide
picture and the thirteen numbered lessons; this document is the focused
plan for the next improvement campaign. User decisions baked in: only
SINGLE-HAND strength matters (no across-hands opponent adaptation), and
the four tracks below were explicitly requested, in this priority order.

## Where the bot stands (2026-08-27)

**v5 is the promoted champion** (`adaptive.py`, registry `v5`,
`play_vs_bot.py v5`). Architecture: the quirk-corrected serial search loop
with (a) an ADAPTIVE STAGED WORLD BUDGET — 16-world screen, escalate
near-significant decisions (mean > 0.5·z·se) to 48 then 192 worlds,
deviate from the teacher prior only on the highest-stage evidence (early
stops only ever CONFIRM the prior, so sequential looks stay conservative);
and (b) 2-PLY CLOSE-CALL ROLLOUTS — at the final stage, each world
branches the hero's first rollout decision over the top-3 pruned moves and
takes the exact (deterministic-within-world) max. Config: T=16, z=1.5,
opp_model (10,1.0), boost 2.5, mid_alpha 0.5, kbranch 3.

Gated numbers (full rules, seat-swapped, selection seed ≠ confirm seed;
`results/v5_q/v5_gates.log` is the full record):

| v5 vs | pts/hand |
|---|---:|
| v4 (prior champion) | **+1.20 ± 0.93** (3984 fresh) |
| v4 at K=80 (compute-matched control) | +1.61 ± 1.55 |
| teacher | −0.12 ± 0.96 (3984, tie) |
| smart | +6.83 ± 1.27 |
| best of 21 probe adversaries | +0.51 ± 3.11 → **unexploited** |

Cost: ~1.7× v4 (~0.16 s/searched-move idle; v4 = 0.103 s/move ≈ 2.1
s/hand — note an older doc said "2.5 s/move"; it is per HAND). The
equal-wall-clock-TUNED variant (stages 16/40/128, mid_alpha .75) TIES v4
(+0.08 ± 0.93): v5's edge is better-ALLOCATED extra compute, proven
architectural by the K=80 control.

## What was tried and is DEAD — do not relearn (details: lessons #10-13)

1. **Q-net action-value distillation** (lesson #10): 783k K=128 EV-labeled
   decisions, net diff-RMSE 4.6 pts vs true candidate gaps of 1–3 pts →
   every use that lets a net SELECT from its own Q selects its own noise
   (standalone −5.43; rollout policy −2.6 with ablation isolating −1.9 to
   the net itself; pruner tie). Bar for retry: RMSE ≈ 2 pts. Dataset +
   `qgen/qtrain/q_bot/q_expert.py` kept; `q_expert.QRolloutExpert` is a
   parity-proven LOCKSTEP searcher that batches all K worlds through one
   forward per rollout step — reuse it for any future batched policy.
2. **Belief-aware rollout play** (lesson #12): hand-reading discard safety
   (and a deadwood-for-safety windowed variant) ties standalone AND
   in-search at equal worlds. General principle, triple-measured: the
   paired design CANCELS symmetric rollout-play improvements; interior
   changes matter only when they break payoff symmetry across candidates
   (termination behavior, leaf scoring, candidate-conditioned depth).
3. **Endgame mode v1**: no-truncation + ALL-discard root candidates was
   NEGATIVE combined (−0.42) — widening the ROOT comparison surface
   re-invites the noisy argmax the top-4 pruning exists to control. The
   safe home for exactness is INSIDE worlds (deterministic max), which is
   why 2-ply worked. This asymmetry is the design key for track 2.
4. **v4 world-weight staleness quirk** (lesson #11, expert.py:200): reads
   post-rollout known_cards for world weights. Harmless (fix ties); all
   new searchers use the corrected loop. Don't edit expert.py.

## FIRST TASK: prior-art survey (user-requested)

Before building, spend a session-chunk on literature/engineering search
(WebSearch) and write findings into `results/v6_prior_art.md`. Targets:

- **PIMC / determinized search pathologies**: strategy fusion and
  non-locality (Frank & Basin); how Skat and Bridge bots mitigate them.
  v5 is PIMC-family; tracks 2/4 walk straight into these — know them.
- **Inference-weighted sampling in trick/card games**: Buro et al.'s Skat
  players (card inference from bids/plays via policy likelihoods) — the
  closest published relative of track 1; also Scrabble opponent-rack
  inference (Richards & Amir), Mahjong Suphx's hand prediction.
- **ISMCTS** (Cowling/Powley/Whitehouse) — information-set search as the
  alternative to per-world determinization; single-observer vs
  multi-observer variants and when they beat PIMC.
- **Poker subgame solving with ranges** (Libratus/DeepStack papers:
  safe/unsafe subgame solving, re-solving) — the theory basis for track 4
  and for what "exact endgame" can legitimately claim under a belief.
- **Gin rummy specifically**: the EAAI-21 gin rummy challenge line —
  confirm nothing there beats what we have; any published hand-estimation
  tricks worth stealing.
- Known negative patterns to respect: perfect-info rollout "clairvoyance"
  (our lesson #8 matches the literature), paranoia when maxing opponent
  replies inside determinized worlds (relevant to track 3's opponent
  branching).

## The four tracks (user-approved order)

### Track 1 — Particle-filter belief: Bayes rule + good play (DO FIRST)

Replace the heuristic multiplicative belief with trajectory-level
inference: N particles (few hundred), each a hypothesis of the opponent's
full hand trajectory (including their hidden stock draws). On every public
event — their discard, pickup, declined upcard — update each particle and
weight it by P(observed action | particle hand, opponent model), where the
model is a SOFTENED teacher (softmax over teacher move-scores, temperature
τ — a brittle 0/1 likelihood would deplete the filter). A discard then
carries its full Bayesian content ("that was their least useful card ⇒
evidence about everything they KEPT"), compounding over the history —
exactly the poker range update. Resample on weight degeneracy.

- FastTeacher calls are µs-scale: per-event per-particle likelihoods are
  affordable. Design the softened-policy scorer once; reuse it in rollouts
  as a realism upgrade for the opponent seat (continuation realism paid
  +3-4 before — lesson in CLAUDE.md roadmap #3).
- Retires the boost=2.5 hack (it patches marginal-independence; particles
  fix the cause). Re-gate anything that assumed it.
- **OFFLINE GATE FIRST (cheap, decisive)**: `belief.py --calibrate`
  already logs ground-truth streams and scores Brier vs uniform; current
  belief = +12.5% over uniform held-out. The particle filter must beat the
  current belief's Brier CLEARLY before touching the searcher. Then
  in-search: swap `sample_free_cards` for particle draws (the searcher
  samples worlds FROM the particle set), gate vs v5 (1500 selection /
  4000 fresh confirm), probe on promotion.
- Watch: particle depletion under model mismatch (opponent ≠ teacher-like:
  v2 nets, random) — the calibration harness already logs vs teacher AND
  clone opponents; add a random-opponent stream. Temperature τ is the
  robustness knob; consider mixing a uniform-policy component.

### Track 2 — In-world exact endgame (safe by the lesson-#13 mechanism)

All exactness lives INSIDE worlds; root candidates stay top-4.

1. **Full-depth k-best world search**: stock ≤ ~6-8 → replace truncated
   rollouts with real search of the determinized world (both seats branch
   top-k=3 teacher-pruned moves to terminal; depth ≈ 2×stock plies;
   memoize). Nails undercut/lay-off arithmetic. Start as a v5 final-stage
   replacement in endgame positions.
2. **World enumeration**: when C(|unseen|, hidden) is small (stock ≤ 2-3),
   enumerate all opponent hands weighted by the belief instead of
   sampling — the world-sampling noise term vanishes.
3. **Belief-exact knock anticipation**: enumerate knock-capable plausible
   hands → exact P(opponent knocks next turn) replacing the (10,1.0)
   step. Feeds both the world weights and rollout termination.

Gate each increment vs v5 (endgame changes are low-frequency: expect small
means — budget hands accordingly, the noise floor is ±1.5/1000 hands;
consider gating on endgame-reached hands only for power).

### Track 3 — Escalation-ladder extensions (proven vein, incremental)

3-ply on the final stage; kbranch 4; a 4th stage where the 2-3 surviving
candidates get a small NESTED search as their continuation. Also one
careful test of branching the OPPONENT's first in-world reply — chosen
under their information set, not the world (paranoia risk — see prior-art
survey). Each rung gates vs current-best at measured wall-clock with a
compute-matched control (the v4-K80 pattern) — "scaling curve" question:
does adaptive+deep keep growing with budget where plain-K v4 didn't?

### Track 4 — Subgame CFR endgame solving (capstone; after track 1)

When the endgame is reached (stock ≤ ~3-4), build the actual
imperfect-information subgame with the PARTICLE posterior as the
opponent's range and solve it with `meta.solve_nash` (dependency-free
RM+/FP, validated). This is the only track producing game-theoretically
exact endgame play rather than exact-given-a-point-belief. Read the safe
subgame-solving literature FIRST (survey above) — naive unsafe re-solving
can be exploitable. Compare vs track-2's k-best worlds head-to-head; keep
whichever wins per compute.

## Infrastructure map (what exists, where)

| File | What you need from it |
|---|---|
| `adaptive.py` | **v5.** AdaptiveExpert (staged budget + deep), `_AdaptivePolicy`, amatch harness with `v4k<N>` compute-control opponents, `--parity` (bit-exact vs corrected reference), `--time`. |
| `expert.py` | v4's RolloutExpert (frozen; has the lesson-#11 quirk — don't edit). `analyze()` EV tables; `_candidates` top-4+knock pruning. |
| `belief.py` | Current heuristic Belief + `build_obs` + **`--calibrate` (the Brier ground-truth harness — track 1's offline gate)**. |
| `belief_teacher.py` | BeliefTeacher (belief-aware FastTeacher variant), BTSerialExpert (clean corrected serial loop to copy), `clone_belief`. |
| `q_expert.py` | Lockstep batched-rollout searcher (parity-proven), `_RefExpert` (corrected-v4 reference), `_decision_stream` parity pattern, `_time_expert`. |
| `sim.py` | Kernel: `fast_legal_actions`, `playout`, snapshot/restore/`apply_world`, FastTeacher, MixedKnock, `set_rules`. `--verify` gate. 536 rollouts/s/core. |
| `fullrules.py` | Exact full-rules scoring (`--test` gate). `defender_residual` = the rules-aware leaf. |
| `meta.py` | Static registry (v5/v4/teacher/smart/grids), `match`, `probe_exploitability`, **`solve_nash` (track 4's solver)**. |
| `evaluate.py` / `viz.py` | Position evaluator (user-facing) — update to optionally use v5/particles once promoted. |
| `results/v5_q/v5_gates.log` | Complete evidence trail of the v5 campaign. |
| `results/v5_q/gen_*.npz` | 783k search-labeled decisions (K=128) — reusable for any future net. |

## Discipline (non-negotiable; violations were caught twice historically)

- Noise floor ±1.5 pts/hand at 1000 hands. Selection seed ≠ confirmation
  seed; every winner confirms fresh at ≥4000 hands. Never report a max of
  many measurements as a finding (lesson #5).
- Cost changes require a compute-matched control (the v4-K80 pattern).
- New searchers prove machinery parity bit-exact against `_RefExpert`
  (via `_decision_stream`) before any match runs.
- Windows spawn: workers re-import; `sim.set_rules(rules)` first act of
  every worker; config as plain dicts/tuples in job tuples; static
  registries. Never edit any imported .py while a background run lives.
- PowerShell mangles inline `-c` quotes → script files; read
  PowerShell-written files with utf-8-sig. scipy unusable (HiGHS blocked).
- Promotion = beats-or-ties every member + fresh-seed confirm +
  `probe_exploitability(rules="full")` clean.

## First-session checklist

1. `sim.py --verify`, `fullrules.py --test` (~2 min, confirm nothing
   rotted; last green 2026-08-27), `adaptive.py --parity`.
2. Prior-art survey → `results/v6_prior_art.md` (see FIRST TASK).
3. Track 1: softened-teacher likelihood scorer + particle filter;
   offline Brier gate vs the current belief (harness exists); only then
   the in-search swap and match gates vs v5.
4. Tracks 2-4 in order, each gated, each logged in
   `results/v5_q/v5_gates.log`-style appendices (start
   `results/v6_gates.log`).
