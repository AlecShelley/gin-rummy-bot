# V5 Handoff — training the next-generation bot (action-value distillation)

Standard gin rummy, FULL RULES (lay-offs, undercut after lay-offs, dead
hands 0-0 — `fullrules.py`). The home variant (`home_variant.py`) is out of
scope here. Read `CLAUDE.md` first for the project-wide picture; this
document is the focused plan for the training track.

## The objective, precisely

Train a network that improves the **search**, not one that replaces it.
The concrete target: a fast policy/Q network that, plugged into the
rollout layer of the v4 searcher, beats the current v4 **at equal
wall-clock per move**. Secondary prizes: a strong standalone millisecond
bot, and network-driven adversaries for the exploitability referee.

## What you are building on (all measured, full rules, seat-swapped)

- **v4 = `meta.population()["v4"]`** — belief-state rollout searcher
  (`expert.py`), K=48 worlds, T=16 truncation, z=1.5 teacher-prior
  deviation test, opponent model (10, 1.0) driving BOTH world weights and
  the rollout continuation, meld-correlated world sampling (boost 2.5).
  Beats v3search +1.59 ± 0.87 (3984 hands), teacher +0.89 ± 0.94 (3984),
  smart +6.59 ± 0.76 (3984); unexploited by 21 dedicated adversaries.
- **FastTeacher (`sim.py`)** — the hand-tuned heuristic that (a) is the
  search's prior, (b) plays every rollout, (c) answers in microseconds.
  This is the bar your network must clear *per unit of compute*.
- Per-hand point differential is the objective. Noise floor: ±1.5 pts/hand
  at 1000 hands (per-hand std ≈ 25). Budget hands accordingly.

## Why the two previous training attempts failed — do NOT repeat them

1. **Policy cloning (v3 era, `exit_gen.py`/`exit_train.py`)**: 95.9%
   action agreement with the search, yet played 4–9 pts/hand WORSE than
   it — worse than the teacher. The search's edge over its prior is rare,
   context-dependent deviations (~4% of decisions); to a classifier
   trained on argmax labels they are indistinguishable from label noise
   and get smoothed away.
2. **Outcome-regression value leaf (2026-08-23, `valuenet.py`,
   `value_train.py`, `value_expert.py`, 20M states in
   `results/v4_value/`)**: beat every offline baseline (+11.4% val MSE),
   LOST inside the search (−1.65 ± 1.37 vs v3search at its best config;
   lower z monotonically worse). Causes: (a) true candidate gaps are 1–3
   pts — exactly the scale a smoothed regressor blurs; (b) determinized
   rollouts SEE the sampled stock order — lookahead no static evaluator
   matches.
3. Related must-know lessons in CLAUDE.md: never argmax noisy rollout
   values (#3); z guards noise not bias (#4); winner's-curse discipline —
   selection seed ≠ reporting seed, confirm every "winner" fresh at ≥4000
   hands (#5, violated twice this project, caught both times).

## The plan: regress the search's per-candidate EVs

The fix for failure #1: train on **dense value targets, not argmax
labels**. `RolloutExpert.analyze(game, belief)` (expert.py) returns, at
every searched decision: `candidates` (ActionEvents), `payoff_means`
(hero-view EV per candidate, payoff scale = points/100), `se`, `t_idx`,
`worlds`. That table is the label.

### Step 1 — data generation (~1 day of background compute)

Adapt `exit_gen.py` (it already does parallel search-labeled generation;
swap the label from action-id to the analyze() table):

- Roll-in mix as measured necessary in v2: self 40%, strong 20%, v2-clone
  20%, random 20% (`SHARES`). Hero decisions labeled by a v4-config
  expert: `RolloutExpert(worlds=128..192, truncate=16, z=1.5,
  opp_model=<per opponent>, sample_boost=2.5)`. K=48 labels have ±2–3 pt
  SE per candidate; K=128–192 halves that — fewer hands, cleaner labels
  is the right trade here (regression averages residual noise).
- Input = `belief.build_obs` (10, 52) — the IMPERFECT-info hero view (the
  network must run from a player's perspective; never leak the sampled
  world). `build_obs` is the single source of truth: generation and every
  deployment path must call it (bit-exactness gate GA1).
- Store per decision: obs float16, candidate action-ids int16, candidate
  EVs float32, se float32, worlds int16. Shard as compressed npz, SMALL
  chunks (long chunks measured badly — stragglers, nothing on disk).
- Windows spawn: workers re-import modules; pass config through job
  tuples; call `sim.set_rules("full")` in every worker entry point.

### Step 2 — the network

- Input 10×52 = 520 (+ optional flatten of scalars plane as-is). Output:
  a 110-dim action-value head (rlcard action space), MSE on the labeled
  candidates only (mask everything else). Start MLP 512-512-256; don't
  over-engineer before the first gate.
- Train/val split BY SHARD (states within a hand share labels; run-1 of
  the value net overfit in ONE epoch on 300k hands — you want millions of
  independent hands, sparse per-hand recording, dropout + weight decay +
  early stopping; see `value_train.py` for the working recipe).
- Offline gate before anything else: on held-out states, does the net's
  candidate RANKING match the search's, and is its EV-gap RMSE below the
  label SE? If ranking accuracy on deviation states (where the search
  left its prior) is poor, stop and rethink — that is exactly where the
  value lives.

### Step 3 — the three online gates (in order, cheap → expensive)

All: full rules, seat-swapped, ≥1500 hands selection / ≥4000 fresh-seed
confirmation, CI excluding 0.

1. **Standalone**: net plays greedily (argmax masked Q). Must beat
   teacher head-to-head. (The v3 clone failed this — it's a real bar.)
2. **Rollout policy at equal wall-clock**: replace FastTeacher in
   `RolloutExpert.rollout` with the net (batch all worlds' rollout
   decisions per step through one forward pass — 48–192 states per call,
   CPU or CUDA; `torch.set_num_threads(1)` in workers). Then re-tune K/T
   so wall-clock per move matches current v4 (~2.5 s), and gate vs v4.
   THIS is the deliverable. Note the v4 lesson: continuation REALISM beat
   continuation strength (+3 pts/hand from making rollout opponents
   actually knock) — keep the opponent seat's knock model wrapped AROUND
   the net (MixedKnock-style) rather than hoping the net learns it.
3. **Candidate proposer/pruner**: use the net to pick/order
   `top_discards` instead of raw deadwood ranking. Cheap add-on gate.

### Step 4 — if gates pass, iterate (ExIt loop)

Stronger search → regenerate labels with it → retrain → re-gate. Gate
EVERY iteration vs the previous best AND vs teacher; run
`meta.probe_exploitability` (full rules) on any promoted bot. If an
iteration doesn't clearly pass, stop the loop — it degrades silently.

## Infrastructure map

| File | What you need from it |
|---|---|
| `expert.py` | `RolloutExpert.analyze()` — the label source. `decide()` is a thin wrapper; decision parity with the pre-refactor bot proven by `refactor_check.py`. |
| `exit_gen.py` | Parallel generation skeleton (roll-in mix, sharding, worker plumbing) — swap labels. |
| `belief.py` | `Belief`, `build_obs` (the network input), `sample_free_cards(boost)`; boost="auto" ramp lives in expert.py. |
| `sim.py` | Kernel: `fast_legal_actions`, `playout`, `FastTeacher`, `set_rules`. `--verify --bench --stress` gates. |
| `value_train.py` | Working GPU training recipe (shard split, uint8 memory trick, dropout/early-stop) — adapt the head/loss. |
| `meta.py` | Matches, population registry (STATIC — Windows spawn), `probe_exploitability`, Nash. `match(a, b, hands, workers, pool, seed, rules="full")`. |
| `evaluate.py`/`viz.py` | Position evaluator (user-facing) — also handy for eyeballing what the net disagrees with the search about. |

## Engineering gotchas (cost hours if forgotten)

- **Never edit any imported .py while a background eval runs** — spawned
  workers lazily re-import; two runs died this way.
- Windows spawn: static registries, config via job tuples, fresh
  `set_rules` per worker.
- torch: cu128 wheel for the RTX 5080; CPU inference in workers with 1
  thread; 16 logical cores ≈ 4–7 effective under fan-out; chunk small.
- Card objects: work in id space (`sim.CARD_ID`); `Card.__hash__` is slow.
- `id(round)` is allocator-reused — key hands on a token object.
- scipy unusable (HiGHS DLL blocked); `meta.solve_nash` is dependency-free.
- PowerShell mangles inline `-c` double quotes — use script files.
- Read npz/position files written by PowerShell with `utf-8-sig`.

## First session checklist

1. `sim.py --verify` + `fullrules.py --test` (2 min) — confirm nothing rotted.
2. Wire analyze()-labeled generation (half a day incl. a 2k-hand smoke
   shard; verify labels: teacher candidate's EV should track the printed
   evaluator output on identical positions).
3. Launch the big generation overnight; train + offline gate next.
4. Only then touch the online gates, in order.
