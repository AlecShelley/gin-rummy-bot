# V8 Handoff — v8home, the measured early game, and the road ahead

> Written 2026-09-04 for transfer to a fresh bot/session. Consolidates
> the 2026-08-31 → 09-03 campaigns (v7 likelihood promotion, the home
> opening rule, the opening/turn-2 theory program). Read `CLAUDE.md`
> for project history and the numbered hard-won lessons. Evidence for
> every claim below: `results/v8_gates.log` (the campaign log —
> read it end to end before changing anything). Prior handoffs:
> `HANDOFF_V7.md` (v6→v7 era), `results/v6_prior_art.md` (survey).

## THE GAME (binding user decisions)

**DEFAULT GAME = THE HOME VARIANT (user decision 2026-09-04). All new
work targets v8home and home rules unless the user says otherwise.**
The standard rlcard-style game and its champion (v7) are legacy —
kept for regression gates and history, not a development target.

Home rules, all binding:
1. 10+10 deal; one card FLIPPED founds the pile; the hand's knock
   threshold = the flip's deadwood value (A=1 … 10/J/Q/K=10).
   `sim.new_game_home`.
2. **Opening rule** (2026-09-01): P1 may take the flip; if declined,
   P2 may take it (out of turn — play then returns to P1); if both
   decline, P1 MUST draw from stock. Passes move no cards and are
   public decline evidence in both beliefs. Implemented in sim's
   `_home_opening` state machine (`HomePassMove`/`HomePassAction`;
   snapshot/restore-safe; gate: `home_variant.py --test-opening`).
3. Full scoring (2026-08-22): lay-offs onto the knocker's melds,
   undercut +25 AFTER lay-offs, no lay-offs against gin
   (`fullrules.py --test`). Standard dead-hand rule (2026-08-30):
   stock ≤ 2 ⇒ knock/gin on the discard or dead hand 0-0 (hardwired
   in `sim.fast_legal_actions`; --verify tolerates ~13 judge diffs).
4. Per-hand point differential is the objective. **GTO ONLY (user
   decision 2026-08-31, permanent): the goal is the unexploitable
   single-hand bot. Match/box scoring, human-eval tracks, and
   adaptive opponent modeling are REMOVED FOREVER — do not
   re-propose.** Exploitability probes are the referee; the oracle
   gap is the compass.

## THE CHAMPION: v8home

`home_variant.HomeV8Policy` (HomeV6Policy = compat alias);
`home_variant.py --bot v8home` (match harness), `--test-opening`
(rule gate). Composition:

- **v7 search + belief**: AdaptiveExpert staged search (16/48/192 +
  deep 2-ply, z=1.5) on particle-filter worlds (`particle.py`,
  `pf_expert.py`), with the 2026-08-31 likelihood redesign
  (binary-rule pickup model + meld tier, tau_p=0.1 mismatch rate,
  mh_all_takes rejuvenation — fixed the hidden-5♠ inversion; promoted
  as v7, registry `v7`; registry `v6` frozen at its gated config).
  Knock model bound per hand to the public flip threshold. The PF
  reads opening actions with option-aware book likelihoods.
- **The opening book** (`sim.flip_book_take`, played directly — no
  search at flip-option nodes): **P1 takes iff the flip melds; P2
  takes iff it melds OR adjacency ≥ 2** (rank-mates + same-suit
  within 2). Evidence: 6000-deal 3-arm study + z=4.9 (ace-band
  P1) and z=4.0 (P2 clause) paired confirmations.
- **The turn-2 book** (`sim.t2_book_take`, at the hand's FIRST free
  draw decision): **take iff it melds, OR it is a low card (≤3) with
  adjacency ≥ 2.** Evidence: non-melding takes there cost −1.92 ±
  1.65 each (2376 forced-arm cell-deals); the clause gated **+1.84 ±
  1.78** head-to-head (the program's first significant match win
  since v4); the low-card exception measured **+3.58 ± 2.41** under
  the naturally-conditioned distribution (see METHODOLOGY), strongest
  for weak hands (outs 0–2: +11.0 ± 7.8). Fires ~1/15,000 hands —
  adopted on causal evidence (no match gate can see it).
- Later turns: the search decides (turn-3+ bias bounded below
  measurement; early-game program CLOSED).

The evaluator (`evaluate.py`/`viz.py`) is in lockstep (standing
rule): particle-ensemble belief, opening/turn-2 book recommendations
with reasons, `my-pass`/`opp-pass` events, legacy-transcript pass
inference, BORDERLINE verdict honesty. `explain.py` = full-playout
outcome explanations.

## Measured early-game theory (the campaign's crown)

- Gin dominates the home game: 69% of v7-self-play hands end in gin
  (92% at ace flips → 57% at ten flips — deliberate patience, not
  forced: a knock-happy baseline knocks 98% at ten flips).
- Points are bimodal (lobes ±25-35, σ≈33, mean≈0); decisions work by
  shifting race-win probability between lobes, not by saving
  deadwood. **Deadwood only matters in the losing branch.**
- The flip: taking a melding flip ≈ +10 (P1) / +12-19 (P2); every
  non-melding class is tie-or-negative for P1; P2's adj≥2 exception
  works because P2's take is an EXTRA TURN (the same take is −5.5
  for P1 and +9.1 for P2 in ace games). Tempo alone (adj<2) is worth
  nothing. Big deadwood dumps are the WORST takes.
- The per-hand opening SOLVER (full-playout calculator, book as
  prior) validated offline but LOST its gate (−0.53 ± 0.62): the
  book is the pooled ground-truth estimator, within ~0.3 pts/hand of
  optimal; per-hand K=160 solves add noise + model error. Kept
  (`opening_solver.py`) for retry with a truer continuation model.

## METHODOLOGY (the reusable instrument — use this ladder)

For any "when is action X correct?" question, in order of authority:
1. **Pooled forced-arm study with REAL-BOT continuations** on natural
   deals (opening_study.py / turn2_study.py pattern): fork the
   decision, play both arms to the END of the hand with v8home both
   seats, paired per deal. This is what beats the bot's own search
   (its T=16 deadwood leaf misprices gin races).
2. **Constructed-state factor maps** (scenario_probe.py) decompose
   WHICH features carry an effect — but constructed-cell EV ≠
   natural-play EV (the broad t2 exception measured +1 constructed,
   tied in situ: selection + over-firing).
3. **Naturally-conditioned rejection sampling** (nat2_study.py, the
   user's design — the sharpest tool): keep only states where the
   opponent CHOSE the triggering action on v8's own line (two-tier:
   ~5ms teacher-line screen over millions of deals, then v8 verify),
   record hand strength (outs = unseen cards that meld into hand).
4. Gate every rule change in situ (paired deal match); adopt on
   ≥ tie only when causal evidence is from the same distribution.

**Harness rules (user directives, non-negotiable):** shard long runs
small (`imap_unordered`, 10-20 items), append rows to JSONL as shards
land, print progress + ETA per shard. Budget wall-clock with the
~2.2× fan-out penalty (16 logical ≈ 5 effective cores under 12-way).
Never launch with a trailing `&` inside a wrapped command (two runs
died / one duplicate-wrote this way — dedupe rows if in doubt).
Never edit imported .py files while spawned workers run.

## Discipline (unchanged, non-negotiable)

Noise floor ±1.5/1000 hands (σ≈25-33/hand); selection seed ≠
confirmation seed; winners confirm fresh; never report a max of noisy
measurements; machinery parity before match runs. Promotion = gates
green + evaluator lockstep + example library re-verified. Windows
spawn: pass config through job tuples (module globals don't reach
workers). All numbered lessons in CLAUDE.md still bind (esp. #3
noisy-argmax, #5 winner's curse, #9 symmetric-belief cancellation,
#12 rollout-interior insensitivity).

## Infrastructure map (v8 additions)

| File | What it is |
|---|---|
| `home_variant.py` | HomeV8Policy (the champion) + match harness + `--test-opening` gate. |
| `sim.py` | + `_home_opening` state machine, `flip_book_take`, `t2_book_take`, HomePassMove/Action. |
| `opening_study.py` / `turn2_study.py` | Forced-arm pooled studies (the truth-makers). |
| `nat2_study.py` | Naturally-conditioned rejection sampler (two-tier screen). |
| `scenario_probe.py` | Constructed-state factor maps. |
| `opening_solver.py` | Per-hand full-playout opening calculator (not adopted; --validate/--gate). |
| `book_gate.py` / `t2_gate.py` / `t2x_gate.py` / `p2adj_gate.py` / `ace_probe.py` | The paired gates (patterns to copy). |
| `home_mc.py` | The self-play MC study (outcome stats, histograms; rows reusable). |
| `results/v8_gates.log` | THE campaign log — every number above, in order. |
| `results/*_rows.json*` | Raw per-deal rows for every study — re-slice, never re-run. |

## Open frontiers (in leverage order)

1. **The oracle gap: ~14 pts/hand** (`oracle_gate.py`, +14.43 ± 1.52
   vs v7) — perfect opponent-hand knowledge. It lives in MID-GAME
   hand-reading and the ENDGAME now that the early game is measured.
   Track A (subgame CFR endgame solving, design rules in
   results/v6_prior_art.md §3) is the standing structural direction.
   NOTE: re-run the oracle gate on v8home under home rules to get
   the home-specific compass before spending.
2. **Mid-game take/discard audit** by the methodology ladder above —
   the turn-2 leak existed; turns 3-6 discard SAFETY (feeding the
   opponent) has never been audited with forced arms.
3. **Opening-solver retry** conditions: a true v8-quality
   continuation model (~100× cost/node) or K≥400 + z≥2.5.
4. Particle likelihood: discard-side energy is still a softened
   heuristic; the take-side redesign paid — the discard side may too.

## First-session checklist

1. `sim.py --verify`, `fullrules.py --test`, `adaptive.py --parity`,
   `ladder.py --parity`, `particle.py --anchors`,
   `home_variant.py --test-opening` (~5 min; all must be green).
2. Read `results/v8_gates.log` tail (the last three entries are the
   turn-2 program).
3. Default all work to the HOME variant / v8home.
4. Log new evidence in `results/v8_gates.log` (append) or open
   `results/v9_gates.log` when a new campaign starts.

## ADDENDUM 2026-09-05 — leak hunt closed (see results/v9_gates.log)

Frontier #2 (mid-game take/discard audit) is DONE and CLEAN, and
so is knock timing: forced-arm studies (turn3_study.py census-then-
fork pattern; knock_study.py; discard_study.py; 10,000+ forks, 0
replay mismatches) found NO systematic misplay on v8home's line —
j>=2 takes clean (-0.80 +/- 2.36 confirmed), knocks +5.60 +/- 2.45
when chosen / declines avoid -10.42 +/- 1.40, safety-first
discarding LOSES -3.47 +/- 1.65. The adjacency/outs take rule
(user hypothesis) tested negative at every registered cell.
Frontier #1 (oracle gap ~14, information not decisions) is by
elimination the whole game now. Registration-first discipline
worked; copy it (v9_gates.log) for the next campaign.

## ADDENDUM 2026-09-07 — V10 PROMOTED: champion is now v10home

Campaign log: `results/v10_gates.log` (read it end to end; the
registration-first pattern continues). Chain since this handoff:
posterior ground-truth check (v9 log tail; belief under-confident,
89/69/60% info capture) -> likelihood sharpening -> PROMOTION.
DEFAULT_PF += tau_po=0.0 (opening events are book-deterministic;
mismatch zero) + eps 0.05->0.02. `home_variant.py --bot v10home`
(default); `--bot v8home` pins the pre-v10 belief; registries v6
AND v7 pin eps=0.05/tau_po=None (partial-pin drift hazard — see
particle.DEFAULT_PF comment). posterior_check.py + six frozen
scenario row files = standing posterior regression benchmarks
(re-slice offline; never re-run). Open frontiers reordered: (1)
discard-side behavior model of the SEARCH (all four simple fixes
measured negative — post-pickup vs post-draw discard behavior
differs structurally); (2) S5-type cloud over-melding (~10 dw);
(3) mid-game tau_p calibration vs true search deviation rate;
(4) Track A subgame CFR unchanged.

## ADDENDUM 2026-09-08 — v11/v12: the compass corrected

Logs: results/v11_gates.log, results/v12_gates.log (read both).
v11/v11b (measured discard model): fitted conditional-logit BEATS
the energy likelihood held-out everywhere and on all draw-type
exact-conditional benchmarks; NOT promoted — the blocker is the MH
frozen-composition rescoring x fitted-weights interaction (model
proven innocent by mh0 parity). 372k-decision harvest + tau_p
asymmetry table (search declines gain>=1 takes 25%) banked.
v12 (Track A): KILLED BY THE POWER BUDGET before any solver code —
only 4.9% of home hands reach stock<=4; phase-conditional oracle:
full home bound +15.25 +/- 1.92 but stock<=10-only +0.95 and
stock<=4-only +0.29. >93% of information value is early/mid and
largely IRREDUCIBLE: retire the "oracle gap = belief headroom"
framing. Frontier #1 is closed for home rules.
REGRESSION LADDER (binding user directive, run every promotion):
v10home vs teacher +3.54 +/- 1.66 (best ever; v4home era +1.23),
vs knocker +7.09 +/- 1.40 — the chain of self-play-tie promotions
IS real strength against fixed baselines.
Registered next: (1) compute-scaling curve (stages 16/48/384+,
kbranch 4 — the only class that ever beat a champion at fixed
information), (2) _window_loglik redesign with per-event
primary-time context (unblocks v11b), (3) perfect-belief-bound
instrument.
