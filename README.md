# Gin Rummy Bot

A gin rummy engine built to answer one question: **when I play a hand, where
exactly did I make a mistake?** It grew into what is likely the strongest
documented gin rummy bot — a belief-state Monte Carlo searcher with a
particle-filter read on the opponent's hidden hand, a measured opening
theory, and a chess-engine-style position analyzer — developed end-to-end
with [Claude Code](https://claude.com/claude-code) over a series of
heavily-gated experimental campaigns.

The game it plays by default is my table's **home variant**: both players
dealt 10, one card flipped to found the pile, knock threshold = the flip's
deadwood value (ace flip → you basically must gin), with full scoring
(lay-offs, undercut +25 after lay-offs) and the standard dead-hand rule.
The opening follows real table rules: first player may take the flip, else
the second player may (stealing the tempo), else a forced stock draw.

## How it works

**No reinforcement learning survived contact with this game** — the bot is
Bayesian inference plus simulation plus statistics.

1. **Belief: a particle filter over the opponent's hidden hand**
   (`particle.py`). Every observable action — a pickup, a declined upcard, a
   discard, an opening pass — multiplies 256 concrete hand hypotheses by
   P(action | hand) under a calibrated opponent model, with resample-move
   (MH card-swap rejuvenation) keeping the ensemble diverse. The likelihood
   model itself went through three measured generations: a softened-energy
   model (v6), a binary take-rule with a meld tier after the likelihood was
   caught scoring junk-dump takes *above* meld-extending takes (v7), and a
   sharpened opening model after a ground-truth posterior check showed the
   filter systematically under-confident (v10).

2. **Search: determinized paired rollouts with a statistical deviation
   test** (`expert.py`, `adaptive.py`, `pf_expert.py`). Sample worlds from
   the posterior, play every candidate move to a rules-aware evaluation in
   each world (paired, so world luck cancels), and deviate from a strong
   heuristic prior only when the paired evidence is significant (z-test).
   A staged budget (16 → 48 → 192 worlds, escalating only on close calls,
   with exact 2-ply verification at the top stage) spends compute exactly
   where decisions are hard. Worst case, the bot plays the prior — that
   one design rule is why noisy simulation never degrades it.

3. **A measured opening & early-game book** (`sim.flip_book_take`,
   `t2_book_take`). The search's truncated rollouts systematically
   misprice the first moves of a gin race (the deadwood they optimize only
   matters in the *losing* branch), so the opening was solved empirically:
   thousands of forced-arm paired deals — play the same deal twice, forcing
   take vs pass, with the full bot on both seats — yield the rule
   *take the flip iff it melds; the second player also takes with two-way
   adjacency* (their take is effectively an extra turn: the identical
   non-melding ace take measures **−5.5 for P1 and +9.1 for P2**).

4. **The evaluator** (`evaluate.py`, `viz.py`, `explain.py`) — the original
   point. Describe any position in a text file and get every candidate's EV
   with significance verdicts, the belief's percent read on the opponent's
   likely holdings, full-playout outcome distributions ("you gin 34%, they
   knock 13%…"), and an HTML report. It always runs the champion's exact
   engine and belief — a standing rule, because an evaluator weaker than
   the bot lies to you.

Every promotion passed a gate ladder: selection match and fresh-seed
confirmation on separate seeds (per-hand σ ≈ 33 pts — the noise floor is
±1.5 pts/hand at 1000 hands), adversarial exploitability probes
(purpose-built adversaries including belief-poisoners and an
opening-deception bot — **nothing we built exploits it**), regression
matches against pinned prior champions and fixed baselines, and
ground-truth posterior benchmarks. Current champion (`v10home`): beats the
patient-teacher heuristic **+3.54 ± 1.66** and knock-when-legal
**+7.09 ± 1.40** per hand; a perfect-information oracle holds **+15.25 ±
1.92** over it, >93% of which was measured to be early-game information
that is largely *irreducible* from the observations — i.e., the bot's
decisions are within noise of optimal on its own beliefs (every decision
surface was forced-arm audited clean).

## What was tried: worked / didn't

Everything below is measured, with the evidence in `results/*_gates.log`
and the full history in `CLAUDE.md` + `HANDOFF_V13.md`.

**Worked (each promoted through the full gate ladder):**

- Belief-state rollout search with a z-tested heuristic prior (v3) — the
  foundational architecture; never lost to anything since.
- Rules-aware leaf evaluation (exact lay-off/undercut arithmetic in the
  rollout horizon) — fixed a bias the z-test can't catch (it guards
  against noise, not bias).
- Opponent-knock-model conditioning of the rollout *continuation* +
  meld-correlated world sampling (v4, **+6.59 ± 0.76** vs weak knockers) —
  the two only work together.
- Adaptive staged world budgets + 2-ply close-call verification (v5,
  +1.20 ± 0.93) — better-*allocated* compute, not more compute.
- Particle-filter posterior with resample-move rejuvenation (v6,
  +1.92 ± 0.97; first-ever head-to-head win over its own teacher).
- Binary take-rule likelihood with a meld tier (v7) — fixed an inverted
  likelihood found by human hand-reading (see below).
- The home opening rule + measured opening/turn-2 books (v8home; the
  turn-2 clause gated **+1.84 ± 1.78**, the program's first significant
  head-to-head match win since v4).
- Opening-likelihood sharpening driven by ground-truth posterior checks
  (v10home).

**Didn't work (all measured, all documented, none relearned):**

- Model-free RL — six configurations (NFSP, DQN variants, policy
  gradient), all ended worse than their own initialization (knocking is
  legal ~7% of random-exploration hands; the signal never arrives).
- Behavior-cloning the search (95.9% action accuracy, played 4–9 pts/hand
  worse — the expert's edge is rare deviations that look like label noise).
- A learned value network as the search leaf (MSE-optimal ≠ good leaf:
  determinized rollouts see the sampled stock order; no static evaluator
  can match that clairvoyance, and regression blurs 1–3 pt candidate gaps).
- Action-value distillation (Q-net diff RMSE 4.6 pts vs 1–3 pt true gaps —
  any use that lets the net select from its own Q selects its own noise).
- Realism-corrected world sampling *alone* (large measured bias, zero
  head-to-head effect — symmetric world bias cancels in paired designs).
- Knock-decision mixing / range balancing (monotone: less knocking is
  simply safer under full rules).
- Widening the candidate set (endgame all-discard mode: more comparisons =
  more noisy-argmax; precision upgrades must not widen the surface).
- Belief-poisoning attacks against the bot (the filter's likelihood mass
  sits on deadwood-after, which tie-break deception cannot move).
- A per-hand opening *solver* (full-playout calculator at the flip node):
  validated cell-by-cell offline, then **lost to the book** −0.53 ± 0.62 —
  a pooled ground-truth rule beats a noisy per-hand calculation with an
  imperfect continuation model.
- Endgame subgame solving for the home variant — formally closed by a
  pre-registered power budget: only 4.9% of hands reach stock ≤ 4, and
  oracle knowledge there is worth +0.29 ± 0.25. Dead before any solver
  code was written.
- A fitted conditional-logit discard likelihood (v11) — beats the energy
  model on every held-out benchmark, yet blocked by a measured interaction
  with MH rescoring; closed honestly, fix spec registered.

## The human-insight ledger

The development loop was: play real hands, compare my judgment to the
bot's, and when we disagreed, turn the disagreement into an explicit
hypothesis and a designed experiment. Every entry below is a human
conjecture or decision that was then *measured* — several were confirmed,
some refuted, and the refutations are kept too.

**Game-theoretic insights (confirmed by experiment):**

- *Laying off is never a strategic decision* — so full rules are a
  terminal-payoff change on the same game tree, needing no engine rework.
- *The hidden-5♠ read*: from an opponent's declined 4♣, taken 4♠, declined
  4♦, I inferred they must hold 5♠+6♠ — the bot's display disagreed. That
  one hand exposed a display-saturation bug, a shrinkage bug, **and** an
  inverted pickup likelihood (junk-dump takes scored above meld takes),
  which became the v7 promotion.
- *"In an ace game the winner gins, so deadwood doesn't matter — taking a
  non-melding ace wastes a tempo"* — confirmed at z = 4.9; became the
  opening book.
- *"The second player's flip take is effectively an extra turn, so
  speculation pays for them"* — confirmed at z = 4.0; became the
  P2 adjacency clause (the largest seat asymmetry we measured: 14.6 pts on
  the identical take).
- *The best-case non-meld take* (adjacency-2 ace, shedding a dead king):
  proposed as a counterexample to the book — real in constructed states
  (+1.1), and the broad rule built from it *failed* its natural-play gate,
  which led to…
- *Naturally-conditioned rejection sampling*: only keep sampled hands where
  the opponent **chose** the triggering action, and record hand strength
  (meld outs) at the decision — this user-designed methodology flipped the
  verdict (+3.58 ± 2.41 for low-card two-way takes; weak hands gain most,
  +11.0), and the refined exception was adopted.
- *The posterior ground-truth check*: fix everything the hero can see and
  rejection-sample the exact Bayes conditional over opponent hands — the
  instrument that proved the filter under-confident (capturing 60–92% of
  available per-node information) and drove the v10 sharpening.
- *"Should the opening be calculated per-hand instead of a rule?"* — the
  right challenge; building the calculator and gating it produced the
  honest answer (the book **is** the calculation, pooled over 6,000
  ground-truth deals, and is within ~0.3 pts/hand of optimal).

**Rules and scope decisions:**

- Full scoring, standard dead-hand rule, and the home variant's real
  opening (P1 option → P2 option → forced draw) as binding rules.
- **GTO only**: the goal is unexploitable single-hand play — no opponent
  modeling, no match scoring, no human-eval track.
- The home variant is the default game; the standard game is legacy.

**Engineering/method rules (each learned the hard way):**

- Rational histories: any constructed example must be sane for every
  actor up to the decision point (caught via a declined K♦ with three
  kings in hand).
- The evaluator must always run the promoted bot's exact engine/belief
  (caught via a 6× EV undervaluation in an outdated analysis engine).
- Stage-aware world sampling (turn-1 opponents were being sampled ~25
  deadwood points too strong).
- Long studies must save shards incrementally and print live progress —
  instituted after a 15-hour silent `pool.map`.
- Every promotion runs the full regression ladder (prior champions +
  fixed baselines), because self-play ties hide symmetric gains.

## Running it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

.venv/Scripts/python sim.py --verify            # engine gates
.venv/Scripts/python home_variant.py --match --vs teacher --hands 300
.venv/Scripts/python evaluate.py --example > pos.txt   # then edit & analyze:
.venv/Scripts/python evaluate.py pos.txt --html
```

Deep documentation: [`CLAUDE.md`](CLAUDE.md) (project history + the
numbered hard-won lessons), [`HANDOFF_V13.md`](HANDOFF_V13.md) (current
state), `results/v*_gates.log` (the registered evidence for every claim
here). Large raw datasets (self-play harvests, posterior benchmark rows,
network checkpoints) are excluded from the repo for size.
