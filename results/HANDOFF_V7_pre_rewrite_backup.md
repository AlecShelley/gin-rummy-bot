# V7 Handoff — after the particle-filter promotion

> Updated 2026-08-29 with the home-variant port, the evaluator
> upgrades (engine, tie annotation, flip badge, follow-up plan), the
> illustrative-example library, and two standing user rules. See
> "Session 2026-08-29" below.

Standard gin rummy, FULL RULES (`fullrules.py`). Read `CLAUDE.md` for
the project-wide picture and numbered lessons; `results/v6_gates.log`
is the complete evidence trail of the v6 campaign;
`results/v6_prior_art.md` is the literature survey whose design rules
are still binding (especially for track A below).

## Where the bot stands (2026-08-28)

**v6 is the promoted champion** (`pf_expert.py` PFPolicy, registry
`v6`, `play_vs_bot.py v6`): the UNCHANGED v5 staged search
(16/48/192 + deep 2-ply, z=1.5, opp_model (10,1.0) veto) with worlds
sampled from a **particle-filter posterior** (`particle.py`):
sequential PF over the opponent's hidden hand, softened-teacher
likelihoods (tau-softmax over the dw+mu·live energy, eps floor,
lik_floor 0.02 robust clip), atomic (draw, discard) turns pricing
drew-and-dropped at 1/|pool|, fully-adapted APF kept-draw step,
belief-guided repair, and — load-bearing — **resample-move**
(windowed-MH card-swap rejuvenation after each ESS resample; without
it the PF only TIES the old belief). Cost ~1.33x v5 (0.44
s/searched-move under fan-out).

| v6 vs | pts/hand |
|---|---:|
| v5 (prior champion) | **+1.92 ± 0.97** (3984 fresh; selection +2.31) |
| v5 @ 16/48/256 (compute-matched control) | +0.50 ± 0.94 (nothing → architecture wins) |
| teacher | **+2.56 ± 1.65 — first head-to-head win ever** |
| smart | +8.76 ± 1.31 |
| best of 21 probe adversaries | **−0.26 ± 3.24 — ALL 21 lose** (cleanest probe in project history) |

**Oracle bound** (new permanent gate, `oracle_gate.py`): perfect
opponent-hand knowledge is worth **+15.91 ± 1.44** over v5. v6
captured ~12% of it. **~14 pts/hand of belief headroom remains open —
this number is the campaign's compass.**

## The saturation result (the v6 campaign's second headline)

Seven independent perturbations of v6 ALL tie within noise
(`results/v6_gates.log`, TRACK 3):

| Perturbation | Result | Cost |
|---|---:|---|
| kbranch 4 | −0.68 ± 1.63 | ~1.2x |
| world budget x2 (16/48/384) | −0.19 ± 1.63 | 2x — **K-scaling is dead on the PF posterior** |
| Brier-calibrated PF params | −0.18 ± 1.63 | 1x |
| no-veto (PF p_knock replaces (10,1.0) veto) | +0.47 ± 0.98 confirm (selection mirage +0.99) | ~1x |
| particles x2 (N=512) | +0.13 ± 1.62 | 1.36x |
| cross-world exact endgame (track 2) | −0.17 ± 1.65 | ~1.1x |
| 3-ply rollouts | +0.44 ± 1.65 | 1.58x |

v6 is at a measured local optimum. Do NOT spend more hands on local
search-config moves; the paired design cancels everything symmetric
(lessons #9/#12) and the asymmetric channels (knock model, leaf,
world source) are now all upgraded. What remains is structural.

## Dead / parked this campaign — do not relearn

1. **Endgame exactness (track 2), second generation**: the
   fusion-free cross-world observation-grouped search (endgame.py,
   correct by construction, parity + 200-hand stress) TIES at match
   level — endgame decisions are too rare (0.64/hand, 29% escalate)
   and gin endgames disambiguate too fast (Long et al. predicted
   exactly this). Resolving a <0.5 pt effect needs ~13k hands or an
   endgame-conditioned gate; parked as not worth the hands.
2. **More worlds / more particles / deeper plies**: see the table.
3. **The Brier-calibrated PF config**: offline-optimal ≠
   match-optimal. The worlds-gate metrics (overlap, dw realism,
   knock-Brier) were the offline criteria that transferred; keep
   using them, not marginal Brier, for any future belief change.

## Two hard-won infrastructure lessons (binding for tracks A/B)

1. **`sim.restore` only TRUNCATES move_sheet/game.actions** — content
   is assumed to be a shared prefix. Restoring snapshots across
   DIVERGED search branches leaves another branch's tail in place and
   the engine reads the PHASE from the last entries →
   fast_legal_actions silently corrupts (scoring actions mid-hand).
   Use `endgame._BSnap/_brestore` for ANY branching search over
   diverged states. Plain snapshot/restore is sound only within one
   line or across content-identical siblings (the v5 world pattern).
2. **A hero-DRAW candidate's drawn card must join any cross-world
   observation key** — it enters the hand before observation
   recording starts (endgame.py bug #1).

## The road forward (priority order)

### Track A — subgame CFR endgame solving (the structural frontier)

The only unbuilt v6-plan track, and the only construction that uses
the particle posterior in a way the paired rollout design cannot
cancel: build the imperfect-information subgame at stock ≤ 3-4 with
the PARTICLE posterior as the opponent's range and solve to
equilibrium. All design rules are in `results/v6_prior_art.md` §3:

- **Unsafe-but-audited**: solve against the posterior (usually best,
  Brown & Sandholm), add an RNR-style λ-mix dial (root range =
  λ·posterior + (1−λ)·uniform-consistent) to turn if the probe leaks,
  and add a **belief-poisoning adversary** to probe_exploitability as
  a promotion gate.
- **Two ranges**: opponent hands from particles; OUR range from a
  mirrored belief (our events are public). The point-hero shortcut =
  paranoid solve — gate it separately.
- **`meta.solve_nash` is unusable here** (normal-form, zero-value
  antisymmetry assumption). Write a small numpy world-vectorized
  sequential CFR+; terminal payoffs are free from cached
  `defender_residual`. Tractability at stock ≤ 3-4: C(13-15,10) hands
  × s! orderings ≈ 1.7k-72k worlds, ~10³-10⁴ public nodes → seconds.
- Sobering context from track 2: the endgame is low-frequency, so
  expect the same power problem. Consider gating on
  endgame-conditioned hands from the start, and decide the hand
  budget BEFORE running (the effect to detect is likely <1 pt/hand
  overall). If the expected effect can't clear the affordable CI,
  skip to track B and revisit.

### Track B — match-style scoring + human evaluation (roadmap #5/#6)

The user's end goal is a bot stronger than any human, and humans play
match scoring. (a) Add the +25 box/game bonuses (one-line payoff
change in `fullrules.py`); the search adapts with zero retraining.
Re-gate v6's config under match scoring — the knock calculus shifts
toward more knocking, and the Maravich/Neller finding (gin attempts
with 6-8 melded cards are conventionally under-valued under 25-pt
bonuses) becomes testable. (b) Then `play_vs_bot.py v6` sessions
and/or a match vs the public EAAI-21 winner (Ahmed et al., Java
framework is public) — the one cheap external falsification of
"strongest documented bot".

### Track C — spend the oracle gap (research, no promises)

~14 pts/hand of hand-knowledge value is provably there and provably
not reachable by local moves. Honest candidate directions, in
decreasing concreteness: (1) a learned PROPOSAL/likelihood for the PF
(the likelihood is still a softened heuristic; a small net fit to
actual policy behavior would concentrate the posterior further —
but respect lesson #10's RMSE bar and gate offline on the
worlds-gate metrics, not Brier); (2) ISMCTS-style shared-tree search
over the particle set (the survey's budget argument — one tree
instead of K independent evaluations — is unexplored here);
(3) opponent-population likelihoods (fit tau/p_knock per opponent
archetype online WITHIN a hand from the observed style — single-hand
only per the user's constraint).

## Evaluator engine upgrade (2026-08-29 — user-caught blind spot)

The user-facing position evaluator (`evaluate.py`) had been left on the
v4-family FLAT engine while the match bots moved on. This was NOT a v6
change and NOT primarily a resolution issue: v5/v6 already carry the
deep 2-ply knock-aware rollouts; the evaluator didn't.

The catch (user's out-counting, pos_take2_swap.txt): take an
unconnected 6C for a 16→13 deadwood swap, or draw — where 11/39 unseen
cards enable an IMMEDIATE next-turn hero knock and the take enables
none. The flat engine's rollout hero is the patient teacher, which
never knocks inside rollouts, so next-turn hero-knock value is priced
as mere gin progress. Measured at K=1536: flat engine draw +0.74 ±
0.35 (resolution alone flips the earlier 384-world "tie" but still
undervalues); deep engine draw **+4.50 ± 0.39** — a ~6× undervaluation
of the draw branch. This is lesson #12's asymmetry principle
(rollout-interior play only matters when it breaks payoff symmetry
across candidates) biting in the analysis tool: take-vs-draw with
knock outs on one side is exactly such a case.

Fix shipped: `evaluate.py` now defaults to `engine="deep"`
(AdaptiveExpert, single stage, deep_from=0, kbranch=3); `--engine v4`
keeps the historical engine for comparison. Reports (text + HTML) also
now explicitly flag statistical ties whenever the kept prior is not
the top raw mean — a kept-prior recommendation below the best point
estimate had read as a contradiction to the user.

**STANDING RULE (user directive, 2026-08-29): every bot upgrade must
be propagated to the evaluator.** Promotion of a new champion is not
complete until `evaluate.py`/`viz.py` run the same search AND belief
components (or the gap is impossible and explicitly documented in the
report output). The evaluator lagging the bot by two generations is
how the 6C blind spot happened. Known gap CLOSED
2026-08-29 (v7 audit): evaluate.py/viz.py now build the v6 particle
posterior from any COMPLETE transcript (4 independent filter
replications, worlds from the mixture) and fall back to the classic
belief with an explicit reason line otherwise -- see "Session
2026-08-29 (v7 adversarial audit)" below.

## Session 2026-08-29 — home port, evaluator hardening, example library

**v6home (the user's table rules).** `home_variant.py --bot v6home`:
fresh PFExpert per hand with opp_model=(flip, 1.0) — exact, the
threshold is public. ParticleBelief is home-aware (flip excluded from
the initial particle pool via ParticleCore pile0; 10+10 deal sizing;
SoftOpponent.knock_max read from game.settings — standard games see
10, so the gated v6 path is bit-identical, verified by smoke).
Validated: **v6home vs v4home +1.90 ± 1.83** (1200 hands; edge
concentrated in knock games, flip≥8: +3.36), vs knock-whenever-legal
+6.39 ± 1.48 (v4home's mark was +5.38). Fidelity gap, flagged to the
user and accepted for now: the real table rule gives the DEALER a
second option on the flip after a non-dealer pass; the engine
transform models only the first actor's option.

**Demo scripts** (both follow the rational-history rule):
- `home_demo.py` — the same hand under an ace vs a ten flip: a
  knock-shaped hand knocks for +43.6 vs +27.1 in the ten game and
  collapses to +26.3 (knock illegal) in the ace game; a gin-shaped
  hand is worth MORE in the ace game.
- `home_upcard_demo.py` — the take-the-upcard connectivity ladder
  (0/1/2/3 combos/immediate meld): speculation is EV-losing below two
  combos, EV-indifferent at 2-3, only completing a meld is clearly
  right; the big practical finding is that VALUE SWAPS (mid card in,
  honor out, ~+6 with zero combos) drive most correct takes.

**Evaluator upgrades** (beyond the engine swap described above):
- Follow-up plan (user directive): a pickup recommendation now
  re-analyzes the resulting 11-card node and reports "then: discard
  X" (or "then: KNOCK ..."), text and HTML. A stock draw cannot carry
  one (unknown card) — stated as such.
- Statistical-tie annotation: when the kept prior is not the top raw
  mean, both outputs say so explicitly with the diff ± SE.
- Home rendering: header banner "HOME VARIANT — flip X -> threshold
  N" driven by an explicit `home` flag (a ten-value flip used to hide
  it), and the pile-bottom flip wears a "FLIP <= N" badge.

**Illustrative position library** (all home-style; every file carries
a FULL transcript that the new validator confirms; .txt + .html):
  sample_home_ten     knock-shaped hand knocks at 10 (+17.4 over on)
  pos_take1_spec      2-way 7H speculation, ten game: tie, take stands
                      (follow-up shown: then discard 7D)
  pos_take2_swap      the 6C value swap: DRAW +4.04 ± 0.57 (the
                      position that exposed the flat-engine blind spot)
  pos_take3_meldnow   meld-now 9H: draw (refuses the 10-knock bait,
                      keeps the QC out for a better knock)
  pos_take4_acegame   same as take1 under an ace flip: take stands
                      (then discard AC)
  pos_ex5_takeplan    take 9S (completes a set); follow-up is a
                      STATISTICAL TIE between the J-drop and shedding
                      4S (re-audited 2026-08-29: the old "keeps the JJ
                      pair" claim was one arm of a seed coin-flip; the
                      evaluator now labels it as a tie)
  pos_ex6_knockrisk   seven game, knock at exactly 7 into a visible
                      club-builder: KNOCK by +12.4 — get in first;
                      their progress makes knocking MORE urgent
  pos_ex7_feeddanger  ace game where deadwood logic walks into their
                      shown cards: engine sheds 9H (+5.18) -- the 8H
                      adjacency is the lesser evil; feeding KS into
                      their QS picture is the real blunder

**Position-file gotcha (caught 2026-08-29):** `picked: X` means the
11th card came from a PILE PICKUP (sets the pickup phase and forbids
re-discarding X). For a stock draw, list 11 cards with NO picked line.
Three early files misused it and got wrong candidate sets (ex7's
verdict changed from "shed 2S" to "shed 9H" once 9H became legally
discardable).

**History validator** (`evaluate.history_check`, user directive):
every position's event transcript is replayed against the declared
pile; pickups/declines must match the replayed top and the final
replayed pile must equal `pile:`. Reports print "History: COMPLETE"
or a loud PARTIAL/INCONSISTENT warning with specifics (text + HTML).
Positions should ALWAYS carry full transcripts — partial histories
cannot be audited for sensibility (the user caught an opp-pickup of a
card that could not have been the pile top).

**Three standing user rules recorded this session** (also in memory):
1. Constructed examples must have RATIONAL play histories for every
   actor up to the decision point (a declined KD with three kings in
   hand was caught; so was a scripted opponent declining JS while
   secretly holding JC/JH/TS).
2. Every bot upgrade propagates to the evaluator as part of promotion
   (see Discipline).
3. Constructed positions must carry FULL, replay-consistent move
   histories — enforced by history_check.


## Session 2026-08-29 later (v7 adversarial audit -- both targets)

User directive: be adversarial to the bot AND the evaluator, then
implement improvements. Full evidence: `results/v7_gates.log`.

**Bot -- the belief-poisoning gate now exists and v6 passes it.**
`poison.py` Advertiser: FastTeacher fundamentals with the discard
tie-break reversed into a false signal (mode self = salt toward own
kept cards, mode jam = maximize live-count energy); deadwood parity
proven, salt fires on 15-25% of discards at zero fundamentals cost
(vs teacher -0.44 +/- 1.11). Same-seed matches (71100, 1488 hands):
v6 vs teacher +3.52 +/- 1.63, vs ad_self +3.58 +/- 1.63, vs ad_jam
+3.15 +/- 1.64, vs ad_self_k10 +8.37 +/- 1.31 -- poisoning gains the
adversary NOTHING; offline worlds-gate metrics on poisoned streams
are unchanged (overlap 0.560 vs 0.561 late). Mechanism: the PF
likelihood's mass sits on deadwood-after, untouchable by zero-cost
deception; the eps/lik_floor Kermit insurance absorbs the tie-break
lie. The prior-art survey's prescribed belief-poisoning adversary is
now a permanent registry member for future promotion probes. PF
invariant stress: 13,561 decision-node checks over 4 opponents x
{standard, home} -- bookkeeping exact, zero desyncs.

**Evaluator -- three defects found by adversarial seed-testing, all
fixed (evaluate.py/viz.py only; no bot files touched):**
1. Champion-belief gap closed: particle posterior (x4 filter
   ensemble) from complete transcripts, classic fallback with reason.
   The PF replay cross-checks h/hand/pile/opp-known against the
   declared position, catching transcript holes history_check cannot
   see (verified: missing final my-draw).
2. Filter-noise verdicts: a single filter on a short history carries
   spurious 1.2-1.7x reads on arbitrary cards (top-read cards
   completely different across filter seeds); the ensemble mean fixes
   the display and the world mixture the EVs.
3. Verdict honesty: pos_take1's draw-vs-take measured 12 ways is a
   ~+1 pt near-tie, yet single seeds printed both "draw +2.70
   significant" and "draw -2.98". Deviations with z < 2.5 are now
   labeled BORDERLINE; follow-up plans carry "(borderline)" /
   "(statistical tie with X)" when the node is nearly indifferent
   (ex5's follow-up landed on 3 different discards in 3 seeds).
Plus: opp-model now DEFAULTS to (knock-max, 1.0) -- the promoted
bot's model ('opp-model: none' opts out); worlds default 96 -> 192
(the champion's final stage); doc rot fixed (sample_turn3 KNOCKS
under every engine -- the old "declines a knock" claim never
reproduces; ex5's "keeps the JJ pair" was one arm of a coin-flip).

**Oracle compass refreshed vs v6** (oracle_gate --vs v6, seed 72500):
oracle vs v6 **+13.47 +/- 1.50** (1488 hands). v6 closed ~2.4 pts of
the old +15.91-vs-v5 bound; **~13.5 pts/hand of belief headroom
remains open against the current champion** -- the compass still
points at structural belief work (tracks A/C).

## Home-variant example library (2026-08-29, engine-generated)

`home_examples.py` plays REAL home games (teacher-family policies both
seats, rotating opponents) and cuts evaluator positions out of them --
transcripts rational and replay-consistent by construction. Ten
positions shipped as pos_home1..10 (.txt + .html), ground-truth key in
`results/home_examples_truth.txt`, full run log in
`results/home_examples_run.log`. Highlights: home2 (deep-2-ply
discard-timing deviation the flat engine cannot see), home7 (knock at
deadwood 2 refused into a 7-pickup opponent -- truth: their deadwood
was 1, knocking loses to the undercut), home9 (decisive early KNOCK
+7.1, z~6), home10 (deadwood-1 knock refused for the gin push; the
belief read had the opponent on kings -- they held all four).
Format additions this session: `flip:` line (home games where the
flip left the pile -- the pile-bottom convention silently broke
history_check/PF replay; caught by generated games), bare `opp-draw`
event (stock draw with an empty pile), and a low-surviving-worlds
NOTE when the opp-model veto rejects most sampled worlds.

## Hard-position library (2026-08-30, mined)

`mine_hard.py`: fast evaluator pass (96 worlds, deep, 2-filter PF)
over every tension node of 600 engine-played home games (6,874 nodes,
rows persisted to `results/mine_hard_rows.json`), keeping nodes where
the search overrules the teacher decisively (z >= 2.5 -- if the
strong heuristic errs, humans will too) or the top answers are within
~1 point. Nine positions shipped as pos_hard1..9 (.txt + .html), all
verified stable at 192 worlds x 3 seeds; truth key appended to
`results/home_examples_truth.txt`. Standouts: hard1 (throw the ACE,
keep the queen as a BLOCKER, z~10), hard3 (last discard of a dying
hand: an ace gifts gin to the visible ace-collector, ~35-pt swing),
hard8 (stock 2: teacher declares the hand dead, engine takes the
useless upcard as a TICKET and knocks, +12 with SE 0.3 -- the
dead-hand forfeit family the miner discovered), hard4/7 (razor
knocks: knock-vs-play measured as genuine coin-flips), hard9 (a
30-point hand where take-vs-draw AND the follow-up discard are all
ties). Evaluator addition: DeclareDeadHand actions now render
properly in both outputs.

## Evaluator display overhaul (2026-08-30, user directives)

viz.py rewritten player-first: (1) prominent variant BANNER (flip card
graphic + knock threshold -- was small header text); (2) our hand marks
the cards the opponent has SEEN us take (dashed + note: they won't
feed those melds); (3) all model meta stripped from the HTML (no
particle/posterior/z/worlds/prior/boost) -- verdicts in plain language
("clearly best by N points" / "close call" / "effectively a tie"),
badges recommended / nearly equal / costly; (4) history fix: the
current turn annotates the final transcript event in place (was
duplicated), hero draws show the drawn card. Terminal output keeps
the developer detail and gains a "My known" line. All library HTMLs
regenerated. Standing rule: user-facing surfaces follow
memory/evaluator-display-rules.md.

## Explanation pipeline (2026-08-30, user directive; explain.py)

`evaluate.py --explain [--playouts N]` (default 600; a few seconds):
for the recommended move + its main rivals (always incl. the best
knock), plays each line to COMPLETION over fresh veto-filtered belief
worlds using the engine's own continuation model (hero = gin-seeking
teacher with the deep first-decision branch = _deep_play, an
outcome-classified copy of adaptive.playout_deep; opp = the knock
model). Reports (text + a "Why" HTML panel, player language):
outcome distributions (we gin / we knock clean / we get undercut /
they knock / we undercut them / dead hand; % + avg points) and
critical next draws (playouts grouped by our first stock draw,
annotated structurally: GIN out / enables knock / improves-by-k /
no-help, plus how often a useless draw got fed to the opponent).
First results: hard3's knock refusal decoded -- KNOCK is 100% clean
but only +11.3, while playing on converts the opponent's FORCED
knock into a 75% undercut at +25 (being knocked into > knocking);
home9's knock: 99.8% clean +29.8 vs waiting = 14.5% gin + 17% they
get in first; home10's refusal: 24% gin + 75% undercut-their-knock
vs a +14.5 knock, gin outs named. Stats share the verdict's world
model (incl. the veto), stated in the output.

## Dead-hand rule option + explain/display refinements (2026-08-30)

User question exposed a RULES fact: rlcard (hence all gated results)
plays an EXTENDED dead-hand variant -- at stock <= 2 drawing is
forbidden but pickup chains continue until someone declines. Standard
rules end the hand 0-0 the moment the 50th-card drawer discards
without knocking. Added `sim.set_dead_hand("strict"|"extended")`
(default extended; --verify still matches the rlcard judge) and
evaluator support (`dead-hand: strict` line / --dead-hand flag,
shown in banner + report). hard3's verdict FLIPS under strict
(knock-or-nothing: KNOCK +11.4, all discards exactly 0; without the
opp-model veto: 97% clean +2.4 / 3% undercut -25, EV +1.6) --
pos_hard3_strict.txt/.html; hard8's dead-hand-ticket cannot arise
under strict. OPEN ITEM: confirm which rule the user's table plays;
if strict, regenerate the home library and consider re-gating v6
under it (the payoff surface changes at the endgame margin).
Explain refinements: outcome values labeled "conditional EV" (plain
font); opponent-first-card tracking ("they take our X n% of the
time", danger cards); layout reordered opponent/pile/hand/history;
em-dash-restatement phrasing removed per user display rules.

## DEAD-HAND RULE: standard, hardwired (user decision 2026-08-30)

The strict/extended distinction (added earlier the same day) is GONE:
the standard rule is now the only rule, engine-wide. `sim.
fast_legal_actions` returns [DeclareDeadHand] whenever the stock is
<= 2 at a draw phase (no pickup); this is the one deliberate
divergence from the rlcard judge and `--verify` accounts for it (13
such nodes per 250 hands, everything else exact). `set_dead_hand`,
the `dead-hand:` position key, and `--dead-hand` flag were removed.
All four machinery gates green post-change. Library regenerated:
every verdict stable except pos_hard3_lastdiscard (now
knock-or-nothing, header rewritten); pos_hard8_deadhandticket
RETIRED (its position cannot arise under standard rules); *_strict
twin files merged away. NOTE: all match results gated before
2026-08-30 used the rlcard variant; the effect is endgame-marginal
-- sanity re-gate v6 vs teacher under the new rule in
results/v7_regate_deadhand.log. A full field re-gate is only
warranted if that number looks off.

## Pile provenance colors (2026-08-30, user directive)

viz.py: pile cards carry a solid green border (our discard) or red
(opponent's), plus an inner dashed ring when the card was picked up
and later thrown back (ring color = last picker); legend under the
pile, flip stays neutral. Derived by replaying the transcript, so
partial histories degrade to neutral cards. Library regenerated.

## The hidden-5S read (2026-08-31, user-caught; display + veto fixes)

User inference on pos_hard3_lastdiscard (opp declined 4C, TOOK 4S,
declined 4D; 2S dead, 3S ours => they hold 5S+6S) vs the report's
"5S 1.3x". Diagnosis: the FILTER WAS RIGHT (raw particle marginals
P(5S)=P(6S)~1.00 every seed; truth 4S5S6S7S) but the display hid it
two ways -- x-vs-chance units saturate late (base 78% caps certainty
at 1.29x) and shrunk_marginal(kappa=80) with n_eff~3 collapses the
read onto the classic belief (6S -> 0.49, below display threshold).
Worse, the opp-knock VETO discarded exactly the proven melded worlds
(110/192), so knock EVs contradicted the read and 5S showed as a hero
gin out. Fixes (evaluator only, user-approved): (1) display uses the
RAW ensemble particle marginal in PERCENT with tiers ("almost
certainly holding: 5S (100%), 6S (99%)...; chance level 78%") --
shrunk_marginal untouched for obs/Brier consumers; (2) the particle
path now runs PFExpert(veto=False) (match-gated tie variant; knock
model still shapes rollouts; --veto restores); (3) explain chips tag
cards with P(opp holds)>=60% "likely in their hand". Effect:
hard3_minus1's knock re-priced +10.1 -> -11.1 (52.5% undercut), reads
now name the true hands (home10: kings at 70-81%; hard1: QS 81%).

CONFIRMED BOT-MODEL DEFECT, next gated campaign: the pickup
likelihood is INVERTED -- P(take 4S | junk hand w/ kings)=0.97 vs
P(take 4S | holds 5S6S7S)=0.38 (gain-logistic rewards deadwood dumps
over run extensions; tau_p=1.0 was NEVER calibrated -- the grid only
sweeps tau_d/p_knock/mu). Also: _window_loglik forgets old evidence
(only last 6 opp events; always include pickups/declines). Plan:
redesign the take likelihood (teacher-consistent: run-extension takes
must not score below junk dumps), add tau_p to the calibration grid,
gate offline (Brier + worlds-gate) then v6 selection + fresh-confirm
matches before touching DEFAULT_PF.

## Infrastructure map (additions since v6 handoff)

| File | What it is |
|---|---|
| `particle.py` | The PF: ParticleCore, SoftOpponent, offline gates (`--calibrate`, `--worlds`, `--time`). DEFAULT_PF = the gated config. |
| `pf_expert.py` | **v6.** ParticleBelief (duck-types Belief), PFExpert, PFPolicy, pfmatch CLI with search_cfg knobs. |
| `oracle_gate.py` | The oracle-bound gate (rerun after any belief change). |
| `endgame.py` | Cross-world exact endgame (parked) + **_BSnap/_brestore** (branch-safe snapshots — track A needs these). |
| `ladder.py` | playout_deep_n (n-ply rollouts, parity-proven; saturated at current cost). |
| `run_probe_v6.py` | Probe runner (seed 20260827). |
| `poison.py` | **v7.** Belief-poisoning probe: Advertiser (deadwood-neutral salting/jamming), PF invariant stress, offline worlds gate on poisoned streams, match runner. Registry: ad_self / ad_jam / ad_self_k10. |
| `results/v6_gates.log` | Complete v6 evidence trail. |
| `results/v6_prior_art.md` | Survey + binding design rules for track A. |

## Discipline (unchanged, non-negotiable)

Noise floor ±1.5/1000 hands; selection seed ≠ confirmation seed;
winners confirm fresh at ≥4000; compute-matched controls for cost
changes; machinery parity before match runs; never report a max of
noisy measurements; promotion = beats-or-ties field + fresh confirm +
clean probe **+ the evaluator (evaluate.py/viz.py) upgraded to the
same engine/belief as the new champion (user directive 2026-08-29 —
see "Evaluator engine upgrade")**. Windows spawn rules and the no-edit-while-running rule
per CLAUDE.md. Two selection mirages died to lesson #5 THIS campaign
(v3boost historically, no-veto now) — it keeps paying.

## First-session checklist (v7)

1. `sim.py --verify`, `fullrules.py --test`, `adaptive.py --parity`,
   `ladder.py --parity` (~3 min).
2. Decide track A vs B by the power question: estimate the affordable
   CI for an endgame-conditioned gate first; if <1 pt effects are
   unresolvable in budget, do track B first (it changes the payoff
   surface and therefore re-opens knock-model questions that track A
   would then solve under the RIGHT scoring).
3. Whichever track: gate per discipline, log in
   `results/v7_gates.log`.
