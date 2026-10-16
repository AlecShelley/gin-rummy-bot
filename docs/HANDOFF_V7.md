# V7 Handoff — bot v6, hardened evaluator, and the road to v7

> Rewritten 2026-08-31 consolidating the 2026-08-29/30/31 sessions
> (adversarial audit, example libraries, explanation pipeline, rules
> decisions, the hidden-5♠ incident). Read `CLAUDE.md` for the
> project-wide picture and numbered lessons. Evidence trails:
> `results/v6_gates.log` (v6 campaign), `results/v7_gates.log` (v7
> audit), `results/v6_prior_art.md` (survey; design rules still bind).
> Pre-rewrite text: `results/HANDOFF_V7_pre_rewrite_backup.md`.

## Where the bot stands

**v6 is the promoted champion** (`pf_expert.py` PFPolicy, registry
`v6`, `play_vs_bot.py v6`): the unchanged v5 staged search (16/48/192
+ deep 2-ply, z=1.5, opp_model (10,1.0) veto) with worlds from a
**particle-filter posterior** (`particle.py`; resample-move
rejuvenation is the load-bearing piece). ~0.44 s/searched-move.

| v6 vs | pts/hand | notes |
|---|---:|---|
| v5 | **+1.92 ± 0.97** | 3984 fresh, selection +2.31 |
| compute-matched v5 control | +0.50 ± 0.94 | architecture wins |
| teacher | **+2.56 ± 1.65** | first-ever head-to-head win |
| teacher (standard dead-hand re-gate) | **+2.04 ± 1.62** | seed 73300, 2026-08-30 |
| smart | +8.76 ± 1.31 | |
| best of 21 probe adversaries | −0.26 ± 3.24 | ALL lose |
| belief-poisoning adversaries (v7) | ad_self +3.58, ad_jam +3.15, ad_self_k10 +8.37 | teacher same-seed baseline +3.52 — poisoning gains NOTHING |

**Oracle compass (vs v6, seed 72500): +13.47 ± 1.50** — perfect
opponent-hand knowledge is still worth ~13.5 pts/hand over the
champion. This number is the campaign's direction-finder.

**v6home** (user's table rules): vs v4home +1.90 ± 1.83, vs
knock-when-legal +6.39 ± 1.48. Fresh PFExpert per hand with
opp_model=(flip, 1.0) — the threshold is public. Known fidelity gap
(user-accepted): the dealer's second option on a passed flip is not
modeled.

## RULES OF THE GAME (all user decisions — binding)

1. **Full rules** (2026-08-22): lay-offs onto the knocker's melds,
   undercut +25 after lay-offs, no lay-offs against gin, dead hands
   0-0 (`fullrules.py`).
2. **Standard dead-hand rule** (2026-08-30): once the stock is down to
   2 cards, the player to move may not draw or pick up — knock/gin on
   the discard before that, or the hand is dead 0-0. Hardwired in
   `sim.fast_legal_actions`; the ONE deliberate divergence from the
   rlcard judge (which allows pickup chains on a dead stock), counted
   and tolerated by `sim.py --verify` (~13 nodes / 250 hands). Match
   results gated before 2026-08-30 used the rlcard variant; the
   teacher re-gate above shows the effect is endgame-marginal.
3. **Home variant** (the user's table): 10+10 deal, one card flipped
   to found the pile, knock threshold = the flip's deadwood value.
   `sim.new_game_home`; the search adapts with zero retraining.

## Saturation + dead ends (do not relearn)

Seven independent v6 perturbations ALL tie (results/v6_gates.log
TRACK 3): kbranch 4 (−0.68), worlds ×2 (−0.19 — K-scaling is dead on
the PF posterior), Brier-calibrated PF (−0.18), no-veto (+0.47
confirm), particles ×2 (+0.13), cross-world exact endgame (−0.17,
parked: endgame decisions too rare), 3-ply (+0.44 at 1.58×). v6 is a
measured local optimum; only structural moves can spend the oracle
gap. Belief-poisoning at zero deadwood cost is also dead as an attack
(v7 audit, poison.py): the PF likelihood's mass sits on
deadwood-after, which tie-break deception cannot move; the PF's
public bookkeeping passed 13,561 invariant checks.

Binding infrastructure lessons: `sim.restore` only truncates
move_sheet — use `endgame._BSnap/_brestore` for ANY branching search
over diverged states (plain snapshot/restore only within one line or
across content-identical siblings); a hero-DRAW candidate's drawn
card must join any cross-world observation key.

## The evaluator (current state — kept in lockstep with the champion)

Standing rule: every bot upgrade propagates to `evaluate.py`/`viz.py`
as part of promotion. User display rules live in
`memory/evaluator-display-rules.md` (player-facing: no model jargon,
percent hand-reads, prominent variant banner, seen-card marking, pile
provenance colors, layout opponent→pile→hand→history, conditional-EV
labels, no em-dash restatements).

Engine & belief: deep 2-ply engine (AdaptiveExpert single stage,
default 192 worlds), belief = v6 particle posterior built from the
position's COMPLETE transcript as a **4-filter ensemble** (worlds
from the mixture — single-filter short-history reads are
initialization noise), classic-weights fallback with a stated reason.
Opp-model defaults to (knock-max, 1.0) = the champion's model
('opp-model: none' opts out). **The particle path runs
`PFExpert(veto=False)`** (2026-08-31): the hard opponent-knock world
filter discarded evidence-proven melded worlds (see incident below);
the filter's own no-knock evidence remains, the knock model still
shapes rollout continuations, and the no-veto variant match-gated as
a tie. `--veto` restores the old filter.

Verdict honesty: deviations with z < 2.5 are labeled BORDERLINE
(single borderline crossings are seed-sensitive; pos_take1 measured
12 ways is a ~+1 near-tie yet single seeds printed ±3-point
"significant" verdicts both ways); follow-up plans carry
"(close call)" / "(statistical tie with X)".

**Explanation pipeline** (`explain.py`, `--explain [--playouts N]`,
default 600, seconds of runtime): plays the recommended move and its
main rivals (always incl. the best knock) to the END of the hand over
belief-sampled worlds, both sides simulated with the engine's own
continuation model. Reports outcome distributions (we gin / knock
clean / undercut / they knock / we undercut them / dead hand — % +
conditional EV) and critical next draws (grouped by our first draw,
structural GIN-out/knock-enable tags, feeds-the-opponent tracking,
"likely in their hand (p%)" tags, opponent-danger cards, "they take
our X n% of the time").

Position format (`evaluate.py --example`): full replay-consistent
transcripts are REQUIRED for the particle belief (history validator
audits the pile; the PF replay cross-checks hand/pile/opp-known/
hidden-count and catches holes the pile replay cannot). Home games:
`knock-max:` + `flip:` (needed whenever the flip left the pile —
learned the hard way). `opp-draw` = stock draw with empty pile.
`picked: X` = the 11th card came from the pile (never also list it as
a my-pickup event). DeclareDeadHand renders properly.

**Example libraries** (all engine-generated -> rational,
replay-consistent histories by construction; ground truth in
`results/home_examples_truth.txt`):
- `home_examples.py` -> pos_home1..10: flip take, discard-timing
  deviation, bridge take, 5-pickup reading, losing seat, endgame gin
  take, undercut refusal, ace-game grind, KNOCK-correct (home9,
  z~6) and NOT-knock-correct (home10, deadwood-1 knock refused for
  the gin push).
- `mine_hard.py` (fast evaluator sweep over 6,874 nodes of 600 games;
  rows in `results/mine_hard_rows.json`) -> pos_hard1..9: blocker
  play (throw the ace, keep their queen), which-poison discard,
  knock-or-die at stock 2, razor knocks (genuine coin-flips),
  refused bait take, anti-greed knock, rich-hand tie. pos_hard8
  (dead-hand ticket) RETIRED — it cannot arise under the standard
  dead-hand rule. pos_hard3_minus1 = the same hand one turn earlier,
  where the knock/discard choice is live.

## The hidden-5♠ incident (2026-08-31) — and what it proved

User inference on pos_hard3_lastdiscard: opponent declined 4♣, TOOK
4♠, declined 4♦; with 2♠ dead and 3♠ in our hand, they must hold
5♠+6♠. The report showed "5♠ 1.3x" and no 6♠. Diagnosis:

- **The filter was RIGHT**: raw particle marginals P(5♠)=P(6♠)≈1.00
  at every seed (truth: they held 4♠5♠6♠7♠). Trajectory-level
  compounding works.
- **The display hid it**: ×-vs-chance units saturate late (base 78%
  caps certainty at "1.3x"), and shrunk_marginal(kappa=80) with
  n_eff≈3 collapsed the read onto the classic belief (6♠ → 0.49,
  below threshold). Fixed: displays use the RAW ensemble marginal in
  percent, tiered ("almost certainly holding: 5♠ (100%), 6♠ (99%)").
- **The veto corrupted the decision**: it rejected 110/192 worlds —
  exactly the melded hands the pickup evidence proves — so knock EVs
  contradicted the read and 5♠ appeared as a hero gin out. With
  veto=False, hard3_minus1's knock re-priced +10.1 → **−11.1 (52.5%
  undercut)** and the danger-card list names the true hand's
  completion set.
- **A real bot-model defect was confirmed** (the user's hypothesis):
  the pickup likelihood is INVERTED — P(take 4♠ | junk hand dumping
  a king) = 0.97 vs P(take 4♠ | holding 5♠6♠7♠) = 0.38. The
  gain-logistic rewards deadwood dumps over run extensions, and
  tau_p=1.0 was NEVER calibrated (the grid sweeps only
  tau_d/p_knock/mu). The filter survived here only because the twin
  4-declines bracketed the pickup.

## NEXT MAJOR IMPROVEMENT — opponent-action likelihood campaign

This is priority #1 and it is also the first concrete step on the
oracle gap (track C item 1 of the v6 survey: "the likelihood is still
a softened heuristic"). The hidden-5♠ incident turned that from a
research direction into a measured defect list:

1. **Redesign the take likelihood** (`particle.SoftOpponent.
   pickup_prob`): run-extension/meld takes by already-developed hands
   must never score below junk-dump takes. Candidate shape: model the
   teacher's actual binary rule (take iff gain ≥ draw_gain) with a
   small mismatch floor, plus a meld/extension term — instead of the
   pure gain logistic. Sanity anchor: P(take 4♠ | holds 5♠6♠7♠) must
   come out near 1, P(take | no use) near the floor.
2. **Calibrate tau_p** — add it to the `particle.py --calibrate`
   grid (it was never swept).
3. **Fix evidence forgetting**: `_window_loglik` re-scores only the
   last 6 opponent events during rejuvenation, so MH swaps can walk
   particles away from old constraints. Always include ALL
   pickup/decline events (rare, information-dense); window only the
   discards.
4. **Gate per discipline** (this changes DEFAULT_PF = the promoted
   bot): offline first — held-out Brier AND the worlds-gate metrics
   (overlap / dw-gap / knock-Brier; Brier alone transferred badly
   before), plus a targeted regression suite of read positions
   (hard3's 5♠6♠, home10's kings, hard1's queens — the truth key
   makes these checkable). Then v6-with-new-PF vs v6: selection run,
   fresh-seed confirm ≥ 4000 hands, probe including the poisoning
   adversaries. Expect the match delta to be small (lesson: most
   belief realism is symmetric and cancels) — the payoff is
   concentrated posteriors (oracle gap) and analysis credibility;
   promote only if ≥ tie with all gates green.

After (or alongside) that campaign, the standing strategic tracks:
- **Track B — match-style scoring + human eval** (roadmap #5/#6): +25
  box bonus is a one-line payoff change; re-gate v6's knock calculus
  under it, then `play_vs_bot.py v6` sessions / the public EAAI-21
  winner. The user's end goal is beating humans at their scoring.
- **Track A — subgame CFR endgame solving** (unbuilt; design rules in
  results/v6_prior_art.md §3; the standard dead-hand rule shrinks the
  endgame further — re-check the power budget before spending hands).
- **Adaptive (t,p) opponent-knock estimation** — the veto/no-veto and
  patient-teacher episodes both show the fixed (t,1.0) model is the
  weakest assumption; estimating knock behavior online (single-hand,
  per user constraint) would retire it.

## Infrastructure map (v7 additions)

| File | What it is |
|---|---|
| `particle.py` / `pf_expert.py` | The v6 filter + policy (unchanged this campaign). |
| `oracle_gate.py` | Oracle bound; rerun after any belief change (last: +13.47 vs v6). |
| `poison.py` | Belief-poisoning probe: Advertiser (salting/jamming), PF invariant stress (13.5k checks), worlds-gate on poisoned streams, match runner. Registry: ad_self / ad_jam / ad_self_k10 — now part of every promotion probe. |
| `home_examples.py` / `mine_hard.py` | Example generators (engine-played games; miner sweeps + classifies hard nodes). |
| `explain.py` | The explanation pipeline (classified full playouts, outcome stats, critical cards). |
| `evaluate.py` / `viz.py` | The evaluator (see section above). |
| `results/v7_gates.log` | v7 audit evidence. Also: `results/v7_probe_matches.log`, `results/v7_oracle_v6.log`, `results/v7_regate_deadhand.log`, `results/mine_hard_rows.json`, `results/home_examples_truth.txt`. |

## Discipline (non-negotiable)

Noise floor ±1.5/1000 hands; selection seed ≠ confirmation seed;
winners confirm fresh at ≥4000; compute-matched controls for cost
changes; machinery parity before match runs; never report a max of
noisy measurements. Promotion = beats-or-ties field + fresh confirm +
clean probe (now incl. the poisoning adversaries) + the evaluator
upgraded to the same engine/belief + the example library re-verified.
Windows spawn rules and no-edit-while-running per CLAUDE.md. Three
standing user rules on constructed examples: rational histories,
full replay-consistent transcripts (enforced by history_check),
engine-generated preferred. Display rules per
memory/evaluator-display-rules.md.

## First-session checklist (next session)

1. `sim.py --verify`, `fullrules.py --test`, `adaptive.py --parity`,
   `ladder.py --parity` (~3 min; verify reports ~13 strict-dead-hand
   diffs vs the judge — expected).
2. Start the likelihood campaign at step 1 (the pickup_prob redesign)
   with the offline gates; do NOT touch DEFAULT_PF without the full
   protocol above.
3. Log everything in `results/v7_gates.log` (append) or a fresh
   `results/v8_gates.log` once the campaign opens.
