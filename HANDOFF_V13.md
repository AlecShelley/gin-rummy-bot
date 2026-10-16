# V13 Handoff — v10home, the corrected compass, and the narrow frontier

> Written 2026-09-09 for transfer to a fresh session. Consolidates the
> 2026-09-04 → 09-08 campaigns (v9 leak hunt, the posterior
> ground-truth check, v10 likelihood sharpening, v11/v11b measured
> discard model, v12 Track-A power budget + oracle decomposition).
> Read `CLAUDE.md` for project history and the numbered lessons — all
> still bind. Evidence, in reading order: `results/v9_gates.log`,
> `results/v10_gates.log`, `results/v11_gates.log`,
> `results/v12_gates.log` (each opens with its registration; read the
> one for whatever you touch END TO END before changing anything).
> Prior handoff: `HANDOFF_V8.md` (v8home era + addenda).

## THE GAME (binding user decisions, unchanged)

**DEFAULT GAME = THE HOME VARIANT** (10+10 deal, flip founds the
pile, knock threshold = flip value, the 2026-09-01 opening rule:
P1 option → P2 option → forced P1 stock draw). Full scoring,
standard dead-hand. **GTO ONLY — permanent: single-hand
unexploitable play; no opponent modeling, no match scoring, no
human-eval track. Do not re-propose.** Per-hand point differential
is the objective; noise floor ±1.5/1000 hands.

**NEW BINDING DIRECTIVE (user, 2026-09-08): every promotion must run
the FULL REGRESSION LADDER** — candidate vs champion gate PLUS
matches vs pinned v8home, vs the patient teacher, and vs
knock-when-legal (`home_variant.py --match`, ≥1500 seat-swapped
hands, fresh seeds). Beat-or-tie required against the whole ladder,
not just the predecessor.

## THE CHAMPION: v10home (promoted 2026-09-07)

`home_variant.py --bot v10home` (the default) = `HomeV8Policy`
(class name kept for compat) on the current `particle.DEFAULT_PF`.
Composition = v8home (v7 search + PF, opening book, turn-2 book with
low-card exception) + the v10 likelihood sharpening:
**tau_po=0.0** (opening pass/take mismatch is ZERO — the flip book
is deterministic in deployment; the ground-truth check measured 10×
excess weight on book-refuted hands under the old tau_p=0.1) and
**eps 0.05→0.02**. Gated per the v7 protocol: posterior capture up
on six ground-truth scenarios, truth reads hold-or-sharpen, stream
Brier ≥ champion (random-mismatch −0.28%), match TIE (pooled
−0.72 ± 1.10 @ 4000 paired — decision rule logged BEFORE the
extension ran), probes clean including a new **book-inverter**
adversary (+0.90 ± 1.81 — opening deception costs the deceiver more
than the belief damage).

**Frozen-identity pins (the v10 lesson — partial cfg dicts merge
over DEFAULT_PF and silently drift when it changes).** Every frozen
identity pins `eps=0.05, tau_po=None` explicitly: registry v6 AND
v7 (`meta.py`, `play_vs_bot.py`), `read_regress.py` v6_cfg,
`--bot v8home` (`home_variant.py`, via the new
`HomeV8Policy(pf_cfg=...)` kwarg). **Any future DEFAULT_PF change
must first add the changing keys to every pin site** — grep for
`pickup_mode="logistic"` and `pf_cfg` to find them. Verify both
directions after a flip: new default reproduces the candidate's
benchmark numbers exactly; the old pin reproduces the old
champion's (post_sweep, below).

**Fresh ladder evidence (2026-09-08)**: v10home vs patient teacher
**+3.54 ± 1.66** (best home-vs-teacher ever; v4home era was +1.23),
vs knock-when-legal **+7.09 ± 1.40** (was +5.38). The chain of
self-play-tie promotions IS real strength against fixed baselines —
lesson #9's symmetric cancellation hides symmetric gains; ladders
reveal them. Logs: `results/ladder_v10_{teacher,knocker}.log`.

## THE CORRECTED COMPASS (v12 — the era's biggest result)

- **Home oracle bound for v10home: +15.25 ± 1.92** (phase_oracle.py,
  first measurement under home rules). But phase-conditional
  decomposition: oracle knowledge only at stock ≤ 10 is worth
  **+0.95 ± 0.52**; only at stock ≤ 4, **+0.29 ± 0.25**.
- **>93% of information value lives at stock > 10** (early/mid), and
  much of it is IRREDUCIBLE — at turn 3 the observations do not
  determine the hand, and the posterior ground-truth checks show the
  PF already captures 60–92% of the per-node information that IS
  there. **Retire the "oracle gap = belief headroom" framing.**
- **Endgame solving is DEAD for the home variant** (formally closed
  by the registered power rule BEFORE any solver code): only 4.9% of
  hands reach stock ≤ 4 (1.4% reach ≤ 2). The v6_prior_art §3
  subgame-CFR design rules remain valid for the standard game only.
- Every DECISION surface is forced-arm audited CLEAN (v9): opening
  and j=1 takes (books), j≥2 take/draw (−0.80 ± 2.36 confirmed),
  knock timing (a measured strength: its knocks +5.60 ± 2.45, its
  declines dodge −10.42 ± 1.40), discard safety (safety-first
  challenger LOSES −3.47 ± 1.65). The bot's play is within noise of
  optimal ON ITS OWN BELIEFS; only information quality and search
  depth remain as levers.

## OPEN FRONTIERS (leverage order, registered in v12's close)

1. **Compute-scaling curve** — the recommended next campaign. Does
   adaptive+deep keep growing with budget (stages 16/48/384+,
   kbranch 4, 2–4× wall-clock)? Historically the ONLY mechanism
   class that beat a champion at fixed information (v5, lesson #13:
   better-ALLOCATED extra compute). Gate: candidate vs v10home
   paired (t10_gate.py pattern with a search-cfg override instead of
   pf_cfg), then the full regression ladder. Watch lesson #13's
   sub-lesson: precision upgrades must not widen the candidate set.
2. **MH-rescoring redesign** — unblocks the VALIDATED v11b discard
   model. The fitted conditional-logit likelihood beats the energy
   model held-out in every context and on every draw-type
   ground-truth benchmark; it is proven innocent of its one gate
   failure (home10 read: bit-parity with v10 at mh_window=0; the
   regression is entirely the `_window_loglik` frozen-composition
   rescoring × fitted-weights interaction — rejuvenation lifts
   v10's read +0.133 but the fitted model's only +0.036). The fix
   spec: give MH rescoring per-event PRIMARY-TIME context (the
   drawn-card attribution and the re-added-discard feature
   distortion are the two named mechanisms). Weights live in
   `particle.py` (V11_W_OPEN/PICKUP/DRAW, 6 features incl. meld
   protection), selected by `discard_mode="fit"`/"fitdraw"
   (default "energy" = frozen champion behavior; anchors +
   post_sweep no-op reproduction green as of 2026-09-08).
3. **Graded take tier** — the harvest measured the take rule's
   mismatch as ASYMMETRIC and gain-graded: P(take) = 0.013 / 0.611 /
   0.605 / 0.870 for gain ≤0 / 1 / 2–3 / ≥4; decline-despite-gain
   0.252 vs the modeled tau_p=0.1; speculative takes 0.013. A
   two-parameter (or gain-graded) take tier in `pickup_prob` is a
   cheap registered candidate — gate on the posterior benchmarks
   first (that is what killed naive sharpening twice).
4. **Perfect-belief-bound instrument** (research): worlds sampled
   from the exact Bayes conditional would measure ACHIEVABLE
   headroom (vs the oracle's irreducible bound). No cheap
   implementation is known; posterior_check's rejection sampling is
   the only existing exact-conditional machinery (per-node, offline).

## METHODOLOGY (the reusable instruments — use these)

- **Registration-first, always**: open `results/v13_gates.log` (or
  continue v12's) with hypotheses, gate bounds, decision rules, and
  seeds BEFORE any data. Retries must be registered; a failed gate
  closes the candidate (v11/v11b died honestly this way). Borderline
  match gates: log the extension decision rule BEFORE running the
  extension (the v10 pattern).
- **Posterior ground-truth check** (`posterior_check.py`) — the
  era's key instrument, the user's design: fix a hero-observable
  info set (hero hand + flip + event history), estimate the TRUE
  conditional over opponent hands by rejection sampling (hero side
  fixed by construction; implication-only screens — never force,
  never teacher-screen a posterior study; hero policy seed FIXED at
  900001, opp seed fresh per sample; accept iff the v10home line
  reproduces the history, early-abort on divergence), compare the
  PF ensemble's marginals. **Eight frozen scenarios s1–s8** with
  stored rows (`results/post_s*_rows.jsonl`) = standing regression
  benchmarks; `post_sweep.py --cfg "..." --scenarios s1,...` scores
  ANY cfg against all of them in seconds — re-slice, never re-run.
- **Census-then-fork** (`turn3_study.py` pattern; also
  knock_study/discard_study/turn3_confirm): play the deal to the end
  on the champion line recording every decision node + the bot's
  choice, pick ONE node per hand deterministically
  (random.Random(deal_seed*2+1)), replay-fork the arms (fresh
  policies, same seeds — 10,000+ forks this era, 0 replay
  mismatches), finish both, pair per deal.
- **Behavior harvest** (`discard_harvest.py` + `discard_fit.py`):
  372,648 decisions from 10k v10home self-play hands in
  `results/discard_harvest.jsonl` — every discard (with context and
  drawn card) and every free-draw take/decline. Conditional-logit
  MLE by plain numpy GD (scipy is blocked on this machine).
- **Seed hygiene**: shard seeds fan as base + 977·i; new bases must
  be residue-distinct mod 977 from ALL used bases:
  {78500, 79100, 80900, 81500, 82100, 83100, 85300, 86200, 90000,
  94000, 100000, 102000, 107000, 108000, 110000, 112000, 114000,
  114500, 116000, 118000, 120000, 122000, 124000, 126000, 128000,
  130000, 132000, 134000, 140000, 142000, 144000, 146000, 148000}
  (92000, 96000, 98000, 104000, 106000, 136000, 138000 retired
  unused — do not reuse). ps multipliers used: 7211, 9013, 10141,
  12347, 12983, 14813, 15733 — pick a fresh prime. Scenario-replay
  hero policy seed: 900001 (opp 900002) — reuse THESE exactly when
  extending the frozen scenarios.

**Harness rules (non-negotiable, all bitten this era):** shard small
(10–20), `imap_unordered`, append JSONL as shards land, print
progress+ETA per shard (first-shard warm-up ~10 min — ignore early
ETAs). **Never a trailing `&` inside a wrapped command** (violated
once this era; audited clean, but the rule stands). Never edit
imported .py while workers run. `rnd.current_player_id` is a numpy
int: **never `is False` on comparisons** (cost a 15-min scan its
result). Machine sleep inflates elapsed/ETA — check marginal shard
rate before believing a stalled ETA.

## Infrastructure map (v9–v12 additions)

| File | What it is |
|---|---|
| `posterior_check.py` | THE posterior ground-truth instrument; SCEN table s1–s8 (kind: decline/take/draw); sampler + PF-ensemble compare + M1–M4 report. |
| `post_sweep.py` | Score any pf_cfg against the stored scenario rows in seconds (capture %, channel gaps, deadwood gap). |
| `turn3_study.py` / `turn3_confirm.py` | Census-then-fork mid-game take/draw studies (v9). |
| `knock_study.py` / `discard_study.py` | Forced-arm knock-timing and discard-safety audits (v9, both clean). |
| `nat2_reslice.py` | Offline reslice of the t2 exception rows by outs. |
| `discard_harvest.py` / `discard_fit.py` | Self-play behavior harvest + conditional-logit MLE (v11). |
| `t10_gate.py` | Paired match gate, pf_cfg candidate vs champion (clone me for any new gate). |
| `book_inverter_probe.py` | Opening-deception adversary probe (v10; part of the probe suite now). |
| `phase_oracle.py` | Phase-conditional oracle (PhaseOraclePolicy; the compass instrument). |
| `particle.py` | + tau_po, nu/sigma/eps_out (measured-negative knobs, frozen at no-op), discard_mode {energy,tier,fit,fitdraw}, V11_W_* weights, context-tagged event log (discard_d/_o/_p), context-aware `_window_loglik`. |
| `home_variant.py` | + `HomeV8Policy(pf_cfg=...)`, --bot {v10home (default), v8home (pinned), v6home, v4home}. |
| `results/v9..v12_gates.log` | The campaign logs — registrations + every number above. |
| `results/*_rows.jsonl` | Raw rows for every study incl. the 8 scenario benchmarks and the 372k harvest — re-slice, never re-run. |

## First-session checklist

1. Gates (~5 min, all must be green): `sim.py --verify`,
   `fullrules.py --test`, `adaptive.py --parity`,
   `ladder.py --parity`, `particle.py --anchors`,
   `home_variant.py --test-opening`.
2. Frozen-behavior spot check:
   `post_sweep.py --cfg "kdraw=4"` must print
   `cap[s1: 91.6% s2: 70.1% s3: 61.4%] mean 74.3%` (the champion
   line); `read_regress.py` must PASS.
3. Read the tail of `results/v12_gates.log` (the compass correction
   + registered next candidates), then the log of whatever frontier
   you pick, end to end.
4. Default all work to the HOME variant / v10home. Open
   `results/v13_gates.log` with a registration before any data.
5. On any promotion: DEFAULT_PF pin discipline (above), evaluator
   lockstep (belief note string in `evaluate.py`), the FULL
   regression ladder (binding), memory + `CLAUDE.md` pointer +
   handoff addendum in BOTH copies (repo + Downloads).
