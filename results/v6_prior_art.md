# V6 prior-art survey (2026-08-27)

Four-topic literature sweep before the v6 campaign (HANDOFF_V6.md "FIRST
TASK"). Each section ends with the design consequences adopted for our
tracks. Full citations at the bottom of each section.

---

## 1. PIMC pathologies: strategy fusion, non-locality, and when they bite

**Definitions (Frank & Basin, AIJ 1998; AAAI-98).** *Strategy fusion*: a
determinized searcher assumes it can choose a different continuation in
each sampled world; the real game requires one strategy per information
set. Long et al. (AAAI 2010) sharpen the practical condition: fusion
causes a MOVE error (not just value inflation) only when a branch has
anti-correlated per-world values ("a different in-world choice wins each
world" — deferral/option-preserving lines) AND a genuinely
world-independent alternative exists. *Non-locality*: a node's value
depends on regions outside its subtree because the opponent's arrival at
a node is itself evidence about the world. Both errors are
**independent of the number of worlds sampled** — architecture bias, not
variance.

**Magnitude in practice.** Skat: in the ~15% of games undecided at 3
tricks left, PIMC loses 0.42 TP/deal to an exact solution — ~0.063
TP/deal overall, small. Long et al.'s parameter study: PIMC is nearly
optimal at high leaf correlation / high disambiguation; its real
weakness shows up as **exploitability, not head-to-head EV** (Kuhn
poker: PIMC ties Nash head-to-head but loses 0.22/game to a best
response). Gin sits in the favorable region (public discards/pickups =
strong disambiguation channel; deadwood arithmetic = high late-game leaf
correlation), which is why our probes keep coming back "unexploited" —
and why the probe must stay the referee.

**Mitigations in deployed programs.**
- GIB (Ginsberg, JAIR 2001): Bayesian world weighting from play events
  (functionally our belief.py), plus achievable-sets search for declarer
  only: one line's won-world SET maximized across all worlds — kills
  hero-side fusion; gain ~0.1 imps/deal.
- αµ (Cazenave & Ventos 2019/2021): value = vector over sampled worlds;
  Max nodes must play the SAME move in all worlds, keep Pareto fronts.
  Exact for the sampled worlds at depth; gains over PIMC modest
  (60.2%→62.0%), ~2% of root decisions change.
- RecPIMC (Furtak & Buro, CIG 2013): rollout players are themselves
  imperfect-information players. Principled fix for opponent modeling in
  rollouts; expensive. Our opp_model-in-rollout is the cheap version.
- KBPS (Edelkamp 2021): when the belief space is small, search for lines
  that win in ALL/most consistent worlds (paranoia as a certificate,
  not as an expectation).
- ISMCTS (Cowling/Powley/Whitehouse 2012): single tree over information
  sets, determinization per iteration. Beats determinized UCT where
  hidden info is decisive; ties/loses where it isn't (Dou Di Zhu
  overall). MO-ISMCTS design rule: the opponent's reply gets statistics
  in HIS information space; your continuation must not condition on
  which indistinguishable reply he chose. Notably SO-ISMCTS can suffer
  MORE fusion than plain determinization. A 2026 gin study ran naive
  uniform-redeal ISMCTS at tiny budgets: 10-26% vs a rule expert
  (oracle variant 41-85%) — quantifies the variance problem our
  belief-weighted paired design exists to solve, not an architecture
  ceiling.

**Paranoia/clairvoyance when branching in-world (the Track 2/3 traps).**
Hero-side per-world depth = clairvoyance that GROWS with depth (the
searcher defers decisions because "the information will be available by
then" — Ginsberg's diagnosis, Long et al.'s low-leaf-correlation
pathology, and the empirical twin of our lesson #13 endgame failure).
Opponent-side per-world min = paranoia: endows the opponent with your
hand and the stock order. The two biases do NOT cancel — they select
different wrong lines. Bridge tolerates best-defence because defenders
really do approach double-dummy play; gin is symmetric-information, so
that justification does not transfer.

**Adopted consequences:**
- (T2) In-world exact endgame search must be **cross-world**: evaluate a
  hero LINE (policy) against all K worlds and pick the line with the
  best belief-weighted score — αµ/achievable-sets style — not per-world
  max. Per-world max inflates option-preserving candidates
  asymmetrically, and the paired z-test **certifies** asymmetric bias
  (lesson #4's mechanism, now with theory). Our gated-positive 2-ply
  deep max survives because it branches the hero's first ROLLOUT
  decision — mostly draw-disambiguation the hero will really observe;
  expect the fusion tax to appear as branching deepens.
- (T3) Opponent-reply branching: aggregate replies by the MODELED
  policy's probabilities (MixedKnock/teacher — the thing that already
  paid +4.26), never min. Min is admissible only as a separate
  best-defence BOUND (veto a candidate that is catastrophic in most
  worlds), not as the expectation. Test first around knock decisions,
  where reply asymmetry across candidates is real (lesson #12's
  criterion).
- (Referee) PIMC failure mode is quiet exploitability — keep
  probe_exploitability as the promotion gate and extend the adversary
  pool as the bot changes.

Sources: Frank & Basin AIJ 1998; Frank/Basin/Matsubara AAAI-98; Ginsberg
JAIR 2001 (GIB); Long/Sturtevant/Buro/Furtak AAAI 2010; Buro et al.
IJCAI 2009; Furtak & Buro CIG 2013; Cazenave & Ventos arXiv:1911.07960 +
2101.12639; Cowling/Powley/Whitehouse TCIAIG 2012; Edelkamp
arXiv:2104.05423; Kelidari et al. arXiv:2607.06854 (2026); Bethe 2010
survey; arXiv:2408.02380.

---

## 2. Inference-weighted sampling (Track 1's relatives)

**Closest relative — Skat policy inference (Rebstock/Solinas/Buro/
Sturtevant, CoG 2019).** Weight each candidate world by the product over
the observed action history of a modeled policy's probability of each
action given that world — exactly our per-event likelihood. Their π is a
stochastic net trained on human play (implicitly softened). Beats both
uniform sampling and per-card marginal models (the analogue of our
current belief.py); cardplay gains +1-3 tournament pts/game, largest for
the information-poor seat and in midgame. Cost: 5× per move because they
re-score the FULL history against fresh uniform samples every decision —
a sequential particle filter amortizes this (score only new events).
They have no answer to depletion or adversarial mismatch.

**Kermit (Buro et al., IJCAI 2009).** REJECTED runtime policy
likelihoods for two reasons: intractable over huge world sets, and their
policy was DETERMINISTIC — "P(move|world) ∈ {0,1} makes the prediction
brittle against players who do not play like ourselves." Used offline
feature-likelihood tables from 22M human games instead. Inference worth
~28% of series score; crucially, against a very dissimilar opponent
(XSkat) inference cost ≈ nothing — soft inference degrades gracefully to
no-worse-than-uniform. That asymmetric-safety shape is the target (and
matches our v4 opp-model gate: wrong model cost −0.64 ± 1.51 ≈ 0).

**Scrabble rack inference (Richards & Amir, IJCAI-07).** Bayes over
hypothesized leaves with a 0/1 argmax likelihood under a static
surrogate evaluator. Posterior concentration: top-10 leaves 60% of mass
vs <10% prior. Strength: +5.2 pts/game realized of a **+37 pts/game
full-knowledge oracle bound** (~14% of the oracle gap). Their 0/1
likelihood only worked because the opponent literally shared the
evaluator — their own caveat.

**Mahjong (Suphx 2020; Mizukami & Tsuruoka 2015).** The counter-model:
neither maintains explicit hand hypotheses. Mizukami marginalizes into
three learned sufficient statistics (P(waiting), P(tile wins | waiting),
E[score]) and plugs those into MC simulation. When the hidden space is
huge and the payoff-relevant question is low-dimensional, predicting
sufficient statistics beats hand enumeration. Gin's hidden space (10
cards from ~25) is small enough for particles — but this is the same
warning as our lessons #9/#12: the searcher consumes only a few
statistics of the belief (knockability, meld-mate location, world
weights); the PF must improve THOSE, not generic realism.

**Particle-filter mechanics (Bard & Bowling AAAI 2007; POMCP line;
SMC textbooks).** Standard fixes for exactly our concern list:
- Depletion → **resample-move** (Gilks & Berzuini 2001): ESS-triggered
  resampling plus MCMC rejuvenation moves that leave the posterior
  invariant — for gin, propose swapping a particle card with an unseen
  non-particle card, accept by likelihood ratio. The principled
  descendant of our boost=2.5 hack.
- Zero-likelihood events → temper P^κ or floor-mix
  L = (1−ε)·softmax_τ + ε·uniform. Guarantees no particle dies from one
  surprising move.
- Resample only when ESS < N/2 (with only ~10-20 informative events per
  hand, per-event resampling is pure variance).
- Bard & Bowling: deliberate small "motion noise" on particles beats
  the exactly-correct static model (counteracts finite-N degeneracy);
  likelihoods must marginalize over what you can't see.

**Softened/quantal likelihoods.** Logit QRE (McKelvey & Palfrey 1995) is
the canonical form; λ = 1/τ is standardly fit by max likelihood on
held-out play. Modern recipe = temperature scaling by held-out NLL.
Scalarizing a lexicographic policy has no published treatment; hazard:
the scalarization's scale is arbitrary, so fit (scalarization weights, τ)
JOINTLY by held-out NLL, and prefer fitting on off-model opponents'
play (clone, knockers, random) for a mismatch-robust τ.

**Gin-specific steals (EAAI-21 line, Francis/Just/Neller AAAI-21).**
(a) Shankar's Plug Rule: attribute an observed pickup to a specific meld
hypothesis and promote its SPECIFIC partners — a joint/correlational
update per-card marginals cannot express. A particle filter with a
decent likelihood subsumes this automatically (particles ARE joint
hypotheses) — a good sanity probe: does the PF posterior reproduce
plug-rule deductions? (b) Event likelihoods conditioned on rank/
suitedness relations (frequency-mined) — finer conditioning than our
(α, β, γ), same Brier harness could calibrate it as a fallback if the
policy likelihood is too slow.

**Adopted consequences:**
- (T1 Gate 0, NEW) **Full-knowledge oracle bound first** (Richards &
  Amir's experiment): v5-with-revealed-opponent-hand vs v5 head-to-head
  bounds what ANY belief improvement can buy in-search. If ≲ noise
  floor, the in-search swap can't pay (the posterior may still pay in
  Tracks 2-4 knock anticipation / ranges).
- (T1) Likelihood = τ-softmax over the belief_teacher.py scalarization
  (energy = dw_after + λ·E[meld-mates] + μ·live), ε-uniform floor,
  (weights, τ) fit jointly by held-out NLL on teacher AND clone AND
  random streams. Never 0/1.
- (T1) Filter = sequential (amortized, unlike published practice);
  ESS < N/2 resampling; repair-by-swap conditioning on hard evidence;
  Metropolis card-swap rejuvenation.
- (T1) Aim the posterior at the symmetry-breaking statistics:
  exact knock-capability probability (replaces the (10,1.0) hard
  rejection AND feeds rollout termination) and meld-mate location.
  Generic hand-realism improvements cancel in the paired design
  (lesson #9's tie, now with a literature mechanism).

Sources: Rebstock et al. arXiv:1905.10911; Buro et al. IJCAI-09; Solinas
et al. AAAI-19; Richards & Amir IJCAI-07; Bard & Bowling AAAI-07; Li et
al. arXiv:2003.13590 (Suphx); Mizukami & Tsuruoka CIG 2015; Černý et al.
AAAI 2021; Guo et al. 2017; Francis/Just/Neller AAAI-21; Doucet et al.;
Gilks & Berzuini 2001; Liu & West 2000.

---

## 3. Subgame solving with ranges (Track 4's theory base)

**The toolkit (poker).** CFR-D/Resolve (Burch/Johanson/Bowling AAAI-14):
re-solve a subgame safely by giving the opponent a gadget choice between
their blueprint counterfactual value (Terminate) and entering (Follow) —
requires a CFV vector per opponent hand at the boundary. DeepStack
(Science 2017): continual re-solving with (own range, opp CFVs) +
counterfactual-VALUE-over-ranges network at depth limit (note: a value
net over ranges, not an MSE state regressor — one reason it worked where
our lesson-#8 net failed). Libratus (Brown & Sandholm NeurIPS 2017):
Unsafe / Resolve / Maxmargin / Reach-Maxmargin comparison.

**The decisive empirical fact about UNSAFE solving** (= fixing both
ranges by Bayes from a blueprint/posterior and solving — our plan):
usually the BEST method tested (small flop hold'em 30k buckets: 0.55
mbb/h vs 4.15 for the best safe method), but in one regime
(large-game/fine-abstraction) it produced exploitability 396.8 vs 41.4
for not solving at all — ~10× WORSE. Failure mechanism: the opponent
deviates UPSTREAM to poison the assumed range, then cashes in
downstream. For us the poisoning pathway runs through the belief update
— an adversary feeding misleading discards before the endgame.

**Solving vs an estimated range = best response to a model.** RNR
(Johanson/Zinkevich/Bowling NIPS-07): full BRs to models are brittle;
mixing "opponent forced to model with prob p, free with 1−p" traces an
efficient exploitation/exploitability frontier. DBR (AISTATS-09): if the
model is estimated from limited data, restrict per-infoset by
observation count. SES (Liu et al. NeurIPS 2022): the modern synthesis —
gadget objective interpolating safety and model-exploitation with proven
bounds both ways; closest published thing to our plan. Ganzfried &
Sandholm: any exploitation beyond banked "gifts" surrenders safety —
there is no free lunch, only a dial.

**Mitigating structural point:** poker's unsafe range is a SELF-PLAY
blueprint's reach — wrong against every real opponent by construction.
Our particle posterior conditions on the ACTUAL opponent's observed
actions under a calibrated model. Against fixed non-adaptive opponents
(our whole population), Bayesian best response is optimal-in-expectation
and there is no upstream deviation. The risk class is only an adversary
that knows and games the belief update — exactly what
probe_exploitability measures, if we add such an adversary.

**Well-posedness (Zhang & Sandholm NeurIPS 2021, KLSS).** Gin subgames
rooted at our information set (not a public state) are legitimate via
knowledge-limited solving, but a proper subgame needs TWO ranges: the
opponent's hands (our posterior) AND the opponent's belief about OUR
hand. Giving the in-subgame opponent our exact hand = paranoid solve
(over-conservative, not a Nash of anything). Mirror belief.py from the
opponent's seat to build our own range — cheap, our events are public.
Gin hands are semi-public (pickups seen), shrinking the point-hero vs
range-hero gap; gate the paranoid approximation separately.

**Precedent & tractability.** Skat endgames: exact world ENUMERATION +
per-world solving + voting (Edelkamp ECAI 2020) — strategy fusion again;
nobody in cards solves the endgame as an imperfect-info game against an
inferred range to equilibrium. So Track 4 is genuinely novel; SES is the
honesty template. Size at stock ≤ 3-4: C(13..15,10) = 286-3003 opponent
hands × 3!-4! stock orders → 1.7k-72k worlds, ~10³-10⁴ public nodes.
CFR+ vectorized over the world dimension (numpy) = seconds/hand; pure
scalar Python would be minutes (vectorization mandatory).
`meta.solve_nash` is NORMAL-FORM and assumes antisymmetric/zero-value —
the endgame needs a small sequential CFR+ (new code), and terminal
payoffs come free from cached `defender_residual`.

**Adopted consequences:**
- (T4) Build unsafe-but-audited: solve against the posterior (the
  usually-best choice and the EV-optimal play vs our population); add an
  RNR-style dial (root range = λ·posterior + (1−λ)·uniform-consistent)
  to turn if the probe finds a leak; add a belief-poisoning adversary to
  probe_exploitability as a promotion gate.
- (T4) Two ranges: opponent hands from particles; our own range from a
  mirrored belief. Gate the paranoid (point-hero) shortcut separately.
- (T4) Write a small numpy world-vectorized sequential CFR+; do NOT use
  meta.solve_nash for the subgame (normal-form, zero-value assumption).
- (T4) The z-test discipline applies at the boundary too: the solver's
  move should override v5's only under the same
  significant-evidence-vs-prior rule.

Sources: Burch et al. arXiv:1303.4441; Moravčík et al. arXiv:1701.01724;
Brown & Sandholm arXiv:1705.02955; Zhang & Sandholm arXiv:2106.06068;
Johanson et al. NIPS-07; Johanson & Bowling AISTATS-09; Ganzfried &
Sandholm EC-12; Liu et al. NeurIPS 2022; Edelkamp ECAI 2020 +
arXiv:2104.05423; Cazenave & Ventos αµ.

---

## 4. Gin rummy prior art: nothing beats us; three steals

**EAAI-21 challenge** (Neller et al.; 13 papers, AAAI-21 vol 35(17); the
only organized gin rummy competition): every entry is 1-ply heuristic or
learned evaluation, some with opponent modeling; NO world sampling, no
rollouts, no exact lay-off optimization in the loop, no exploitability
measurement. Winner (Ahmed et al., Penn State): hand-strength heuristic
+ MCCFR on the knock decision only. Opponent-modeling papers output
independent per-card marginals (Francis/Just/Neller ensemble; Hein et
al. random forests; Mishra & Aggarwal DNN — which explicitly calls for
the joint model it lacks).

**Independent replications of our lessons:** Kelidari et al. 2026
("gold-standard" study): a rule expert with exact melding but NO
hidden-card reasoning beats every trained RL/ISMCTS/NFSP agent 70-99%
(best challenger 34.2%) — our lessons #1-#3 in the wild. Dhumbal 2025:
aggressive heuristic 88%, ISMCTS 9%, PPO 1.5%. Kotnik & Kalita 2003
(TD-rummy): weak. No CFR/deep-CFR/ReBeL-style system for full gin
exists. No published agent of note on rlcard's gin env. Commercial apps
(AI Factory): uncalibrated adjacency heuristics — a naive belief.py.

**Assessment: "strongest documented gin rummy bot" SURVIVES the survey.**
Our teacher alone ≈ the strongest published class; v5 adds everything
the literature lacks. One cheap falsification test remains available:
Neller's Java framework is public — a direct match vs the Ahmed et al.
winner is possible if we ever want the external scalp.

**Steals:** (1) Plug-Rule correlational deductions — subsumed by
particles, use as a posterior sanity probe. (2) Rank/suitedness-
conditioned event likelihoods — fallback likelihood family, same Brier
harness. (3) Maravich/Neller knock regression: under 25-pt-bonus match
scoring, gin attempts with 6-8 melded cards are conventionally
UNDER-valued — check when match scoring lands (roadmap #5).

Sources: EAAI-21 proceedings (Ahmed et al.; Francis/Just/Neller;
Maravich/Neller; Hein et al.; Mishra & Aggarwal; Goldman et al.; Truong/
Neller; Nguyen et al.); Kelidari et al. arXiv:2607.06854; Malla
arXiv:2510.11736; Kotnik & Kalita ICML-03; Zha et al. arXiv:1910.04376;
Mohan arXiv:2606.21975.

---

## Campaign-level synthesis (what changed vs HANDOFF_V6.md)

1. **NEW Track-1 Gate 0:** full-knowledge oracle bound (v5 with revealed
   opponent hand vs v5) BEFORE building the filter — bounds the entire
   in-search value of belief improvement. ~30 min of compute.
2. **Track 1 design fixed:** sequential PF, τ-softmax likelihood over the
   belief_teacher scalarization with ε-uniform floor, (weights, τ) fit by
   held-out NLL; ESS-triggered resample + swap-repair + MH rejuvenation;
   posterior aimed at knock-probability and meld-mate statistics (the
   symmetry-breaking consumers), not generic realism.
3. **Track 2 design changed:** in-world exactness must be CROSS-world
   (one hero line scored over all K worlds, αµ-style), not per-world max
   — per-world max is strategy fusion, its bias is candidate-asymmetric,
   and the z-test would certify it. Root candidates stay top-4.
4. **Track 3 design fixed:** opponent-reply branching aggregates by the
   modeled policy, never min; min only as a separate catastrophic-bound
   veto. Test at knock decisions first.
5. **Track 4 design fixed:** unsafe-but-audited solve on the particle
   posterior; λ-mix robustness dial; two ranges (mirror belief for our
   own hand); new numpy sequential CFR+ (NOT meta.solve_nash);
   belief-poisoning adversary added to the probe.
6. **Referee unchanged and reinforced:** determinized searchers fail
   quietly to best responses (Long et al.), so probe_exploitability
   remains the promotion gate for every track.
