# Gin Rummy Bot — Handoff (v2, method known to work)

## Goal

Train a gin rummy bot for RLCard's `gin-rummy` that **beats the smart-draw rule
agent** (defined below). A previous session already produced a bot that beats
rlcard's stock rule agent; your job is to build the next one with the improved
method and clear that higher bar.

**The bar to beat** — current bot (`results/final/gin_rummy_bot.policy.pt`),
1500 hands, seats swapped, 95% CI:

| Opponent | Points/hand |
|---|---:|
| rlcard stock novice rule agent | +8.37 ± 1.14 (wins) |
| random | +52.74 ± 0.87 (wins) |
| **smart-draw rule agent** | **−2.25 ± 1.14 (loses)** ← beat this |

Reference points for the same opponent, already measured:

| Match | Points/hand |
|---|---:|
| stock rule agent vs smart-draw rule agent | −11.26 ± 1.07 |
| smart-draw rule agent vs random | +51.23 ± 0.80 |

**Your target: a bot that beats the smart-draw rule agent with a 95% CI
excluding zero.** That means gaining ≥2.25 points/hand on the current bot.

> **v2 status: done.** `results/v2_final/gin_rummy_bot.policy.pt` scores
> **+2.30 ± 1.10** vs the smart-draw rule agent over 2000 hands, and **+7.27 ±
> 1.14** vs the bot above. See RESULTS.md.
>
> **v2 correction — the reasoning below is wrong.** It argued that cloning a
> teacher that does not flip coins "should recover most of that gap on its own".
> It cannot. `draw_decision()` in `gin_rummy_bot.py` and `SmartDraw.step` are the
> same rule, so the shipped bot and the smart-draw opponent *already* made
> identical draw decisions; the whole −2.25 gap was in discard/knock/gin, and a
> perfect clone of that teacher would only tie it. Beating the bar required
> building a stronger teacher first (`teacher.py`, `tune_teacher.py`).

A previous instance spent a long session discovering what does and does not work
here. Everything below is measured, not assumed. **Read the "Do not retry"
section before writing any training code** — it will save you 15+ hours.

## Setup

Working venv already exists at `.venv/`. From scratch:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install rlcard matplotlib
```

The **CUDA 12.8 build is mandatory** on this machine: the GPU is an RTX 5080
(sm_120) and the default PyPI torch wheel cannot run on it. Python 3.13.7,
rlcard 1.2.0, numpy 2.5.2 all work together fine.

Note: for these tiny networks, CPU was measured *faster* than GPU (kernel-launch
overhead dominates). GPU is used mainly to keep the machine's cores free.

---

## The key discovery

`GinRummyNoviceRuleAgent.step` (in `rlcard/models/gin_rummy_rule_models.py`)
branches on gin → knock → discard. **The draw decision (`DrawCardAction` vs
`PickUpDiscardAction`) matches none of those branches**, so `actions` falls
through as the entire legal set and the method ends at
`np.random.choice(actions)`.

**The rule agent decides whether to take the upcard by coin flip.** Every turn
opens with that decision, so roughly half of the "expert's" moves are noise.

> **v2 correction — the draw is not the only coin flip.** `_get_best_discards`
> returns *every* discard tied on resulting deadwood and the caller ends at
> `np.random.choice`: **40.6% of discard decisions are random tie-breaks** (mean
> 1.64 options). The choice among legal `KnockAction`s (mean 2.5) is random too.
> Fixing both is worth only +0.14 ± 1.14 points/hand on its own, but it is what
> raises a clone's achievable accuracy from 76% to 96%. See `teacher.py`.

Consequences that drive the whole method:

1. Fixing only that decision is worth **+10.91 ± 1.05 points/hand** — the rule
   agent beats *itself* by that much (`smart_draw.py`).
2. A behaviour clone of the stock rule agent tops out near **76.5% action
   accuracy**, and that is *not* a deficiency: if half the teacher's decisions
   are 50/50 coin flips, a perfect imitator scores ~75%. The clone was faithfully
   copying the randomness.
3. **So: clone a teacher that does not flip coins.** That is the improvement this
   handoff exists to implement.

The heuristic (already implemented as `draw_decision()` in `gin_rummy_bot.py`):
take the upcard only when it strictly lowers your best-melding deadwood,
otherwise draw blind. Picking up otherwise is doubly bad — it fails to help, and
it tells the opponent exactly which card you hold.

---

## The method

### Step 0 — confirm the baseline reproduces (~10 min)

The teacher is `SmartDraw(GinRummyNoviceRuleAgent())` from `smart_draw.py`.
Reproduce the −2.25 ± 1.14 figure above before changing anything, so you know the
harness works and you have a like-for-like comparison later:

```python
from smart_draw import SmartDraw, match
from gin_rummy_bot import GinRummyBot
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
bot = GinRummyBot("results/final/gin_rummy_bot.policy.pt")
print(match(bot, SmartDraw(GinRummyNoviceRuleAgent()), 1500, 2024))
```

### Step 1 — behaviour-clone the smart-draw teacher

Reuse `pretrain.py`, changing the teacher from `GinRummyNoviceRuleAgent()` to the
smart-draw wrapper. Keep the noisy roll-in (`--teacher_epsilon 0.15`): the
*executed* action is sometimes random for state coverage, but the *label* is
always what the teacher would have done.

```bash
.venv/Scripts/python.exe pretrain.py --episodes 8000 --epochs 30 --out results/v2_pretrained.pth
```

**Expected signal that it is working:** action accuracy should climb well above
the old 76.5% — probably 90%+ — because the coin-flip decisions are now
deterministic and learnable. If accuracy is still ~76%, the teacher swap did not
take effect. That is your gate.

### Step 2 — DAgger with diversified roll-in

```bash
.venv/Scripts/python.exe dagger.py --rounds 5 --seed_episodes 8000 \
    --episodes_per_round 6000 --epochs 15 --diverse_rollin \
    --init results/v2_pretrained.pth --out results/v2_dagger
```

Two details matter more than the algorithm:

- **Keep D_expert.** Retraining on new data alone catastrophically forgets:
  measured −5.43 → −8.41 points/hand. The dataset must be the union of expert
  data and every round's clone-visited data.
- **Diversify the roll-in.** Rolling in only against itself produces a policy
  that has never seen a hand run to stock exhaustion, so it dead-hands 75% of
  games against random. Split roll-in across self / random / teacher opponents.
  With that fix, round 1 beat the clone on *both* metrics at once.

### Step 3 — select honestly, then benchmark

DAgger's per-round evals used 400 hands (95% CI ≈ ±2.2) and swung from −0.88 to
−6.27. **Selecting on that is selecting noise** — round 4's headline −0.88 became
−2.74 at 1500 hands. Re-score every saved round at ≥1500 hands (`select_final.py`),
then report the winner's score on a *different seed*. Expect the held-out number
to be worse than the selection number; that gap is the winner's curse and the
held-out one is the honest figure (measured: −2.14 → −3.15).

Ship with `greedy=True` (argmax, worth 2–3 points/hand over sampling) and
`smart_draw=True`.

### Step 4 — report

Benchmark the final model against: **smart-draw rule agent** (the real bar),
stock rule agent, random, and `results/final/gin_rummy_bot.policy.pt` (the
previous bot). Always swap seats at halfway and always report a 95% CI.

---

## Do not retry these (all measured, all failed)

**Reinforcement learning does not work in this environment from scratch.** Six
configurations were tested; every one ended *worse than its own random
initialisation* (knock rate 7% → 2%, dead hands 91% → 97%):

| Tried | Result |
|---|---|
| NFSP self-play (the original plan) | flat in **both** networks over 150k-episode attempt |
| DQN, lr 1e-4 / 1e-3 | worse than random init |
| DQN, `train_every` 1 and 4 | worse than random init |
| DQN, γ=1.0 + fast target updates | worse than random init |
| Potential-based shaping alone | helps (−0.489 → −0.367) but nowhere near enough |
| NFSP warm-started from a good clone | **degraded it monotonically**: +0.443 → +0.270 vs random over 10k episodes |
| Policy gradient (REINFORCE + baseline + KL leash) from the clone | oscillated −5.69 to −7.80 over 8000 episodes, no trend |

**Why**, so you do not rediscover it: knocking only becomes legal once deadwood
drops to 10, which random exploration reaches in ~7% of hands. The DQN therefore
almost never observes the reward for going out, and learns to avoid it. More
episodes, more gradient steps and higher learning rates all fail *identically*
because none of them fix what the agent never sees.

For NFSP specifically: its average policy is trained to imitate its inner DQN's
best responses. Measured on a warm-started agent, the average policy knocked
74.7% of hands while its own DQN knocked **2.0%** — so self-play systematically
overwrote a good policy with a bad one. Action-value regression handles a
decisive action that is 2% of transitions spread over 52 action ids badly;
cross-entropy handles it fine. That asymmetry is the whole story.

**It is not the representation.** Behaviour cloning with the identical 5×52
observation, 110-action space and network reached −0.169 vs random where RL
managed −0.55. Do not go rewrite the state encoding.

If you want to attempt RL anyway, do it *after* delivering a working bot, start
from the clone, gate every checkpoint on a held-out benchmark, and keep the best
— never ship whatever the last episode produced.

---

## Gotchas that cost real time

- **`torch.save(nfsp_agent)` writes ~1.4 GB** — it pickles both 200k-transition
  replay buffers. Save policy-only files (`save_policy()` in
  `train_gin_rummy.py`, 1.8 MB) for anything you evaluate or ship. Verified to
  produce bit-identical actions to the full agent.
- **rlcard spams stdout**: `dqn_agent.py:223` and `nfsp_agent.py:164` are raw
  `print(..., end='')` with a leading `\r`. 60 episodes produced 627 KB of
  single-line garbage. Use `_QuietStdout` from `train_gin_rummy.py`.
- **`get_best_meld_clusters()` and `get_deadwood_count()` hard-require exactly 10
  cards.** You hold 11 between drawing and discarding — precisely when you want
  to evaluate the hand. Use `get_meld_clusters()` (no size restriction) and score
  it yourself; see `_min_deadwood()` in `shaping.py`.
- **`AveragePolicyNetwork` starts with `BatchNorm1d`.** Keep it in `.eval()` for
  single-state inference, and skip training batches of size 1.
- **`is_allowed_to_discard_picked_up_card` defaults to `False`** — you cannot
  throw the upcard straight back.
- **Evaluation is expensive**: every action is a separate small forward pass, so
  a 500-hand eval costs ~145s. Budget for it; it was ~40% of wall clock at one
  point.
- Throughput with `train_every=4` is ~6 eps/s in steady state (the 9 eps/s
  benchmark was taken before the replay buffers filled).

## What is already verified about the engine

You do not need to re-check these:

- The custom points scorer in `train_gin_rummy.py` is **exactly zero-sum**
  (max |p0+p1| = 0.0 over 400 hands) and its knock/gin/undercut arithmetic is
  correct against `games/gin_rummy/round.py`.
- Random self-play dead-hands **97.25%** of the time (389/400).
- Gin rummy's double-actions (same player draws *then* discards) interleave
  correctly through `Env.run` and `reorganize`.
- rlcard's NFSP→DQN positional argument list (13 params) maps correctly.
- **RLCard's gin rummy has no lay-offs.** This is an engine limitation, not a
  bug. Do not "fix" it in the scorer — the reward would then disagree with the
  game actually being played.

> **v2 correction — knocking is *weaker* here, not stronger.** The sentence that
> used to end this bullet ("It does mean knocking is stronger here than under
> paper rules") is wrong, and it was the most expensive wrong belief in this
> document. A dead hand scores **0 for both players**, so holding costs nothing
> but the knock points forgone, while an opponent who knocks early hands you
> +25 undercuts. Never knocking at all — gin only — beats the smart-draw rule
> agent by **+3.77 ± 1.59**, where knocking at the first legal chance scores
> +0.90 ± 1.39. This is where v2's entire margin came from. Full sweep in
> RESULTS.md.

## Existing files to reuse

| File | What it gives you |
|---|---|
| `gin_rummy_bot.py` | `GinRummyBot` wrapper + `draw_decision()` heuristic |
| `smart_draw.py` | `SmartDraw` wrapper — **your new teacher** — and the isolation experiment |
| `pretrain.py` | behaviour cloning with noisy-expert roll-in |
| `dagger.py` | DAgger with `--diverse_rollin` |
| `select_final.py` | high-power model selection |
| `head_to_head.py`, `benchmark.py` | seat-swapped matches with 95% CIs |
| `play_vs_bot.py` | terminal play, renders melds/deadwood/knock-at |
| `shaping.py` | `_min_deadwood()`, potential-based shaping |
| `train_gin_rummy.py` | original NFSP trainer, fixed and instrumented (works, just doesn't learn) |
| `RESULTS.md` | full evidence for every number quoted here |

## Success criteria

1. BC action accuracy on the smart-draw teacher is **clearly above 76.5%** (gate:
   if not, the teacher swap silently failed).
2. Final model beats the **smart-draw rule agent** over ≥1500 hands with a 95% CI
   that excludes zero.
3. Final model beats `results/final/gin_rummy_bot.policy.pt` head-to-head over
   ≥1500 hands.
4. Report every number with seats swapped and a confidence interval. If something
   did not beat its baseline, say so plainly.

## Where the remaining headroom is

Once the above works, the teacher is still beatable — and this is where *learning*
can finally add value that a heuristic cannot:

- **The teacher ignores most of the observation.** It looks only at its own hand.
  The 5×52 state also carries the discard pile, dead cards, and the opponent's
  known cards. A network can learn opponent-aware play (what is safe to discard,
  when the opponent is close to going out) that no hand-written teacher here uses.
  This is the most promising direction.
- **The discard rule is 1-ply greedy** — it minimises immediate deadwood and
  ignores draw potential (e.g. keeping a card that could complete a run). A 2-ply
  or expectation-based discard would be a stronger teacher still.
- **Knock timing.** Both teachers knock the instant it is legal. Holding for gin
  (+25 bonus) is sometimes better, especially against a slow opponent — the
  previous session found the clone *beat* its teacher against random precisely by
  holding for gin more often.

A reasonable path: build the stronger teacher first, clone it, and only then try
learned improvement on top — measuring against a fixed, held-out benchmark every
step of the way.
