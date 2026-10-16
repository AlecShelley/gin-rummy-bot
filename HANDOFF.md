# Gin Rummy NFSP Trainer — Project Handoff

## Goal
Train a strong gin rummy bot via NFSP (Neural Fictitious Self-Play) using RLCard,
then fine-tune it on true gin rummy scoring. The deliverable is a trained model
checkpoint plus a learning curve showing improvement over training.

## Files
- `train_gin_rummy.py` — the complete trainer. Already tested end-to-end
  (both reward modes, checkpoint resume, custom scorer verified zero-sum).
  Prefer editing this file over rewriting it.

## Setup
```bash
pip install rlcard torch
```
Python 3.10+. GPU strongly preferred (auto-detected via `rlcard.utils.get_device`);
CPU works but expect roughly overnight per stage.

## How it works (short version)
Two NFSP agents play each other. Each agent has:
- a DQN ("best response") network trained by TD learning,
- a supervised "average policy" network imitating its own past best responses
  (reservoir buffer). The average policy is the real deliverable — it converges
  toward Nash equilibrium in two-player zero-sum games.

State: 5×52 binary card planes. Actions: 110 discrete. Reward: terminal only.

## Reward modes
- `--reward shaped` — RLCard default: +1 gin, +0.2 knock, −deadwood/100 otherwise.
  Dense signal; use for pre-training from scratch.
- `--reward points` (default) — true per-hand score differential, zero-sum, ÷100:
  gin = opponent deadwood + 25 bonus; knock = deadwood difference; undercut
  (defender ties or beats knocker) = difference + 25 to the defender; dead hand = 0.

**Why the curriculum matters:** untrained/random agents dead-hand ~97% of games
(verified empirically), so pure points reward from scratch is almost all zeros.
Pre-train with shaped reward first.

## Run this
```bash
# Stage 1: learn to form melds (dense reward)
python train_gin_rummy.py --reward shaped --num_episodes 50000 \
    --log_dir results/stage1

# Stage 2: fine-tune knock/gin risk decisions on real scoring
python train_gin_rummy.py --reward points --num_episodes 100000 \
    --load_model results/stage1/gin_rummy_nfsp.pth \
    --log_dir results/stage2
```

## Monitoring
- Each run writes `performance.csv` + a learning-curve `fig.png` and the final
  model `gin_rummy_nfsp.pth` into `--log_dir`.
- The eval metric is agent-0's average payoff vs a **random** opponent over
  `--num_eval_games` hands (every `--evaluate_every` episodes).
  - shaped mode: expect climb from ~−0.6 toward positive values.
  - points mode: value ×100 = average points won per hand vs random.
- Training prints per-step rl-loss/sl-loss lines (very noisy — fine to redirect
  to a file). `sl-loss: None` early on is normal: the supervised net waits for
  `min_buffer_size_to_learn` samples.

## Known caveats / gotchas
- RLCard's gin rummy has **no lay-offs**, so knocking is slightly stronger than
  in real rules. Don't "fix" this in the scorer; it's an engine limitation.
- Checkpoint loading uses `torch.load(..., weights_only=False)` — required
  because the whole agent object is pickled (torch ≥2.6 defaults to True).
- Gin rummy episodes are long; RLCard's env is not vectorized. If throughput is
  the bottleneck, reduce eval frequency before touching the env.
- Evaluation vs random saturates once the bot is decent. For later progress
  checks, pit the current model against an earlier checkpoint
  (`tournament` with two loaded agents) instead.

## Success criteria
1. Stage 1: eval reward clearly climbing and positive vs random.
2. Stage 2: average points/hand vs random keeps improving; knock/gin frequency
   rises and dead-hand frequency falls (worth logging if you add metrics).
3. Final model beats a Stage-1 checkpoint head-to-head over ≥500 hands.

## Nice-to-haves (only after the above works)
- Log outcome frequencies (gin / knock / undercut / dead hand) per eval block.
- Checkpoint every N episodes instead of only at the end.
- Score-aware match play: append both players' match scores (2 normalized
  scalars) to the observation and fine-tune on hands sampled at random score
  states. Requires touching the env's state extraction — nontrivial.
- Deep CFR as a stronger (heavier) alternative to NFSP.
