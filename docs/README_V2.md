# Gin Rummy Bot

A trained gin rummy bot for RLCard's `gin-rummy` environment.

**The bot:** `results/v2_final/gin_rummy_bot.policy.pt` (1.8 MB)

| Opponent | Points/hand (2000 hands, 95% CI) |
|---|---|
| random | **+76.55 ± 1.11** |
| rlcard novice rule agent | **+16.23 ± 1.12** |
| smart-draw rule agent | **+2.30 ± 1.10** |
| the previous bot (`results/final/…`) | **+7.27 ± 1.14** |

The smart-draw rule agent is the hard opponent: rlcard's novice agent with its
coin-flip draw replaced by a heuristic. The previous bot lost to it by −2.25.

See [RESULTS.md](RESULTS.md) for how it was built, what failed on the way, and the
honest limitations.

## Setup

Already installed in `.venv/`. From scratch:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install rlcard matplotlib
```

The CUDA 12.8 build is required for RTX 50-series (sm_120) cards; the default
PyPI wheel cannot run on them.

## Play against it

```bash
.venv/Scripts/python.exe play_vs_bot.py results/v2_final/gin_rummy_bot.policy.pt --hands 5
```

Expect it to hold for gin rather than knock early — that is deliberate, and it is
where its edge comes from. See RESULTS.md.

It shows your melds, your deadwood, and what you would knock at after your best
discard, then lists your legal moves worst-card-first. Type the number, or `q` to
quit.

## Use it in code

`GinRummyBot` is API-compatible with an rlcard agent, so it drops straight into
`env.set_agents([...])`:

```python
import rlcard
from gin_rummy_bot import GinRummyBot

bot = GinRummyBot("results/v2_final/gin_rummy_bot.policy.pt")
env = rlcard.make("gin-rummy")
env.set_agents([bot, some_opponent])
trajectories, payoffs = env.run()
```

It loads either a policy-only `.policy.pt` or a full pickled NFSP agent. Two
defaults are on because both measured stronger:

- `greedy=True` — argmax rather than sampling, worth 2–3 points/hand. Pass
  `greedy=False` for a mixed, less exploitable strategy.
- `smart_draw=True` — the draw/pick-up choice uses an explicit heuristic instead
  of the network. For **this** bot it is worth only +0.41 points/hand, because it
  was cloned from a teacher that makes that decision deterministically and the
  network learned the rule itself: the network alone still beats the smart-draw
  rule agent, at +1.89 ± 1.11. (For the previous bot the same flag was worth ~11
  points/hand.) Pass `smart_draw=False` for the pure network.

To score with true gin rummy points rather than RLCard's default shaped reward:

```python
from train_gin_rummy import make_env
class Args: seed = 0; reward = "points"
env = make_env(Args)
```

## Benchmark it

```bash
.venv/Scripts/python.exe benchmark.py results/v2_final/gin_rummy_bot.policy.pt --greedy --vs smart rule random --num_games 2000
```

Any two agents can be matched directly. `random`, `rule` (rlcard's novice agent),
`smart` (that agent with the coin-flip draw fixed — the bar) and `strong` (the
deterministic teacher in `teacher.py`) are keywords; anything else is a model
path:

```bash
.venv/Scripts/python.exe head_to_head.py results/v2_final/gin_rummy_bot.policy.pt smart --num_games 2000
```

Both swap seats at the halfway point so neither side keeps a dealer advantage,
and report a 95% confidence interval rather than a bare average.
