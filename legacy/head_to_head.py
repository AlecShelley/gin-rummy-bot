import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Head-to-head evaluation of two gin rummy agents under true points scoring.

Plays a match of N hands, swapping seats halfway so neither model gets a
dealer/seat advantage, and reports average points per hand (score
differential x100), win rate, and the outcome mix.

Either side may be a model path, `random`, or `rule` (rlcard's novice rule agent).

Usage:
  python head_to_head.py A.policy.pt B.policy.pt --num_games 1000
  python head_to_head.py A.policy.pt rule --num_games 1000
"""

import argparse
import math

import numpy as np
from rlcard.utils import get_device, set_seed

from legacy.train_gin_rummy import make_env, classify_outcome
from gin_rummy_bot import load_opponent


class _Args:
    """Minimal stand-in for the trainer's argparse namespace."""
    reward = "points"
    seed = 0


def play(env, num_games, a_seat):
    """Run games and return (payoffs for A, outcome labels, winner-is-A flags)."""
    payoffs, labels, a_wins = [], [], []
    for _ in range(num_games):
        _, p = env.run(is_training=False)
        payoffs.append(p[a_seat])
        label, winner = classify_outcome(env.game)
        labels.append(label)
        a_wins.append(None if winner is None else winner == a_seat)
    return payoffs, labels, a_wins


def run_match(spec_a, spec_b, num_games=1000, seed=12345, device=None,
              env=None, greedy_a=False, greedy_b=False,
              smart_draw_a=True, smart_draw_b=True):
    """Play a seat-swapped match and return summary statistics for side A."""
    device = device or get_device()
    _Args.seed = seed
    env = env or make_env(_Args)
    a, name_a = load_opponent(spec_a, env.num_actions, device, greedy_a,
                              smart_draw_a)
    b, name_b = load_opponent(spec_b, env.num_actions, device, greedy_b,
                              smart_draw_b)

    half = num_games // 2
    # First half: A is player 0. Second half: A is player 1 (seats swapped), on
    # the same deals, so neither side keeps a dealer or seat advantage.
    env.seed(seed)
    env.set_agents([a, b])
    p1, l1, w1 = play(env, half, a_seat=0)
    env.seed(seed)
    env.set_agents([b, a])
    p2, l2, w2 = play(env, num_games - half, a_seat=1)

    payoffs = np.array(p1 + p2, dtype=float)
    labels = l1 + l2
    wins = [w for w in (w1 + w2) if w is not None]

    n = len(payoffs)
    pts = payoffs * 100.0           # payoff is score differential / 100
    mean = float(pts.mean())
    stderr = float(pts.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    ci = 1.96 * stderr
    decisive = len(wins)
    return {
        "name_a": name_a, "name_b": name_b, "n": n,
        "points_per_hand": mean, "ci95": ci,
        "decisive": decisive, "decisive_rate": decisive / n,
        "win_rate": (sum(wins) / decisive) if decisive else float("nan"),
        "mix": {k: labels.count(k) / n for k in ("gin", "knock", "undercut", "dead")},
        "significant": bool(n > 1 and abs(mean) > ci),
        "winner": (name_a if mean > 0 else name_b) if (n > 1 and abs(mean) > ci) else None,
    }


def print_match(r):
    print(f"\n=== {r['name_a']}  vs  {r['name_b']} ===")
    print(f"hands played         : {r['n']} (seats swapped at halfway)")
    print(f"points/hand for A    : {r['points_per_hand']:+.3f}  "
          f"+/- {r['ci95']:.3f} (95% CI)")
    print(f"decisive hands       : {r['decisive']} ({r['decisive_rate']:.1%})")
    print(f"A win rate (decisive): {r['win_rate']:.1%}")
    print("outcome mix          : "
          + "  ".join(f"{k} {v:.1%}" for k, v in r["mix"].items()))
    print("VERDICT              : "
          + (f"{r['winner']} is stronger (95% confidence)" if r["significant"]
             else "no statistically significant difference"))


def main():
    ap = argparse.ArgumentParser("Gin rummy head-to-head match")
    ap.add_argument("model_a")
    ap.add_argument("model_b")
    ap.add_argument("--num_games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    set_seed(args.seed)
    print_match(run_match(args.model_a, args.model_b, args.num_games, args.seed))


if __name__ == "__main__":
    main()
