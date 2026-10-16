"""
Monte Carlo study of v7 self-play in the HOME variant (user questions,
2026-08-31). Hero = the first player (the former 11-card seat, who
faces the flip decision). Both seats play the promoted v7
(HomeV6Policy, which inherits v7 through DEFAULT_PF), full rules +
standard dead-hand.

Per hand we record:
  flip value | hero's dealt-meld count (max disjoint melds in the 10
  dealt cards) | whether hero's first action takes the flip | outcome
  class (hero_gin / hero_knock / opp_gin / opp_knock / dead) | hero
  points.

    .venv/Scripts/python.exe home_mc.py --hands 4800 --workers 12
    .venv/Scripts/python.exe home_mc.py --report          # from the json

Raw rows land in results/home_mc_rows.json so follow-up questions
never need a re-run.
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import time
from collections import Counter, defaultdict

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    GinAction, KnockAction, PickUpDiscardAction)
from rlcard.games.gin_rummy.utils.melding import get_meld_clusters

import sim
from home_variant import HomeV6Policy
from sim import fast_legal_actions, step_fast

ROWS_PATH = "results/home_mc_rows.json"


def dealt_meld_count(cards):
    """Max number of DISJOINT melds among the dealt cards (0..3)."""
    clusters = get_meld_clusters(hand=list(cards))
    return max((len(c) for c in clusters), default=0)


def _chunk(job):
    hands, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    for h in range(hands):
        game = sim.new_game_home(rng.randrange(1 << 30))
        rnd = game.round
        hero = rnd.current_player_id     # first player, faces the flip
        flip = game.settings.going_out_deadwood_count
        melds0 = dealt_meld_count(rnd.players[hero].hand)
        policies = {hero: HomeV6Policy(seed=seed * 1009 + 2 * h),
                    1 - hero: HomeV6Policy(seed=seed * 1009 + 2 * h + 1)}
        took_flip = None
        steps = 0
        while rnd.going_out_action is None and not rnd.is_over \
                and steps < 250:
            legal = fast_legal_actions(game)
            action = legal[0] if len(legal) == 1 else \
                policies[rnd.current_player_id].act(game, legal)
            if took_flip is None:        # hero's very first decision
                took_flip = isinstance(action, PickUpDiscardAction)
            step_fast(game, action)
            steps += 1
        out = rnd.going_out_action
        who = "hero" if rnd.going_out_player_id == hero else "opp"
        if isinstance(out, GinAction):
            cat = who + "_gin"
        elif isinstance(out, KnockAction):
            cat = who + "_knock"
        else:
            cat = "dead"
        pts = sim._payoff(game) * 100.0
        if hero != 0:
            pts = -pts
        rows.append((flip, melds0, bool(took_flip), cat, float(pts)))
    return rows


def run(hands, workers, seed):
    per = max(1, hands // workers)
    jobs = [(per, seed + 97 * w) for w in range(workers)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = [r for chunk in pool.map(_chunk, jobs) for r in chunk]
    print(f"{len(rows)} hands in {time.time() - t0:.0f}s", flush=True)
    with open(ROWS_PATH, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return rows


def _hist(pts, lo=-120, hi=120, width=10):
    edges = list(range(lo, hi + width, width))
    counts = Counter()
    for p in pts:
        b = min(max(int(math.floor(p / width)) * width, lo), hi - width)
        counts[b] += 1
    n = len(pts)
    lines = []
    for b in range(lo, hi, width):
        c = counts.get(b, 0)
        bar = "#" * int(round(60.0 * c / max(1, max(counts.values()))))
        lines.append(f"  [{b:+4d},{b + width:+4d}) {c:5d} "
                     f"{100.0 * c / n:5.1f}%  {bar}")
    return lines


def report(rows=None):
    if rows is None:
        with open(ROWS_PATH, encoding="utf-8") as fh:
            rows = json.load(fh)
    n = len(rows)
    flips, melds0, took, cats, pts = zip(*rows)
    pts = np.array(pts)
    took = np.array(took, dtype=bool)
    melds0 = np.array(melds0)
    se = pts.std(ddof=1) / math.sqrt(n)
    print(f"=== v7 self-play, home variant, hero = first player "
          f"({n} hands) ===")
    print(f"P(take the flip)          : {100.0 * took.mean():.1f}% "
          f"+/- {196.0 * took.std(ddof=1) / math.sqrt(n):.1f}")
    for m in (1, 2):
        p = (melds0 >= m).mean()
        print(f"P(dealt >= {m} meld{'s' if m > 1 else ' '})       : "
              f"{100.0 * p:.1f}%")
    print("dealt-meld distribution   : " + "  ".join(
        f"{m}:{100.0 * (melds0 == m).mean():.1f}%"
        for m in sorted(set(melds0))))
    print(f"hero points               : mean {pts.mean():+.2f} "
          f"+/- {1.96 * se:.2f} (95% CI), std {pts.std(ddof=1):.1f}")
    print("outcomes:")
    cc = Counter(cats)
    for cat in ("hero_knock", "hero_gin", "opp_knock", "opp_gin",
                "dead"):
        print(f"  {cat:10s}: {100.0 * cc.get(cat, 0) / n:5.1f}%")
    uc_us = sum(1 for c, p in zip(cats, pts)
                if c == "hero_knock" and p < 0)
    uc_them = sum(1 for c, p in zip(cats, pts)
                  if c == "opp_knock" and p > 0)
    print(f"  (hero knocks undercut: {100.0 * uc_us / n:.1f}% of hands;"
          f" hero undercuts their knock: {100.0 * uc_them / n:.1f}%)")
    print("\npoints histogram (all hands):")
    for line in _hist(pts):
        print(line)
    for m in (0, 1, 2):
        sel = melds0 == m if m < 2 else melds0 >= 2
        label = f"{m} dealt melds" if m < 2 else ">=2 dealt melds"
        sub = pts[sel]
        if len(sub) == 0:
            continue
        print(f"\npoints histogram | {label} ({len(sub)} hands, "
              f"mean {sub.mean():+.2f} +/- "
              f"{1.96 * sub.std(ddof=1) / math.sqrt(len(sub)):.2f}, "
              f"std {sub.std(ddof=1):.1f}):")
        for line in _hist(sub):
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("home-variant v7 self-play MC study")
    ap.add_argument("--hands", type=int, default=4800)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=75500)
    ap.add_argument("--report", action="store_true",
                    help="report from the saved rows, no simulation")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        rows = run(args.hands, args.workers, args.seed)
        report(rows)
