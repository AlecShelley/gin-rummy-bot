"""
Turn-2 EXCEPTION gate (2026-09-03): paired-deal match of v8home with
the t2x exception (non-melding take at adj>=2 shedding a dead
10-value card) vs the plain meld-only t2 clause. The constructed-state
factor map measured the exception cell at +0.6..+3 (both playout
conventions); this gate measures it under NATURAL play, where
opponent-discard selection attenuates it. Sharded + incremental.

    .venv/Scripts/python.exe t2x_gate.py --deals 1500 --workers 12
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np

import sim
from home_variant import HomeV8Policy
from sim import fast_legal_actions, step_fast

ROWS_PATH = "results/t2x_gate_rows.jsonl"


def _play(deal_seed, hero_t2x, pol_seed):
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = rnd.current_player_id
    pols = {hero: HomeV8Policy(seed=pol_seed, t2x=hero_t2x),
            1 - hero: HomeV8Policy(seed=pol_seed + 1)}
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < 250:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    pts = sim._payoff(game) * 100.0
    return pts if hero == 0 else -pts


def _shard(job):
    n, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        deal_seed = rng.randrange(1 << 30)
        ps = seed * 2803 + k * 3
        rows.append(dict(x=_play(deal_seed, True, ps),
                         no=_play(deal_seed, False, ps)))
    return rows


def main():
    ap = argparse.ArgumentParser("t2 exception paired gate")
    ap.add_argument("--deals", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=85300)
    ap.add_argument("--shard", type=int, default=20)
    args = ap.parse_args()
    open(ROWS_PATH, "w").close()
    jobs = [(args.shard, args.seed + 977 * i)
            for i in range(max(1, args.deals // args.shard))]
    t0 = time.time()
    rows = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_shard, jobs), 1):
            with open(ROWS_PATH, "a", encoding="utf-8") as fh:
                for row in r:
                    fh.write(json.dumps(row) + "\n")
            rows.extend(r)
            el = time.time() - t0
            print(f"shard {i}/{len(jobs)} ({len(rows)} deals) "
                  f"elapsed {el / 60:.0f}m "
                  f"ETA {el / i * (len(jobs) - i) / 60:.0f}m", flush=True)
    a = np.array([r["x"] for r in rows])
    b = np.array([r["no"] for r in rows])
    d = a - b
    n = len(rows)
    ndiff = int((d != 0).sum())
    print(f"=== t2-exception gate, {n} paired deals ===")
    print(f"with exception {a.mean():+.2f}  without {b.mean():+.2f}  "
          f"paired diff {d.mean():+.2f} +/- "
          f"{1.96 * d.std(ddof=1) / math.sqrt(n):.2f}  "
          f"(games diverged on {ndiff})")


if __name__ == "__main__":
    main()
