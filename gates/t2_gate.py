import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Turn-2 clause gate (2026-09-02): paired-deal match of v8home WITH the
t2book clause (first free draw decision: take iff melds) vs WITHOUT,
same deals, same v8home opponent. The forced-arm confirmation already
measured the intervention at -2.06 +/- 1.64 per affected take
(results/turn2_confirm.log); this gate checks the packaged clause for
machinery interactions at match level (expected: tie leaning +,
~+0.35/hand diluted). Sharded + incremental.

    .venv/Scripts/python.exe t2_gate.py --deals 1500 --workers 12
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

ROWS_PATH = "results/t2_gate_rows.jsonl"


def _play(deal_seed, hero_t2, pol_seed):
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = rnd.current_player_id
    pols = {hero: HomeV8Policy(seed=pol_seed, t2book=hero_t2),
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
        ps = seed * 3907 + k * 5
        rows.append(dict(t2=_play(deal_seed, True, ps),
                         no=_play(deal_seed, False, ps)))
    return rows


def main():
    ap = argparse.ArgumentParser("turn-2 clause paired gate")
    ap.add_argument("--deals", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=83100)
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
    a = np.array([r["t2"] for r in rows])
    b = np.array([r["no"] for r in rows])
    d = a - b
    n = len(rows)
    print(f"=== t2-clause gate, {n} paired deals ===")
    print(f"with clause {a.mean():+.2f}  without {b.mean():+.2f}  "
          f"paired diff {d.mean():+.2f} +/- "
          f"{1.96 * d.std(ddof=1) / math.sqrt(n):.2f}")


if __name__ == "__main__":
    main()
