import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
v10 likelihood-sharpening match gate (registered
results/v10_gates.log): HomeV8Policy with the candidate PF config
(DEFAULT_PF + tau_po=0.0 + eps=0.02) vs the champion config, paired
full home hands (t2_gate pattern: same deal_seed + policy seed both
arms, hero seat vs champion opponent).

    .venv/Scripts/python.exe t10_gate.py --deals 2000 --workers 12
    .venv/Scripts/python.exe t10_gate.py --report
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

ROWS_PATH = "results/t10_gate_rows.jsonl"
CAND = dict(tau_po=0.0, eps=0.02)


def _play(deal_seed, hero_cand, ps):
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = int(rnd.current_player_id)
    pols = {hero: HomeV8Policy(seed=ps,
                               pf_cfg=CAND if hero_cand else None),
            1 - hero: HomeV8Policy(seed=ps + 1)}
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < 250:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[int(rnd.current_player_id)].act(game, legal)
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
        ps = seed * 12347 + k * 23
        rows.append(dict(deal_seed=deal_seed, ps=ps,
                         cand=_play(deal_seed, True, ps),
                         champ=_play(deal_seed, False, ps)))
    return rows


def report(rows):
    uniq = {}
    for r in rows:
        uniq[(r["deal_seed"], r["ps"])] = r
    rows = list(uniq.values())
    a = np.array([r["cand"] for r in rows])
    b = np.array([r["champ"] for r in rows])
    d = a - b
    n = len(rows)
    print(f"=== v10 candidate gate, {n} paired deals ===")
    print(f"candidate {a.mean():+.2f}  champion {b.mean():+.2f}  "
          f"paired diff {d.mean():+.2f} +/- "
          f"{1.96 * d.std(ddof=1) / math.sqrt(n):.2f}")
    nd = int((d != 0).sum())
    print(f"diverging deals: {nd} ({100 * nd / n:.0f}%)")


def main():
    global ROWS_PATH
    ap = argparse.ArgumentParser("v10 candidate match gate")
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=120000)
    ap.add_argument("--shard", type=int, default=20)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=ROWS_PATH)
    args = ap.parse_args()
    ROWS_PATH = args.out
    if args.report:
        report([json.loads(l) for l in open(ROWS_PATH, encoding="utf-8")])
        return
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
    report(rows)


if __name__ == "__main__":
    main()
