import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
P2 speculative-take experiment (user hypothesis, 2026-09-02): as the
SECOND player offered a passed flip, take with adjacency >= 2 even
when it doesn't meld -- P2's pass just hands the move back to P1, so
taking is "an extra turn" and speculation costs P2 less. The opening
study measured +1.98 +/- 2.40 (n=1167) for this cell: suggestive, not
proven.

Design: generate deals, KEEP only the target cell -- the flip does NOT
meld for P2 and P2's adjacency >= 2 (rank-mates + same-suit within 2)
-- force P1 to pass, then play both arms on the same deal:
  armB: P2 takes the flip (forced)
  armC: P2 passes (forced) -> P1's forced stock draw
Both P1 and the downstream P2 are book-v7 (HomeV6Policy). Points are
recorded from P2's PERSPECTIVE (positive = taking helps P2).

Sharded + incremental per the study-harness rule.

    .venv/Scripts/python.exe p2adj_gate.py --deals 3000 --workers 12
    .venv/Scripts/python.exe p2adj_gate.py --report
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DrawCardAction, PickUpDiscardAction)

import sim
from home_variant import HomeV6Policy
from sim import fast_legal_actions, ids_of, min_deadwood_ids, step_fast

ROWS_PATH = "results/p2adj_rows.jsonl"


def _adj(hand, flip):
    fr, fs = flip % 13, flip // 13
    return sum(1 for c in hand
               if c % 13 == fr or (c // 13 == fs and abs(c % 13 - fr) <= 2))


def _force(game, want):
    legal = fast_legal_actions(game)
    for e in legal:
        if isinstance(e, want):
            step_fast(game, e)
            return True
    return False


def _play_out(game, p2_seat, pols, cap=250):
    rnd = game.round
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    pts = sim._payoff(game) * 100.0      # player-0 perspective
    return pts if p2_seat == 0 else -pts


def _shard(job):
    n_cell, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    while len(rows) < n_cell:
        deal_seed = rng.randrange(1 << 30)
        game = sim.new_game_home(deal_seed)
        rnd = game.round
        p1 = rnd.current_player_id
        p2 = 1 - p1
        flip = rnd._home_flip_id
        hand2 = ids_of(rnd.players[p2].hand)
        melds2 = min_deadwood_ids(hand2 + [flip]) <= \
            min_deadwood_ids(hand2)
        a2 = _adj(hand2, flip)
        if melds2 or a2 < 2:
            continue                     # not the target cell
        v = game.settings.going_out_deadwood_count
        ps = seed * 5471 + len(rows) * 11
        res = {}
        for arm, want in (("B", PickUpDiscardAction),
                          ("C", DrawCardAction)):
            g = sim.new_game_home(deal_seed)
            pols = {p1: HomeV6Policy(seed=ps),
                    p2: HomeV6Policy(seed=ps + 1)}
            assert _force(g, DrawCardAction)      # P1 passes
            assert _force(g, want)                # P2's forced choice
            res[arm] = _play_out(g, p2, pols)
        rows.append(dict(v=v, adj=a2, B=res["B"], C=res["C"]))
    return rows


def report(rows):
    n = len(rows)
    B = np.array([r["B"] for r in rows])
    C = np.array([r["C"] for r in rows])
    d = B - C                            # P2 perspective: take - pass
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"=== P2 adj>=2 speculative take, {n} paired cell-deals ===")
    print(f"P2 take (B): {B.mean():+.2f}  P2 pass (C): {C.mean():+.2f}")
    print(f"paired dEV(P2 take): {d.mean():+.2f} +/- {1.96 * se:.2f}")
    v = np.array([r["v"] for r in rows])
    adj = np.array([r["adj"] for r in rows])
    for lo, hi, name in ((1, 3, "A-3"), (4, 7, "4-7"), (8, 10, "8-10")):
        s = (v >= lo) & (v <= hi)
        m = int(s.sum())
        if m < 2:
            continue
        x = d[s]
        print(f"  flip {name:5s} n={m:5d}  "
              f"{x.mean():+6.2f} +/- {1.96 * x.std(ddof=1) / math.sqrt(m):.2f}")
    for a in (2, 3):
        s = adj >= a if a == 3 else adj == a
        m = int(s.sum())
        if m < 2:
            continue
        x = d[s]
        print(f"  adj {'>=3' if a == 3 else '==2'}  n={m:5d}  "
              f"{x.mean():+6.2f} +/- {1.96 * x.std(ddof=1) / math.sqrt(m):.2f}")


def main():
    ap = argparse.ArgumentParser("P2 speculative-take gate")
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=79100)
    ap.add_argument("--shard", type=int, default=15)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        rows = [json.loads(l) for l in open(ROWS_PATH, encoding="utf-8")]
        report(rows)
        return
    open(ROWS_PATH, "w").close()
    n_shards = max(1, args.deals // args.shard)
    jobs = [(args.shard, args.seed + 977 * i) for i in range(n_shards)]
    t0 = time.time()
    done = []
    with mp.Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(_shard, jobs), 1):
            with open(ROWS_PATH, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            done.extend(rows)
            el = time.time() - t0
            eta = el / i * (n_shards - i)
            print(f"shard {i}/{n_shards} ({len(done)} cell-deals) "
                  f"elapsed {el / 60:.0f}m ETA {eta / 60:.0f}m",
                  flush=True)
    report(done)


if __name__ == "__main__":
    main()
