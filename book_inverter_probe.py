"""
v10 probe: the BOOK-INVERTER adversary vs the sharpened opening
likelihood (registered results/v10_gates.log). The candidate sets
tau_po=0 — opening pass/take evidence becomes near-certain — so the
one new attack surface is deliberate book deviation at flip-option
nodes: an opponent who takes exactly when the book says pass (and
passes when it says take) feeds the filter maximally-wrong opening
evidence (clipped only by lik_floor/eps). If the inverter gains
MORE against the candidate hero than against the champion hero, the
sharpening bought exploitability; if not, the clips hold.

Inverter = FastTeacher everywhere except flip-option nodes, where it
plays NOT flip_book_take (a strictly worse opening by the measured
theory -- the question is only whether the belief damage exceeds the
opening EV it gives up).

    .venv/Scripts/python.exe book_inverter_probe.py --deals 1500 --workers 12
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
from home_variant import HomeV8Policy
from sim import cid, fast_legal_actions, flip_book_take, ids_of, step_fast

ROWS_PATH = "results/book_inverter_rows.jsonl"
CAND = dict(tau_po=0.0, eps=0.02)


class BookInverter:
    """FastTeacher with the opening book INVERTED."""

    def __init__(self):
        self.base = sim.FastTeacher()

    def act(self, game, legal):
        rnd = game.round
        opening = getattr(rnd, "_home_opening", 0)
        if opening and len(legal) > 1:
            hand = ids_of(rnd.players[rnd.current_player_id].hand)
            top = cid(rnd.dealer.discard_pile[-1])
            take = not flip_book_take(hand, top, opening)
            want = PickUpDiscardAction if take else DrawCardAction
            for a in legal:
                if isinstance(a, want):
                    return a
        return self.base.act(game, legal)


def _play(deal_seed, hero_cand, ps):
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = int(rnd.current_player_id)
    pols = {hero: HomeV8Policy(seed=ps,
                               pf_cfg=CAND if hero_cand else None),
            1 - hero: BookInverter()}
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
        ps = seed * 12983 + k * 29
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
    print(f"=== book-inverter probe, {n} paired deals "
          f"(hero vs inverter; positive = hero wins) ===")
    print(f"candidate hero {a.mean():+.2f} +/- "
          f"{1.96 * a.std(ddof=1) / math.sqrt(n):.2f}   "
          f"champion hero {b.mean():+.2f} +/- "
          f"{1.96 * b.std(ddof=1) / math.sqrt(n):.2f}")
    print(f"candidate - champion (paired): {d.mean():+.2f} +/- "
          f"{1.96 * d.std(ddof=1) / math.sqrt(n):.2f}   "
          f"[clean iff NOT significantly negative]")


def main():
    global ROWS_PATH
    ap = argparse.ArgumentParser("book-inverter probe")
    ap.add_argument("--deals", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=124000)
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
