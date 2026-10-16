"""
Turn-2 study (user direction, 2026-09-02): does the turn-1 bias --
over-taking non-melding cards in gin-race games -- persist at the
FIRST post-opening draw decision? The mechanism (T=16 deadwood leaf
vs 35-step gin races) is not opening-specific, so probably yes.

Method (the instrument that actually beat the bot at turn 1: pooled
forced-arm study with REAL-BOT continuations): play v8home self-play
until P1 faces their first free draw decision (both take and draw
legal), record features + what v8home would choose, then fork:
  armT: force TAKE the pile top   armD: force DRAW from stock
and play both to the end of the hand with v8home on both seats.
Paired per deal. Sharded + incremental per the harness rule.

    .venv/Scripts/python.exe turn2_study.py --deals 2000 --workers 12
    .venv/Scripts/python.exe turn2_study.py --report
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
from sim import cid, fast_legal_actions, ids_of, min_deadwood_ids, step_fast

ROWS_PATH = "results/turn2_rows.jsonl"


def _adj(hand, card):
    fr, fs = card % 13, card // 13
    return sum(1 for c in hand
               if c % 13 == fr or (c // 13 == fs and abs(c % 13 - fr) <= 2))


def _run_to_decision(deal_seed, ps):
    """Play until P1's first free draw decision. Returns (game, pols,
    p1, features) or None if the hand ended / no such node."""
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    p1 = rnd.current_player_id
    pols = {p1: HomeV8Policy(seed=ps),
            1 - p1: HomeV8Policy(seed=ps + 1)}
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < 40:
        legal = fast_legal_actions(game)
        kinds = {type(a).__name__ for a in legal}
        if rnd.current_player_id == p1 and not \
                getattr(rnd, "_home_opening", 0) and \
                kinds == {"DrawCardAction", "PickUpDiscardAction"}:
            hand = ids_of(rnd.players[p1].hand)
            top = cid(rnd.dealer.discard_pile[-1])
            melds = min_deadwood_ids(hand + [top]) <= \
                min_deadwood_ids(hand)
            feats = dict(v=game.settings.going_out_deadwood_count,
                         melds=bool(melds), adj=_adj(hand, top),
                         nmoves=steps)
            return game, pols, p1, feats, legal
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    return None


def _finish(game, pols, p1, cap=250):
    rnd = game.round
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    pts = sim._payoff(game) * 100.0
    return pts if p1 == 0 else -pts


def _shard(job):
    n, seed, cell_only = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    tries = 0
    while len(rows) < n:
        tries += 1
        deal_seed = rng.randrange(1 << 30)
        ps = seed * 7211 + tries * 13
        hit = _run_to_decision(deal_seed, ps)
        if hit is None:
            continue
        game, pols, p1, feats, legal = hit
        bot_action = pols[p1].act(game, legal)
        feats["bot_takes"] = isinstance(bot_action, PickUpDiscardAction)
        if cell_only and not (feats["bot_takes"]
                              and not feats["melds"]):
            continue                     # confirmation targets the
            # bot-takes-no-meld leak cell only
        res = {}
        for arm, want in (("T", PickUpDiscardAction),
                          ("D", DrawCardAction)):
            h2 = _run_to_decision(deal_seed, ps)
            g2, pols2, _, _, legal2 = h2
            step_fast(g2, next(a for a in legal2 if isinstance(a, want)))
            res[arm] = _finish(g2, pols2, p1)
        rows.append(dict(**feats, T=res["T"], D=res["D"]))
    return rows


def report(rows):
    n = len(rows)
    T = np.array([r["T"] for r in rows])
    D = np.array([r["D"] for r in rows])
    d = T - D
    melds = np.array([r["melds"] for r in rows])
    adj = np.array([r["adj"] for r in rows])
    v = np.array([r["v"] for r in rows])
    bt = np.array([r["bot_takes"] for r in rows])
    print(f"=== turn-2 take study, {n} paired deals ===")
    print(f"v8home takes at this node: {100 * bt.mean():.1f}% "
          f"(offered card melds: {100 * melds.mean():.1f}%)")
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"dEV(take) overall: {d.mean():+.2f} +/- {1.96 * se:.2f}")

    def cell(tag, s):
        m = int(s.sum())
        if m < 2:
            return
        x = d[s]
        b = bt[s]
        print(f"  {tag:26s} n={m:5d}  dEV(take) {x.mean():+6.2f} +/- "
              f"{1.96 * x.std(ddof=1) / math.sqrt(m):.2f}  "
              f"(bot takes {100 * b.mean():.0f}%)")
    cell("melds", melds)
    cell("no meld", ~melds)
    for lo, hi, name in ((1, 3, "A-3"), (4, 7, "4-7"), (8, 10, "8-10")):
        sel = (v >= lo) & (v <= hi)
        cell(f"no meld, flip {name}", ~melds & sel)
    cell("no meld, adj>=2", ~melds & (adj >= 2))
    cell("no meld, adj<=1", ~melds & (adj <= 1))
    # the leak cells: where the bot takes but shouldn't / passes but
    # should -- measured on the bot's OWN choices
    cell("bot takes, no meld", bt & ~melds)
    cell("bot draws, melds", ~bt & melds)


def main():
    global ROWS_PATH
    ap = argparse.ArgumentParser("turn-2 take study")
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=81500)
    ap.add_argument("--shard", type=int, default=10)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--cell", action="store_true",
                    help="confirmation mode: only the bot-takes-"
                         "no-meld leak cell")
    ap.add_argument("--out", default=ROWS_PATH)
    args = ap.parse_args()
    ROWS_PATH = args.out
    if args.report:
        report([json.loads(l) for l in open(ROWS_PATH, encoding="utf-8")])
        return
    open(ROWS_PATH, "w").close()
    jobs = [(args.shard, args.seed + 977 * i, args.cell)
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
