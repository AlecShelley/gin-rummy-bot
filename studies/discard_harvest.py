import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
V11 harvest (registered results/v11_gates.log, 2026-09-07): log the
SEARCH'S OWN decisions from v10home-vs-v10home self-play — the raw
material for the measured discard behavior model. Perfect-info
logging from the acting seat's perspective:
  discard rows: hand11, discarded id, context ("pickup" = the hand
    is complete from a pickup or the 11-card opening take; "draw" =
    post-stock-draw, with the drawn card id), actor-view pile and
    opp-known ids, threshold, stock depth, turn index, seat.
  take rows: every free-draw decision (both take and draw legal):
    hand10, top id, deadwood gain of taking, took?, threshold,
    stock, turn — the tau_p calibration table.

    .venv/Scripts/python.exe discard_harvest.py --hands 10000 --workers 12
"""

import argparse
import json
import multiprocessing as mp
import random
import time

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, DrawCardAction, PickUpDiscardAction)

import sim
from home_variant import HomeV8Policy
from sim import cid, fast_legal_actions, ids_of, min_deadwood_ids, step_fast

OUT = "results/discard_harvest.jsonl"


def _harvest_hand(deal_seed, ps, cap=150):
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    first = int(rnd.current_player_id)
    pols = {first: HomeV8Policy(seed=ps),
            1 - first: HomeV8Policy(seed=ps + 1)}
    rows = []
    # per-seat context for the NEXT discard: ("draw", drawn_id) or
    # ("pickup", None); set by the preceding draw/pickup action.
    pending = {0: None, 1: None}
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(game)
        mover = int(rnd.current_player_id)
        opening = getattr(rnd, "_home_opening", 0)
        action = legal[0] if len(legal) == 1 else \
            pols[mover].act(game, legal)
        kinds = {type(a).__name__ for a in legal}
        me = rnd.players[mover]
        opp = rnd.players[1 - mover]
        if not opening and kinds == {"DrawCardAction",
                                     "PickUpDiscardAction"}:
            hand10 = sorted(ids_of(me.hand))
            top = cid(rnd.dealer.discard_pile[-1])
            before = min_deadwood_ids(hand10)
            after = min(min_deadwood_ids(
                [i for i in hand10 + [top] if i != d]) for d in hand10)
            rows.append(dict(
                kind="take", hand=hand10, top=top,
                gain=before - after,
                took=isinstance(action, PickUpDiscardAction),
                v=game.settings.going_out_deadwood_count,
                stock=len(rnd.dealer.stock_pile), nmoves=steps,
                seat=int(mover == first)))
        if isinstance(action, DiscardAction) and len(legal) > 1:
            ctx = pending.get(mover)
            if ctx is not None and len(me.hand) == 11:
                rows.append(dict(
                    kind="discard", hand=sorted(ids_of(me.hand)),
                    disc=cid(action.card), ctx=ctx[0], drawn=ctx[1],
                    pile=ids_of(rnd.dealer.discard_pile),
                    opp_known=sorted(ids_of(opp.known_cards)),
                    v=game.settings.going_out_deadwood_count,
                    stock=len(rnd.dealer.stock_pile), nmoves=steps,
                    seat=int(mover == first)))
        # update pending BEFORE stepping (the drawn card is the
        # stock top at decision time)
        if isinstance(action, DrawCardAction) and not opening:
            if rnd.dealer.stock_pile:
                pending[mover] = ("draw",
                                  cid(rnd.dealer.stock_pile[-1]))
        elif isinstance(action, PickUpDiscardAction):
            pending[mover] = ("pickup", None)
        step_fast(game, action)
        steps += 1
    return rows


def _shard(job):
    n, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        deal_seed = rng.randrange(1 << 30)
        ps = seed * 14813 + k * 31
        for r in _harvest_hand(deal_seed, ps):
            r["deal_seed"] = deal_seed
            r["ps"] = ps
            rows.append(r)
    return rows


def main():
    global OUT
    ap = argparse.ArgumentParser("v11 discard/take harvest")
    ap.add_argument("--hands", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=128000)
    ap.add_argument("--shard", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    OUT = args.out
    open(OUT, "w").close()
    jobs = [(args.shard, args.seed + 977 * i)
            for i in range(max(1, args.hands // args.shard))]
    t0 = time.time()
    total = 0
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_shard, jobs), 1):
            with open(OUT, "a", encoding="utf-8") as fh:
                for row in r:
                    fh.write(json.dumps(row) + "\n")
            total += len(r)
            el = time.time() - t0
            print(f"shard {i}/{len(jobs)} ({total} rows) "
                  f"elapsed {el / 60:.0f}m "
                  f"ETA {el / i * (len(jobs) - i) / 60:.0f}m", flush=True)


if __name__ == "__main__":
    main()
