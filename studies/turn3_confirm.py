import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Turn-3 (j=2) leak CONFIRMATION (registered results/v9_gates.log,
2026-09-04, seed 94000): the phase-1 selection study found the bot's
non-melding takes at the SECOND free-draw node cost -8.67 +/- 7.65
(n=64, one of 11 registered margin cells -- a selection, not a
finding). This instrument confirms it on a fresh seed: roll each
deal to hero's j=2 free-draw node on the champion line, probe the
bot's choice, ACCEPT iff it takes a non-melding card, then fork
(force TAKE / force DRAW via full replay -- the turn2_study
mechanism) and play both arms to the end with v8home both seats.

    .venv/Scripts/python.exe turn3_confirm.py --accepted 1000 --workers 12
    .venv/Scripts/python.exe turn3_confirm.py --report
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
from sim import (DEADWOOD_VALUE, cid, fast_legal_actions, ids_of,
                 min_deadwood_ids, step_fast)
from turn3_study import _adj, _finish, _is_free_draw, _outs

ROWS_PATH = "results/turn3_confirm_rows.jsonl"


def _roll_to_j2(deal_seed, ps, cap=150):
    """Play the champion line to hero's SECOND free-draw node.
    Returns (game, pols, hero, legal) or None."""
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = rnd.current_player_id
    pols = {hero: HomeV8Policy(seed=ps),
            1 - hero: HomeV8Policy(seed=ps + 1)}
    seen = 0
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(game)
        if _is_free_draw(rnd, hero, legal):
            seen += 1
            if seen == 2:
                return game, pols, hero, legal
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    return None


def _shard(job):
    n_acc, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    tried = reached = 0
    while len(rows) < n_acc and tried < n_acc * 300:
        tried += 1
        deal_seed = rng.randrange(1 << 30)
        ps = seed * 10141 + tried * 19
        hit = _roll_to_j2(deal_seed, ps)
        if hit is None:
            continue
        reached += 1
        game, pols, hero, legal = hit
        rnd = game.round
        hand = sorted(ids_of(rnd.players[hero].hand))
        top = cid(rnd.dealer.discard_pile[-1])
        melds = min_deadwood_ids(hand + [top]) <= min_deadwood_ids(hand)
        if melds:
            continue
        bot_action = pols[hero].act(game, legal)
        if not isinstance(bot_action, PickUpDiscardAction):
            continue                     # cell = the bot's OWN takes
        pile = ids_of(rnd.dealer.discard_pile)
        opp_known = sorted(ids_of(rnd.players[1 - hero].known_cards))
        res = {}
        for arm, want in (("T", PickUpDiscardAction),
                          ("D", DrawCardAction)):
            h2 = _roll_to_j2(deal_seed, ps)
            g2, pols2, hero2, legal2 = h2
            step_fast(g2, next(a for a in legal2 if isinstance(a, want)))
            res[arm] = _finish(g2, pols2, hero2)
        rows.append(dict(
            deal_seed=deal_seed, ps=ps, shard_seed=seed,
            hand=hand, top=top, pile=pile,
            opp_known=opp_known,
            v=game.settings.going_out_deadwood_count,
            stock=len(rnd.dealer.stock_pile),
            adj=_adj(hand, top), topv=DEADWOOD_VALUE[top],
            outs=_outs(hand, set(hand) | set(pile) | set(opp_known)),
            tried=tried, reached=reached,
            T=res["T"], D=res["D"]))
    return rows


def report(rows):
    uniq = {}
    for r in rows:
        uniq[(r["deal_seed"], r["ps"])] = r
    dropped = len(rows) - len(uniq)
    rows = list(uniq.values())
    n = len(rows)
    d = np.array([r["T"] - r["D"] for r in rows])
    adj = np.array([r["adj"] for r in rows])
    topv = np.array([r["topv"] for r in rows])
    outs = np.array([r["outs"] for r in rows])
    # tried/reached accumulate within a shard: per-shard max is the
    # shard total
    per_shard = {}
    for r in rows:
        s = r.get("shard_seed")
        t, rc = per_shard.get(s, (0, 0))
        per_shard[s] = (max(t, r["tried"]), max(rc, r["reached"]))
    reached = sum(rc for _, rc in per_shard.values())
    acc = n / reached if reached else float("nan")
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"=== j=2 leak confirmation, {n} accepted rows "
          f"({dropped} dupes dropped) ===")
    print(f"cell rate: {100 * acc:.1f}% of hands reaching j=2 "
          f"({reached} reached)")
    print(f"dEV(take) {d.mean():+.2f} +/- {1.96 * se:.2f}   "
          f"[registered: < 0 confirms]")

    def cell(tag, s):
        m = int(s.sum())
        if m < 2:
            print(f"  {tag:22s} n={m:5d}  (too small)")
            return
        x = d[s]
        print(f"  {tag:22s} n={m:5d}  dEV(take) {x.mean():+6.2f} +/- "
              f"{1.96 * x.std(ddof=1) / math.sqrt(m):.2f}")
    cell("adj <= 1", adj <= 1)
    cell("adj >= 2", adj >= 2)
    cell("topv <= 3", topv <= 3)
    cell("topv 4-7", (topv >= 4) & (topv <= 7))
    cell("topv 8-10", topv >= 8)
    cell("outs <= 4", outs <= 4)
    cell("outs 5+", outs >= 5)


def main():
    global ROWS_PATH
    ap = argparse.ArgumentParser("j=2 leak confirmation")
    ap.add_argument("--accepted", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=94000)
    ap.add_argument("--shard", type=int, default=5)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=ROWS_PATH)
    args = ap.parse_args()
    ROWS_PATH = args.out
    if args.report:
        report([json.loads(l) for l in open(ROWS_PATH, encoding="utf-8")])
        return
    open(ROWS_PATH, "w").close()
    jobs = [(args.shard, args.seed + 977 * i)
            for i in range(max(1, args.accepted // args.shard))]
    t0 = time.time()
    rows = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_shard, jobs), 1):
            with open(ROWS_PATH, "a", encoding="utf-8") as fh:
                for row in r:
                    fh.write(json.dumps(row) + "\n")
            rows.extend(r)
            el = time.time() - t0
            att = (sum(x["tried"] for x in r) / max(1, len(r)))
            print(f"shard {i}/{len(jobs)} ({len(rows)} rows) "
                  f"elapsed {el / 60:.0f}m "
                  f"ETA {el / i * (len(jobs) - i) / 60:.0f}m", flush=True)
    report(rows)


if __name__ == "__main__":
    main()
