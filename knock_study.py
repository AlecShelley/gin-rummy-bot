"""
KNOCK-TIMING forced-arm study (leak hunt, registered
results/v9_gates.log 2026-09-04, seed 100000): at hero nodes where
knocking is optional (>=1 KnockAction, >=1 non-knock action, no
mandatory gin), fork the hand:
  arm K: force the BEST KNOCK (min residual deadwood -- the
         expert._candidates / MixedKnock rule)
  arm C: force FastTeacher's choice among the non-knock actions
and play both to the end with v8home both seats (census-then-fork,
the turn3_study instrument). One node per hand, sampled uniformly.
Leak read-out: dEV(K-C) split by the bot's OWN choice -- bot knocks
with dEV<0 = over-knocking; bot continues with dEV>0 =
under-knocking.

    .venv/Scripts/python.exe knock_study.py --deals 2000 --workers 12
    .venv/Scripts/python.exe knock_study.py --report
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    GinAction, KnockAction)

import sim
from home_variant import HomeV8Policy
from sim import (DEADWOOD_VALUE, cid, fast_legal_actions, ids_of,
                 min_deadwood_ids, step_fast)
from turn3_study import _finish

ROWS_PATH = "results/knock_rows.jsonl"


def _knock_node(rnd, hero, legal):
    if rnd.current_player_id != hero or len(legal) == 1:
        return False
    if any(isinstance(a, GinAction) for a in legal):
        return False                     # gin is mandatory
    has_knock = any(isinstance(a, KnockAction) for a in legal)
    has_other = any(not isinstance(a, KnockAction) for a in legal)
    return has_knock and has_other


def _best_knock(rnd, hero, legal):
    hand_ids = ids_of(rnd.players[hero].hand)
    best, best_dw = None, 999
    for a in legal:
        if not isinstance(a, KnockAction):
            continue
        drop = cid(a.card)
        dw = min_deadwood_ids([i for i in hand_ids if i != drop])
        if dw < best_dw:
            best, best_dw = a, dw
    return best, best_dw


def _census(deal_seed, ps, cap=150):
    """Champion-line playthrough; returns hero knock-option node
    records (j 1-based)."""
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = rnd.current_player_id
    pols = {hero: HomeV8Policy(seed=ps),
            1 - hero: HomeV8Policy(seed=ps + 1)}
    nodes = []
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        if _knock_node(rnd, hero, legal):
            hand = sorted(ids_of(rnd.players[hero].hand))
            _, kdw = _best_knock(rnd, hero, legal)
            nodes.append(dict(
                j=len(nodes) + 1, nmoves=steps, hand=hand,
                v=game.settings.going_out_deadwood_count,
                pile=ids_of(rnd.dealer.discard_pile),
                opp_known=sorted(ids_of(rnd.players[1 - hero].known_cards)),
                stock=len(rnd.dealer.stock_pile), kdw=kdw,
                bot_knocks=isinstance(action, KnockAction)))
        step_fast(game, action)
        steps += 1
    return nodes


def _replay_to_node(deal_seed, ps, k, cap=150):
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = rnd.current_player_id
    pols = {hero: HomeV8Policy(seed=ps),
            1 - hero: HomeV8Policy(seed=ps + 1)}
    seen = 0
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(game)
        if _knock_node(rnd, hero, legal):
            seen += 1
            if seen == k:
                return game, pols, hero, legal
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    return None


def _shard(job):
    n, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    tries = 0
    mism = 0
    while len(rows) < n and tries < n * 3000:
        tries += 1
        deal_seed = rng.randrange(1 << 30)
        ps = seed * 10141 + tries * 19
        nodes = _census(deal_seed, ps)
        if not nodes:
            continue
        k = random.Random(deal_seed * 2 + 1).choice(
            [nd["j"] for nd in nodes])
        node = next(nd for nd in nodes if nd["j"] == k)
        res = {}
        ok = True
        for arm in ("K", "C"):
            h2 = _replay_to_node(deal_seed, ps, k)
            if h2 is None:
                ok = False
                break
            g2, pols2, hero2, legal2 = h2
            rnd2 = g2.round
            if sorted(ids_of(rnd2.players[hero2].hand)) != node["hand"]:
                ok = False
                break
            if arm == "K":
                action, _ = _best_knock(rnd2, hero2, legal2)
            else:
                others = [a for a in legal2
                          if not isinstance(a, KnockAction)]
                action = others[0] if len(others) == 1 else \
                    sim.FastTeacher().act(g2, others)
            step_fast(g2, action)
            res[arm] = _finish(g2, pols2, hero2)
        if not ok:
            mism += 1
            continue
        rows.append(dict(
            deal_seed=deal_seed, ps=ps, k=k, n_nodes=len(nodes),
            **{key: node[key] for key in
               ("nmoves", "hand", "v", "pile", "opp_known", "stock",
                "kdw", "bot_knocks")},
            K=res["K"], C=res["C"], mism=mism))
    return rows


def report(rows):
    uniq = {}
    for r in rows:
        uniq[(r["deal_seed"], r["ps"])] = r
    dropped = len(rows) - len(uniq)
    rows = list(uniq.values())
    n = len(rows)
    d = np.array([r["K"] - r["C"] for r in rows])
    bk = np.array([r["bot_knocks"] for r in rows])
    v = np.array([r["v"] for r in rows])
    stock = np.array([r["stock"] for r in rows])
    kdw = np.array([r["kdw"] for r in rows])
    j = np.array([r["k"] for r in rows])
    mism = sum(r.get("mism", 0) for r in rows)
    print(f"=== knock-timing study, {n} paired deals "
          f"({dropped} dupes, {mism} replay mismatches) ===")
    print(f"bot knocks at the sampled node: {100 * bk.mean():.1f}%   "
          f"nodes/hand mean {np.mean([r['n_nodes'] for r in rows]):.1f}")
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"dEV(knock - continue) overall: {d.mean():+.2f} +/- "
          f"{1.96 * se:.2f}")

    def cell(tag, s):
        m = int(s.sum())
        if m < 2:
            print(f"  {tag:30s} n={m:5d}  (too small)")
            return
        x = d[s]
        print(f"  {tag:30s} n={m:5d}  dEV(K-C) {x.mean():+6.2f} +/- "
              f"{1.96 * x.std(ddof=1) / math.sqrt(m):.2f}")
    print("-- KA (registered): bot knocks -- leak iff < 0 --")
    cell("KA: bot knocks", bk)
    for tag, s in (("thresh A-3", v <= 3), ("thresh 4-7",
                                            (v >= 4) & (v <= 7)),
                   ("thresh 8-10", v >= 8),
                   ("stock >= 20", stock >= 20),
                   ("stock 10-19", (stock >= 10) & (stock <= 19)),
                   ("stock < 10", stock < 10),
                   ("kdw 0-3", kdw <= 3),
                   ("kdw 4-6", (kdw >= 4) & (kdw <= 6)),
                   ("kdw 7-10", kdw >= 7),
                   ("j = 1", j == 1), ("j 2+", j >= 2)):
        cell(f"  KA & {tag}", bk & s)
    print("-- KB (registered): bot continues -- leak iff > 0 --")
    cell("KB: bot continues", ~bk)
    for tag, s in (("thresh A-3", v <= 3), ("thresh 4-7",
                                            (v >= 4) & (v <= 7)),
                   ("thresh 8-10", v >= 8),
                   ("stock >= 20", stock >= 20),
                   ("stock 10-19", (stock >= 10) & (stock <= 19)),
                   ("stock < 10", stock < 10),
                   ("kdw 0-3", kdw <= 3),
                   ("kdw 4-6", (kdw >= 4) & (kdw <= 6)),
                   ("kdw 7-10", kdw >= 7),
                   ("j = 1", j == 1), ("j 2+", j >= 2)):
        cell(f"  KB & {tag}", ~bk & s)


def main():
    global ROWS_PATH
    ap = argparse.ArgumentParser("knock-timing study")
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--shard", type=int, default=10)
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
