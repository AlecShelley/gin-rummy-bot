"""
DISCARD-SAFETY forced-arm study (leak hunt, registered
results/v9_gates.log 2026-09-04, seed 102000): at hero discard
nodes (no knock legal), does swapping the bot's discard for the
SAFETY-FIRST alternative pay? Challenger = among legal discards
whose resulting min-deadwood <= best+1, the one minimizing
_live_count(card, hero-unseen pool) (ties: higher deadwood value);
nodes where the challenger equals the bot's own discard are
skipped. Census-then-fork with a free natural arm: the census
plays the champion line to the END, which IS the bot-discard arm;
only the challenger arm replays. d = EV(challenger) - EV(bot);
a leak reads d > 0.

    .venv/Scripts/python.exe discard_study.py --deals 2000 --workers 12
    .venv/Scripts/python.exe discard_study.py --report
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, GinAction, KnockAction)

import sim
from home_variant import HomeV8Policy
from sim import (DEADWOOD_VALUE, _live_count, cid, fast_legal_actions,
                 ids_of, min_deadwood_ids, step_fast)
from turn3_study import _finish

ROWS_PATH = "results/discard_rows.jsonl"


def _discard_node(rnd, hero, legal):
    if rnd.current_player_id != hero or len(legal) == 1:
        return False
    if any(isinstance(a, (KnockAction, GinAction)) for a in legal):
        return False
    return all(isinstance(a, DiscardAction) for a in legal)


def _challenger(rnd, hero, legal):
    """The safety-first discard: min hero-unseen live-count among
    discards within +1 deadwood of the best; ties -> higher value.
    Returns (action, feats)."""
    hand_ids = ids_of(rnd.players[hero].hand)
    pile = set(ids_of(rnd.dealer.discard_pile))
    opp_known = set(ids_of(rnd.players[1 - hero].known_cards))
    unseen = [c for c in range(52)
              if c not in hand_ids and c not in pile
              and c not in opp_known]
    scored = []
    for a in legal:
        drop = cid(a.card)
        dw = min_deadwood_ids([i for i in hand_ids if i != drop])
        scored.append((a, drop, dw))
    best_dw = min(s[2] for s in scored)
    window = [s for s in scored if s[2] <= best_dw + 1]
    chall = min(window, key=lambda s: (_live_count(s[1], unseen),
                                       -DEADWOOD_VALUE[s[1]]))
    feats = dict(best_dw=best_dw, chall_card=chall[1],
                 chall_dw=chall[2],
                 chall_live=_live_count(chall[1], unseen))
    return chall[0], feats, unseen


def _census(deal_seed, ps, cap=150):
    """Champion-line playthrough to the END. Returns (nodes, score)
    where nodes are eligible discard nodes (challenger differs from
    the bot's discard) and score is the natural-line hero payoff."""
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
        if _discard_node(rnd, hero, legal):
            chall_a, feats, unseen = _challenger(rnd, hero, legal)
            bot_card = cid(action.card)
            if cid(chall_a.card) != bot_card:
                hand_ids = ids_of(rnd.players[hero].hand)
                bot_dw = min_deadwood_ids(
                    [i for i in hand_ids if i != bot_card])
                nodes.append(dict(
                    j=len(nodes) + 1, nmoves=steps,
                    hand=sorted(hand_ids),
                    v=game.settings.going_out_deadwood_count,
                    pile=ids_of(rnd.dealer.discard_pile),
                    opp_known=sorted(ids_of(
                        rnd.players[1 - hero].known_cards)),
                    stock=len(rnd.dealer.stock_pile),
                    bot_card=bot_card, bot_dw=bot_dw,
                    bot_live=_live_count(bot_card, unseen), **feats))
        step_fast(game, action)
        steps += 1
    pts = sim._payoff(game) * 100.0
    return nodes, (pts if hero == 0 else -pts)


def _replay_force(deal_seed, ps, k, cap=150):
    """Replay to the k-th ELIGIBLE discard node and force the
    challenger discard; returns hero payoff or None."""
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = rnd.current_player_id
    pols = {hero: HomeV8Policy(seed=ps),
            1 - hero: HomeV8Policy(seed=ps + 1)}
    seen = 0
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        if _discard_node(rnd, hero, legal):
            chall_a, _, _ = _challenger(rnd, hero, legal)
            if cid(chall_a.card) != cid(action.card):
                seen += 1
                if seen == k:
                    step_fast(game, chall_a)
                    return _finish(game, pols, hero)
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
        nodes, nat_score = _census(deal_seed, ps)
        if not nodes:
            continue
        k = random.Random(deal_seed * 2 + 1).choice(
            [nd["j"] for nd in nodes])
        node = next(nd for nd in nodes if nd["j"] == k)
        chall_score = _replay_force(deal_seed, ps, k)
        if chall_score is None:
            mism += 1
            continue
        rows.append(dict(
            deal_seed=deal_seed, ps=ps, k=k, n_nodes=len(nodes),
            **{key: node[key] for key in
               ("nmoves", "hand", "v", "pile", "opp_known", "stock",
                "bot_card", "bot_dw", "bot_live", "best_dw",
                "chall_card", "chall_dw", "chall_live")},
            B=nat_score, C=chall_score, mism=mism))
    return rows


def report(rows):
    uniq = {}
    for r in rows:
        uniq[(r["deal_seed"], r["ps"])] = r
    dropped = len(rows) - len(uniq)
    rows = list(uniq.values())
    n = len(rows)
    d = np.array([r["C"] - r["B"] for r in rows])
    v = np.array([r["v"] for r in rows])
    stock = np.array([r["stock"] for r in rows])
    dwcost = np.array([r["chall_dw"] - r["bot_dw"] for r in rows])
    opp_picked = np.array([len(r["opp_known"]) > 0 for r in rows])
    lgain = np.array([r["bot_live"] - r["chall_live"] for r in rows])
    mism = sum(r.get("mism", 0) for r in rows)
    print(f"=== discard-safety study, {n} paired deals "
          f"({dropped} dupes, {mism} replay divergences) ===")
    print(f"eligible nodes/hand mean "
          f"{np.mean([r['n_nodes'] for r in rows]):.1f}   "
          f"live-count saved mean {lgain.mean():.1f}")
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"DA dEV(safety-first - bot) pooled: {d.mean():+.2f} +/- "
          f"{1.96 * se:.2f}   [leak iff > 0]")

    def cell(tag, s):
        m = int(s.sum())
        if m < 2:
            print(f"  {tag:26s} n={m:5d}  (too small)")
            return
        x = d[s]
        print(f"  {tag:26s} n={m:5d}  dEV {x.mean():+6.2f} +/- "
              f"{1.96 * x.std(ddof=1) / math.sqrt(m):.2f}")
    cell("opp pickups = 0", ~opp_picked)
    cell("opp pickups >= 1", opp_picked)
    cell("stock >= 20", stock >= 20)
    cell("stock 10-19", (stock >= 10) & (stock <= 19))
    cell("stock < 10", stock < 10)
    cell("thresh A-3", v <= 3)
    cell("thresh 4-7", (v >= 4) & (v <= 7))
    cell("thresh 8-10", v >= 8)
    cell("dw cost 0 (tied)", dwcost <= 0)
    cell("dw cost 1", dwcost == 1)
    cell("live saved >= 3", lgain >= 3)


def main():
    global ROWS_PATH
    ap = argparse.ArgumentParser("discard-safety study")
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=102000)
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
