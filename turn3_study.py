"""
Turn-3+ study (user direction, 2026-09-04): does the take/draw bias
persist at MID-GAME free-draw nodes, and is it conditioned on hand
adjacency and OUTS (unseen cards that meld into the hand) -- the
user's hand-strength proxy? Registration: results/v9_gates.log.

Method: CENSUS-THEN-FORK (the turn2_study forced-arm instrument,
generalized past the first node). Census plays the deal to the END
on the champion line (v8home both seats), recording every hero
free-draw node j = 1, 2, ... (non-opening, take and draw both
legal) with full state + the bot's natural choice. ONE node k is
picked uniformly among eligible nodes (j >= 2; cell-matching in
--cell c2 mode) by random.Random(deal_seed*2+1) -- deterministic
and replay-independent. Both arms then REPLAY the deal (fresh
policies, same seeds -- the proven turn2 fork mechanism) to node k,
state-checked against the census, force TAKE / DRAW, and play to
the end. Paired per deal; one row per deal; full state stored so
any future rule is re-sliceable offline (re-slice, never re-run).

    .venv/Scripts/python.exe turn3_study.py --deals 2000 --workers 12
    .venv/Scripts/python.exe turn3_study.py --report
    .venv/Scripts/python.exe turn3_study.py --cell c2 --seed 92000 \
        --deals 1400 --out results/turn3_c2_rows.jsonl
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

ROWS_PATH = "results/turn3_rows.jsonl"


def _adj(hand, card):
    fr, fs = card % 13, card // 13
    return sum(1 for c in hand
               if c % 13 == fr or (c // 13 == fs and abs(c % 13 - fr) <= 2))


def _outs(hand, seen):
    """Distinct unseen cards that meld into `hand` (contribute zero
    deadwood) -- the user's hand-strength measure."""
    dw = min_deadwood_ids(list(hand))
    n = 0
    for c in range(52):
        if c in seen:
            continue
        if min_deadwood_ids(list(hand) + [c]) <= dw:
            n += 1
    return n


def _is_free_draw(rnd, hero, legal):
    if rnd.current_player_id != hero or getattr(rnd, "_home_opening", 0):
        return False
    return {type(a).__name__ for a in legal} == \
        {"DrawCardAction", "PickUpDiscardAction"}


def _census(deal_seed, ps, cap=150):
    """Play the whole hand on the champion line; return the list of
    hero free-draw node records (j 1-based)."""
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
        if _is_free_draw(rnd, hero, legal):
            hand = sorted(ids_of(rnd.players[hero].hand))
            nodes.append(dict(
                j=len(nodes) + 1, nmoves=steps, hand=hand,
                v=game.settings.going_out_deadwood_count,
                top=cid(rnd.dealer.discard_pile[-1]),
                pile=ids_of(rnd.dealer.discard_pile),
                opp_known=sorted(ids_of(rnd.players[1 - hero].known_cards)),
                stock=len(rnd.dealer.stock_pile),
                bot_takes=isinstance(action, PickUpDiscardAction)))
        step_fast(game, action)
        steps += 1
    return nodes


def _replay_to_node(deal_seed, ps, k, cap=150):
    """Replay the deal to the k-th hero free-draw node. Returns
    (game, pols, hero, legal) or None (should not happen)."""
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
            if seen == k:
                return game, pols, hero, legal
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    return None


def _finish(game, pols, hero, cap=250):
    rnd = game.round
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    pts = sim._payoff(game) * 100.0
    return pts if hero == 0 else -pts


def _node_feats(node):
    hand, top = node["hand"], node["top"]
    melds = min_deadwood_ids(hand + [top]) <= min_deadwood_ids(hand)
    return dict(melds=bool(melds), adj=_adj(hand, top),
                topv=DEADWOOD_VALUE[top])


def _in_c2(node, f=None):
    f = f or _node_feats(node)
    return not f["melds"] and f["topv"] <= 3 and f["adj"] >= 2


def _shard(job):
    n, seed, cell = job
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
        elig = [nd for nd in nodes if nd["j"] >= 2]
        if cell == "c2":
            elig = [nd for nd in elig if _in_c2(nd)]
        if not elig:
            continue
        k = random.Random(deal_seed * 2 + 1).choice(
            [nd["j"] for nd in elig])
        node = next(nd for nd in nodes if nd["j"] == k)
        hand, top = node["hand"], node["top"]
        feats = _node_feats(node)
        seen_full = set(hand) | set(node["pile"]) | set(node["opp_known"])
        cand = hand + [top]
        best = min(min_deadwood_ids([i for i in cand if i != d])
                   for d in hand)
        drops = [d for d in hand
                 if min_deadwood_ids([i for i in cand if i != d]) == best]
        res = {}
        ok = True
        for arm, want in (("T", PickUpDiscardAction),
                          ("D", DrawCardAction)):
            h2 = _replay_to_node(deal_seed, ps, k)
            if h2 is None:
                ok = False
                break
            g2, pols2, hero2, legal2 = h2
            rnd2 = g2.round
            if sorted(ids_of(rnd2.players[hero2].hand)) != hand or \
                    cid(rnd2.dealer.discard_pile[-1]) != top:
                ok = False                 # census/replay divergence
                break
            step_fast(g2, next(a for a in legal2 if isinstance(a, want)))
            res[arm] = _finish(g2, pols2, hero2)
        if not ok:
            mism += 1
            continue
        rows.append(dict(
            deal_seed=deal_seed, ps=ps, k=k,
            n_free=len(nodes),
            n_elig=sum(1 for nd in nodes if nd["j"] >= 2),
            nmoves=node["nmoves"], stock=node["stock"],
            hand=hand, top=top, pile=node["pile"],
            opp_known=node["opp_known"],
            v=node["v"],
            melds=feats["melds"], adj=feats["adj"], topv=feats["topv"],
            outs=_outs(hand, seen_full),
            outs_pileonly=_outs(hand, set(hand) | set(node["pile"])),
            dead10=max(DEADWOOD_VALUE[d] for d in drops) >= 10,
            bot_takes=node["bot_takes"],
            T=res["T"], D=res["D"], mism=mism))
    return rows


def report(rows):
    uniq = {}
    for r in rows:
        uniq[(r["deal_seed"], r["ps"])] = r
    dropped = len(rows) - len(uniq)
    rows = list(uniq.values())
    n = len(rows)
    T = np.array([r["T"] for r in rows])
    D = np.array([r["D"] for r in rows])
    d = T - D
    melds = np.array([r["melds"] for r in rows])
    adj = np.array([r["adj"] for r in rows])
    topv = np.array([r["topv"] for r in rows])
    outs = np.array([r["outs"] for r in rows])
    j = np.array([r["k"] for r in rows])
    bt = np.array([r["bot_takes"] for r in rows])
    mism = sum(r.get("mism", 0) for r in rows)
    print(f"=== turn-3+ take study, {n} paired deals "
          f"({dropped} dupes dropped, {mism} replay mismatches) ===")
    print(f"node k: mean {j.mean():.1f}  j=2 {100 * (j == 2).mean():.0f}%  "
          f"3-4 {100 * ((j >= 3) & (j <= 4)).mean():.0f}%  "
          f"5+ {100 * (j >= 5).mean():.0f}%")
    print(f"bot takes {100 * bt.mean():.1f}%  card melds "
          f"{100 * melds.mean():.1f}%  outs mean {outs.mean():.1f}")
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"dEV(take) overall: {d.mean():+.2f} +/- {1.96 * se:.2f}")

    def cell(tag, s):
        m = int(s.sum())
        if m < 2:
            print(f"  {tag:30s} n={m:5d}  (too small)")
            return
        x = d[s]
        b = bt[s]
        print(f"  {tag:30s} n={m:5d}  dEV(take) {x.mean():+6.2f} +/- "
              f"{1.96 * x.std(ddof=1) / math.sqrt(m):.2f}  "
              f"(bot takes {100 * b.mean():.0f}%)")
    print("-- C1 (registered primary): bot takes, no meld --")
    c1 = bt & ~melds
    cell("C1: bot takes, no meld", c1)
    cell("  outs 0-4", c1 & (outs <= 4))
    cell("  outs 5-8", c1 & (outs >= 5) & (outs <= 8))
    cell("  outs 9+", c1 & (outs >= 9))
    cell("  adj <= 1", c1 & (adj <= 1))
    cell("  adj >= 2", c1 & (adj >= 2))
    cell("  topv <= 3", c1 & (topv <= 3))
    cell("  topv 4-7", c1 & (topv >= 4) & (topv <= 7))
    cell("  topv 8-10", c1 & (topv >= 8))
    cell("  j = 2", c1 & (j == 2))
    cell("  j 3-4", c1 & (j >= 3) & (j <= 4))
    cell("  j 5+", c1 & (j >= 5))
    print("-- C2 / C2b (registered exception) --")
    c2 = ~melds & (topv <= 3) & (adj >= 2)
    cell("C2: no meld, topv<=3, adj>=2", c2)
    cell("C2b: C2 & outs<=4", c2 & (outs <= 4))
    cell("  C2 & outs>4 (sens.)", c2 & (outs > 4))
    cell("  C2 & outs<=2 (sens.)", c2 & (outs <= 2))
    cell("  C2 & outs<=6 (sens.)", c2 & (outs <= 6))
    print("-- C3 (registered dose-response): no meld, adj>=2 --")
    c3 = ~melds & (adj >= 2)
    m3 = int(c3.sum())
    if m3 >= 10 and outs[c3].var() > 0:
        sl, ic = np.polyfit(outs[c3], d[c3], 1)
        resid = d[c3] - (sl * outs[c3] + ic)
        sse = math.sqrt(resid.var(ddof=2) / (m3 * outs[c3].var()))
        print(f"  OLS d ~ outs (n={m3}): slope {sl:+.2f} +/- "
              f"{1.96 * sse:.2f} (intercept {ic:+.2f})")
    print("-- context cells --")
    cell("melds", melds)
    cell("bot draws, melds", ~bt & melds)
    cell("no meld", ~melds)
    cell("no meld, adj<=1", ~melds & (adj <= 1))
    # dilution: per-hand impact of the operational rule
    # R = melds OR (topv<=3 & adj>=2) [and the C2b outs<=4 variant]
    for tag, rule_takes in (
            ("R (no outs gate)", melds | c2),
            ("R-b (outs<=4 gate)", melds | (c2 & (outs <= 4)))):
        flips = rule_takes != bt
        s = np.where(rule_takes, 1.0, -1.0)
        nf = np.array([r["n_elig"] for r in rows])
        imp = np.where(flips, nf * s * d, 0.0)
        sei = imp.std(ddof=1) / math.sqrt(n)
        print(f"  rule {tag:20s} flips {100 * flips.mean():.1f}% of "
              f"nodes  est per-hand impact {imp.mean():+.2f} +/- "
              f"{1.96 * sei:.2f}")


def main():
    global ROWS_PATH
    ap = argparse.ArgumentParser("turn-3+ take study")
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=90000)
    ap.add_argument("--shard", type=int, default=10)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--cell", default="none", choices=["none", "c2"],
                    help="c2: sample only nodes in the registered "
                         "exception cell (no meld, topv<=3, adj>=2)")
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
