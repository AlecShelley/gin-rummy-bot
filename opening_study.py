"""
Opening-theory study (user request, 2026-09-01): when is it right to
take the FIRST card (the flip) in the home variant -- for the first
player AND for the second player, who in the real table rules may take
the flip if the first player passes (stealing the tempo).

Runs on the NATIVE opening rule (implemented 2026-09-01: sim's
_home_opening state machine; a DrawCardAction at an opening node is a
PASS, and passes are public decline evidence in both beliefs):

  arm A : P1 takes the flip (forced), v7 everywhere after.
  arm B : P1 passes (forced) -> P2 takes the flip (forced).
  arm C : P1 passes -> P2 passes -> P1's forced stock draw.

Per deal we also record, for each seat: does the flip meld into the
dealt 10 (min_deadwood(hand+flip) <= min_deadwood(hand)), the teacher
deadwood gain of taking it, the adjacency count (rank-mates + same-
suit neighbors held), plus what v7 itself would choose at P1's opening
and at P2's offer. Points are ALWAYS from P1's perspective.

Derived analysis (--report):
  P1 theory : dEV(take) = A - (B if v7-P2 would take else C),
              binned by flip value band x melds x gain.
  P2 theory : dEV(P2 take) = C - B (P2 perspective), same binning on
              P2's features; the tempo question = its mean on deals
              where the flip does NOT meld for P2.

    .venv/Scripts/python.exe opening_study.py --deals 6000 --workers 12
    .venv/Scripts/python.exe opening_study.py --report
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

ROWS_PATH = "results/opening_rows.json"


def _feats(hand, flip):
    dw0 = min_deadwood_ids(list(hand))
    melds = min_deadwood_ids(list(hand) + [flip]) <= dw0
    cand = list(hand) + [flip]
    after = min(min_deadwood_ids([i for i in cand if i != d])
                for d in hand)
    gain = dw0 - after
    fr, fs = flip % 13, flip // 13
    adj = sum(1 for c in hand
              if c % 13 == fr or (c // 13 == fs and abs(c % 13 - fr) <= 2))
    return dict(melds=bool(melds), gain=int(gain), adj=int(adj),
                dw0=int(dw0))


def _play_out(game, hero, pol_hero, pol_opp, cap=250):
    rnd = game.round
    policies = {hero: pol_hero, 1 - hero: pol_opp}
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            policies[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    pts = sim._payoff(game) * 100.0
    return pts if hero == 0 else -pts


def _force_first(game, want):
    """Apply the current player's forced first action (pickup/draw)."""
    legal = fast_legal_actions(game)
    for e in legal:
        if isinstance(e, want):
            step_fast(game, e)
            return True
    return False


def _would_take(game, policy):
    """What the policy would do at the current draw decision (no step)."""
    legal = fast_legal_actions(game)
    action = legal[0] if len(legal) == 1 else policy.act(game, legal)
    return isinstance(action, PickUpDiscardAction)


def _chunk(job):
    n_deals, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    for k in range(n_deals):
        deal_seed = rng.randrange(1 << 30)
        game0 = sim.new_game_home(deal_seed)
        rnd0 = game0.round
        hero = rnd0.current_player_id
        flip = rnd0._home_flip_id
        v = game0.settings.going_out_deadwood_count
        f1 = _feats(ids_of(rnd0.players[hero].hand), flip)
        f2 = _feats(ids_of(rnd0.players[1 - hero].hand), flip)

        def fresh_policies():
            return (HomeV6Policy(seed=seed * 7919 + k * 17),
                    HomeV6Policy(seed=seed * 7919 + k * 17 + 1))

        # v7's own choices (probe calls on throwaway copies)
        g = sim.new_game_home(deal_seed)
        p1, _ = fresh_policies()
        p1_takes = _would_take(g, p1)
        g = sim.new_game_home(deal_seed)
        assert _force_first(g, DrawCardAction)       # P1 passes
        _, p2 = fresh_policies()
        p2_takes = _would_take(g, p2)                # P2 at its option

        rec = dict(v=v, f1=f1, f2=f2, p1_takes=bool(p1_takes),
                   p2_takes=bool(p2_takes))

        # arm A: P1 takes the flip
        g = sim.new_game_home(deal_seed)
        p1, p2 = fresh_policies()
        assert _force_first(g, PickUpDiscardAction)
        rec["A"] = _play_out(g, hero, p1, p2)

        # arm B: P1 passes -> P2 takes (native option node)
        g = sim.new_game_home(deal_seed)
        p1, p2 = fresh_policies()
        assert _force_first(g, DrawCardAction)
        assert _force_first(g, PickUpDiscardAction)
        rec["B"] = _play_out(g, hero, p1, p2)

        # arm C: both pass -> P1's forced stock draw (auto in playout)
        g = sim.new_game_home(deal_seed)
        p1, p2 = fresh_policies()
        assert _force_first(g, DrawCardAction)
        assert _force_first(g, DrawCardAction)
        rec["C"] = _play_out(g, hero, p1, p2)

        rows.append(rec)
    return rows


def _ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else 0.0, 99.0
    return float(x.mean()), 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x))


def report(rows):
    n = len(rows)
    print(f"=== opening study, {n} deals x 3 arms ===")
    A = np.array([r["A"] for r in rows])
    B = np.array([r["B"] for r in rows])
    C = np.array([r["C"] for r in rows])
    p2t = np.array([r["p2_takes"] for r in rows])
    passv = np.where(p2t, B, C)          # P1's pass value vs v7-P2 response
    d1 = A - passv
    bands = ((1, 3, "A-3"), (4, 7, "4-7"), (8, 10, "8-10/JQK"))

    print("\n-- P1 theory: dEV(take) = A - pass (v7-P2 responds) --")
    m, c = _ci(d1)
    print(f"all deals: {m:+.2f} +/- {c:.2f}   "
          f"(v7-P1 takes {100 * np.mean([r['p1_takes'] for r in rows]):.1f}%"
          f", v7-P2 would take a passed flip {100 * p2t.mean():.1f}%)")
    for lo, hi, name in bands:
        sel = np.array([lo <= r["v"] <= hi for r in rows])
        for melds in (True, False):
            s = sel & np.array([r["f1"]["melds"] == melds for r in rows])
            if s.sum() < 2:
                continue
            m, c = _ci(d1[s])
            print(f"  flip {name:8s} melds={str(melds):5s} "
                  f"n={int(s.sum()):5d}  dEV(take) {m:+6.2f} +/- {c:.2f}")
    print("  by teacher gain (non-melding deals only):")
    gains = np.array([r["f1"]["gain"] for r in rows])
    nm = np.array([not r["f1"]["melds"] for r in rows])
    for glo, ghi in ((0, 0), (1, 3), (4, 6), (7, 10)):
        s = nm & (gains >= glo) & (gains <= ghi)
        if s.sum() < 2:
            continue
        m, c = _ci(d1[s])
        print(f"    gain {glo}-{ghi}: n={int(s.sum()):5d}  "
              f"dEV(take) {m:+6.2f} +/- {c:.2f}")

    print("\n-- P2 theory: dEV(P2 take | P1 passed) = C - B "
          "(P2 perspective) --")
    d2 = C - B
    m, c = _ci(d2)
    print(f"all deals: {m:+.2f} +/- {c:.2f}")
    for lo, hi, name in bands:
        sel = np.array([lo <= r["v"] <= hi for r in rows])
        for melds in (True, False):
            s = sel & np.array([r["f2"]["melds"] == melds for r in rows])
            if s.sum() < 2:
                continue
            m, c = _ci(d2[s])
            print(f"  flip {name:8s} melds={str(melds):5s} "
                  f"n={int(s.sum()):5d}  dEV(take) {m:+6.2f} +/- {c:.2f}")
    nm2 = np.array([not r["f2"]["melds"] for r in rows])
    m, c = _ci(d2[nm2])
    print(f"  tempo question -- flip does NOT meld for P2: "
          f"{m:+.2f} +/- {c:.2f} (n={int(nm2.sum())})")
    g2 = np.array([r["f2"]["gain"] for r in rows])
    for glo, ghi in ((0, 0), (1, 3), (4, 6), (7, 10)):
        s = nm2 & (g2 >= glo) & (g2 <= ghi)
        if s.sum() < 2:
            continue
        m, c = _ci(d2[s])
        print(f"    gain {glo}-{ghi}: n={int(s.sum()):5d}  "
              f"dEV(P2 take) {m:+6.2f} +/- {c:.2f}")


def main():
    ap = argparse.ArgumentParser("home-variant opening-theory study")
    ap.add_argument("--deals", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=77300)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(json.load(open(ROWS_PATH, encoding="utf-8")))
        return
    per = max(1, args.deals // args.workers)
    jobs = [(per, args.seed + 97 * w) for w in range(args.workers)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        rows = [r for c in pool.map(_chunk, jobs) for r in c]
    print(f"{len(rows)} deals x 3 arms in {time.time() - t0:.0f}s",
          flush=True)
    with open(ROWS_PATH, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    report(rows)


if __name__ == "__main__":
    main()
