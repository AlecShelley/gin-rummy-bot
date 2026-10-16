import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Ace-flip probe (user hypothesis, 2026-09-01): in ace-threshold home
games the winner gins, so MELDS matter, not deadwood -- taking the ace
as a pure deadwood swap may be a leak (v7 takes the ace flip 95.7% of
the time and LOSES ace games -5.85 +/- 3.32, results/home_mc_rows.json).

Design: paired intervention on identical ace-flip deals, hero = first
player, opponent = unmodified v7 in every arm.
  default  : v7 as promoted (its own flip decision)
  meldonly : first decision overridden -- take the ace ONLY if it
             melds into the dealt 10 (min_deadwood(hand+A) <=
             min_deadwood(hand)); otherwise draw from stock
  decline  : first decision forced to stock draw (structural control:
             if this arm still loses, the deficit is second-player
             advantage in gin races, not the take)

Per deal we also record whether the ace melds at deal and what the
default arm actually did, so results can be conditioned on the
pure-deadwood-take subgroup the hypothesis is about.

    .venv/Scripts/python.exe ace_probe.py --deals 1200 --workers 12
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

ROWS_PATH = "results/ace_probe_rows.json"
ARMS = ("default", "meldonly", "decline")


class FirstMoveOverride:
    """Wrap a policy; override only the hand's FIRST hero decision."""

    def __init__(self, base, mode, ace_melds):
        self.base = base
        self.mode = mode                 # 'meldonly' | 'decline'
        self.ace_melds = ace_melds
        self._done = False

    def act(self, game, legal):
        if not self._done:
            self._done = True
            take = self.mode == "meldonly" and self.ace_melds
            want = PickUpDiscardAction if take else DrawCardAction
            for e in legal:
                if isinstance(e, want):
                    return e
        return self.base.act(game, legal)


def _play(game, hero, hero_policy, opp_policy):
    rnd = game.round
    policies = {hero: hero_policy, 1 - hero: opp_policy}
    steps = 0
    first = None
    while rnd.going_out_action is None and not rnd.is_over and steps < 250:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            policies[rnd.current_player_id].act(game, legal)
        if first is None:
            first = isinstance(action, PickUpDiscardAction)
        step_fast(game, action)
        steps += 1
    pts = sim._payoff(game) * 100.0
    return (pts if hero == 0 else -pts), first


def _chunk(job):
    n_deals, seed, arms = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    found = 0
    while found < n_deals:
        deal_seed = rng.randrange(1 << 30)
        game = sim.new_game_home(deal_seed)
        if game.settings.going_out_deadwood_count != 1:
            continue
        found += 1
        rnd = game.round
        hero = rnd.current_player_id
        hand = ids_of(rnd.players[hero].hand)
        ace = rnd._home_flip_id
        ace_melds = min_deadwood_ids(hand + [ace]) <= \
            min_deadwood_ids(hand)
        rec = {"ace_melds": bool(ace_melds)}
        for arm in arms:
            g = sim.new_game_home(deal_seed)
            h = g.round.current_player_id
            base = HomeV6Policy(seed=seed * 7919 + found * 13)
            if arm == "default":
                pol = base
            else:
                pol = FirstMoveOverride(base, arm, ace_melds)
            opp = HomeV6Policy(seed=seed * 7919 + found * 13 + 1)
            pts, took = _play(g, h, pol, opp)
            rec[arm] = pts
            rec[arm + "_took"] = bool(took)
        rows.append(rec)
    return rows


def report(rows):
    n = len(rows)
    arms = [a for a in ARMS if a in rows[0]]
    print(f"=== ace-flip probe, {n} paired deals x {len(arms)} arms ===")
    melds = np.array([r["ace_melds"] for r in rows])
    print(f"ace melds into the dealt hand: {100 * melds.mean():.1f}% "
          f"of deals")
    took_d = np.array([r["default_took"] for r in rows])
    print(f"default arm takes the flip   : {100 * took_d.mean():.1f}%")
    pts = {a: np.array([r[a] for r in rows]) for a in arms}
    for a in arms:
        se = pts[a].std(ddof=1) / math.sqrt(n)
        print(f"  {a:8s}: mean {pts[a].mean():+6.2f} +/- "
              f"{1.96 * se:.2f}")
    for a, b in (("meldonly", "default"), ("decline", "default"),
                 ("meldonly", "decline")):
        if a not in pts or b not in pts:
            continue
        d = pts[a] - pts[b]
        se = d.std(ddof=1) / math.sqrt(n)
        print(f"  paired {a} - {b}: {d.mean():+6.2f} +/- "
              f"{1.96 * se:.2f}")
    sub = ~melds                         # the hypothesis subgroup:
    m = int(sub.sum())                   # ace does NOT meld at deal
    if m and "meldonly" in pts:
        print(f"subgroup: ace does NOT meld at deal ({m} deals)")
        for a in arms:
            v = pts[a][sub]
            print(f"  {a:8s}: mean {v.mean():+6.2f} +/- "
                  f"{1.96 * v.std(ddof=1) / math.sqrt(m):.2f}")
        d = (pts["meldonly"] - pts["default"])[sub]
        print(f"  paired meldonly - default: {d.mean():+6.2f} +/- "
              f"{1.96 * d.std(ddof=1) / math.sqrt(m):.2f}")


def main():
    global ARMS, ROWS_PATH
    ap = argparse.ArgumentParser("ace-flip take probe")
    ap.add_argument("--deals", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=76100)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--arms", default="default,meldonly,decline")
    ap.add_argument("--out", default=ROWS_PATH)
    args = ap.parse_args()
    ARMS = tuple(args.arms.split(","))
    ROWS_PATH = args.out
    if args.report:
        report(json.load(open(ROWS_PATH, encoding="utf-8")))
        return
    per = max(1, args.deals // args.workers)
    jobs = [(per, args.seed + 97 * w, ARMS)
            for w in range(args.workers)]
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
