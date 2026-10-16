"""
NATURALLY-CONDITIONED turn-2 take study (user methodology, 2026-09-03):
sample fresh deals, play v8home self-play, and KEEP only hands where
the OPPONENT CHOSE to discard a low card (value <= 3) that lands on
hero's first free draw decision, doesn't meld, and has adjacency >= 2
-- the true conditional distribution of the user's scenarios (the
constructed-state probe measured the wrong conditional; its rule gated
at a tie).

Per accepted state: record hero hand / top / pile / threshold / OUTS
(number of distinct unseen cards that meld into hero's hand -- the
user's hand-strength measure) and whether a dead 10-value shed is
available; then fork forced TAKE vs forced DRAW and play both to the
end with v8home on both seats (paired).

Slices of interest (from the recorded features):
  scenario 1: threshold >= 8, dead-10 shed (take ace/low, drop king)
  scenario 2: threshold <= 3, top is a 2/A     (low-game meld race)
  outs strata: dEV(take) by outs -- prediction: worse when outs high.

    .venv/Scripts/python.exe nat2_study.py --accepted 600 --workers 12
    .venv/Scripts/python.exe nat2_study.py --report
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

ROWS_PATH = "results/nat2_rows.jsonl"


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


def _roll_in(deal_seed, ps, cheap=False):
    """Play to hero's first free draw decision. Returns
    (game, pols, hero, state) or None. state records everything.
    cheap=True rolls in with FastTeacher both sides -- the fast
    SCREEN over millions of deals; every accepted state is then
    re-rolled with v8home (the true chooser) and re-checked, so the
    final sample is v8's own line hitting the cell (screen bias:
    deals where v8 hits but the teacher line doesn't are missed --
    a minority, since v8's discard prior IS the teacher)."""
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = rnd.current_player_id
    if cheap:
        pols = {hero: sim.FastTeacher(), 1 - hero: sim.FastTeacher()}
    else:
        pols = {hero: HomeV8Policy(seed=ps),
                1 - hero: HomeV8Policy(seed=ps + 1)}
    steps = 0
    last_discarder = None
    while rnd.going_out_action is None and not rnd.is_over and steps < 40:
        legal = fast_legal_actions(game)
        kinds = {type(a).__name__ for a in legal}
        if rnd.current_player_id == hero and not \
                getattr(rnd, "_home_opening", 0) and \
                kinds == {"DrawCardAction", "PickUpDiscardAction"}:
            if last_discarder != 1 - hero:
                return None              # top is not the OPP'S CHOICE
            hand = ids_of(rnd.players[hero].hand)
            top = cid(rnd.dealer.discard_pile[-1])
            pile = ids_of(rnd.dealer.discard_pile)
            state = dict(hand=hand, top=top, pile=pile,
                         v=game.settings.going_out_deadwood_count)
            return game, pols, hero, state
        mover = rnd.current_player_id
        action = legal[0] if len(legal) == 1 else \
            pols[mover].act(game, legal)
        if type(action).__name__ == "DiscardAction":
            last_discarder = mover
        step_fast(game, action)
        steps += 1
    return None


def _accept(state):
    hand, top = state["hand"], state["top"]
    if DEADWOOD_VALUE[top] > 3:
        return False                     # opp chose a LOW discard
    if min_deadwood_ids(hand + [top]) <= min_deadwood_ids(hand):
        return False                     # melds: the book already takes
    return _adj(hand, top) >= 2


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


def _shard(job):
    n_acc, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    attempts = 0
    while len(rows) < n_acc and attempts < n_acc * 30000:
        attempts += 1
        deal_seed = rng.randrange(1 << 30)
        ps = seed * 9013 + attempts * 17
        # tier 1: cheap teacher-line screen
        hit = _roll_in(deal_seed, ps, cheap=True)
        if hit is None or not _accept(hit[3]):
            continue
        # tier 2: the true v8 line must hit the cell too
        hit = _roll_in(deal_seed, ps)
        if hit is None:
            continue
        game, pols, hero, state = hit
        if not _accept(state):
            continue
        hand, top = state["hand"], state["top"]
        seen = set(hand) | set(state["pile"])
        state["outs"] = _outs(hand, seen)
        cand = hand + [top]
        best = min(min_deadwood_ids([i for i in cand if i != d])
                   for d in hand)
        drops = [d for d in hand
                 if min_deadwood_ids([i for i in cand if i != d]) == best]
        state["dead10"] = max(DEADWOOD_VALUE[d] for d in drops) >= 10
        state["adj"] = _adj(hand, top)
        res = {}
        for arm, want in (("T", PickUpDiscardAction),
                          ("D", DrawCardAction)):
            h2 = _roll_in(deal_seed, ps)
            g2, pols2, _, _ = h2
            legal2 = fast_legal_actions(g2)
            step_fast(g2, next(a for a in legal2
                               if isinstance(a, want)))
            res[arm] = _finish(g2, pols2, hero)
        state["T"] = res["T"]
        state["D"] = res["D"]
        state["attempts"] = attempts
        rows.append(state)
    return rows


def report(rows):
    n = len(rows)
    d = np.array([r["T"] - r["D"] for r in rows])
    v = np.array([r["v"] for r in rows])
    outs = np.array([r["outs"] for r in rows])
    dead10 = np.array([r["dead10"] for r in rows])
    topv = np.array([DEADWOOD_VALUE[r["top"]] for r in rows])
    print(f"=== natural-conditioned t2 take (opp CHOSE a low discard, "
          f"adj>=2, non-melding), {n} paired states ===")

    def cell(tag, s):
        m = int(s.sum())
        if m < 2:
            print(f"  {tag:34s} n={m:4d}  (too few)")
            return
        x = d[s]
        print(f"  {tag:34s} n={m:4d}  dEV(take) {x.mean():+6.2f} +/- "
              f"{1.96 * x.std(ddof=1) / math.sqrt(m):.2f}")
    cell("ALL accepted", np.ones(n, dtype=bool))
    cell("scenario 1: t>=8 & dead-10 shed", (v >= 8) & dead10)
    cell("scenario 2: t<=3 & top<=2", (v <= 3) & (topv <= 2))
    cell("dead-10 shed available", dead10)
    cell("no dead-10 shed", ~dead10)
    print("  -- by OUTS (user hand-strength measure) --")
    for lo, hi, name in ((0, 2, "outs 0-2 (weak hand)"),
                         (3, 5, "outs 3-5"),
                         (6, 8, "outs 6-8"),
                         (9, 99, "outs 9+ (strong hand)")):
        cell(name, (outs >= lo) & (outs <= hi))
    print("  -- outs x dead-10 --")
    cell("weak (outs<=4) & dead-10", (outs <= 4) & dead10)
    cell("strong (outs>=5) & dead-10", (outs >= 5) & dead10)


def main():
    global ROWS_PATH
    ap = argparse.ArgumentParser("naturally-conditioned t2 study")
    ap.add_argument("--accepted", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=86200)
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
            att = sum(x["attempts"] for x in r) / max(len(r), 1)
            print(f"shard {i}/{len(jobs)} ({len(rows)} accepted, "
                  f"~{att:.0f} attempts/accept) elapsed {el / 60:.0f}m "
                  f"ETA {el / i * (len(jobs) - i) / 60:.0f}m", flush=True)
    report(rows)


if __name__ == "__main__":
    main()
