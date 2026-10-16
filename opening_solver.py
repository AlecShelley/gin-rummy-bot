"""
Opening SOLVER (user directive 2026-09-02: "calculate every time
instead of hoping a simple rule solves it").

At a flip-option node the standard search is measurably biased (the
T=16 deadwood leaf prices the branch of the game that doesn't matter
in a gin race). The solver replaces the leaf with what the opening
study used as ground truth: FULL-LENGTH paired playouts to the end of
the hand -- take and pass played out in the SAME sampled world, K
worlds, decided by a z-test with the opening book as the prior
(deviate only on significant per-hand evidence; worst case = the
book). One node per hand, so the compute is affordable.

Playout continuation: hero seat = patient FastTeacher, opponent seat =
MixedKnock(threshold, 1.0) -- the champion's gated rollout convention
-- with the book governing any opening decisions inside the playout
(FastTeacher already follows it).

    .venv/Scripts/python.exe opening_solver.py --validate --deals 300
    .venv/Scripts/python.exe opening_solver.py --gate --deals 2000

--validate: per-cell calibration against the 6000-deal study's
ground-truth arms (results/opening_rows.json).
--gate: paired match, solver-bot vs book-bot on identical deals.
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
from endgame import _BSnap, _brestore
from sim import (FastTeacher, MixedKnock, apply_world, cid,
                 fast_legal_actions, flip_book_take, ids_of,
                 make_scratch, min_deadwood_ids, restore, snapshot,
                 step_fast, unseen_pool)

DEFAULT_K = 160


def _playout_full(scratch, hero, threshold, rng, cap=250):
    """Play the scratch game to the END of the hand; hero-seat payoff
    in points. Hero = patient teacher; opponent = MixedKnock(t, 1.0)."""
    rnd = scratch.round
    pols = {hero: FastTeacher(),
            1 - hero: MixedKnock(FastTeacher(), threshold, 1.0,
                                 rng=random.Random(
                                     int(rng.integers(1 << 30))))}
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < cap:
        legal = fast_legal_actions(scratch)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(scratch, legal)
        step_fast(scratch, action)
        steps += 1
    pts = sim._payoff(scratch) * 100.0
    return pts if hero == 0 else -pts


def solve_opening(game, rng, k=DEFAULT_K, z=1.5):
    """(action, diff_mean, diff_se, book_take): the flip-option
    decision by full-playout paired evidence vs the book prior."""
    rnd = game.round
    opening = rnd._home_opening
    hero = rnd.current_player_id
    threshold = game.settings.going_out_deadwood_count
    hand = ids_of(rnd.players[hero].hand)
    top = cid(rnd.dealer.discard_pile[-1])
    book_take = flip_book_take(hand, top, opening)
    legal = fast_legal_actions(game)
    take_a = next(a for a in legal if isinstance(a, PickUpDiscardAction))
    pass_a = next(a for a in legal if isinstance(a, DrawCardAction))

    scratch = make_scratch(game)
    base = snapshot(scratch)
    pool = unseen_pool(scratch, hero)
    diffs = np.empty(k)
    for w in range(k):
        pick = rng.choice(len(pool), size=10, replace=False)
        apply_world(scratch, 1 - hero, [pool[i] for i in pick], rng)
        b = _BSnap(scratch, len(scratch.round.move_sheet),
                   len(scratch.actions))
        step_fast(scratch, take_a)
        p_take = _playout_full(scratch, hero, threshold, rng)
        _brestore(scratch, b)
        step_fast(scratch, pass_a)
        p_pass = _playout_full(scratch, hero, threshold, rng)
        diffs[w] = p_take - p_pass
        restore(scratch, base)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1)) / math.sqrt(k)
    if book_take:
        act = take_a if mean > -z * se or se == 0 else pass_a
    else:
        act = pass_a if mean < z * se or se == 0 else take_a
    return act, mean, se, book_take


# ---------------------------------------------------------------------------
# --validate: does the solver's per-deal estimate reproduce the
# study's ground-truth cells?
# ---------------------------------------------------------------------------

def _val_shard(job):
    n, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        deal_seed = rng.randrange(1 << 30)
        game = sim.new_game_home(deal_seed)
        rnd = game.round
        hero = rnd.current_player_id
        flip = rnd._home_flip_id
        v = game.settings.going_out_deadwood_count
        hand1 = ids_of(rnd.players[hero].hand)
        f1m = min_deadwood_ids(hand1 + [flip]) <= min_deadwood_ids(hand1)
        g = make_scratch(game)
        _, m1, se1, _ = solve_opening(g, nrng)
        # P2 node: force P1 pass, solve for P2
        g2 = make_scratch(game)
        legal = fast_legal_actions(g2)
        step_fast(g2, next(a for a in legal
                           if isinstance(a, DrawCardAction)))
        hand2 = ids_of(g2.round.players[g2.round.current_player_id].hand)
        f2m = min_deadwood_ids(hand2 + [flip]) <= \
            min_deadwood_ids(hand2)
        fr, fs = flip % 13, flip // 13
        adj2 = sum(1 for c in hand2
                   if c % 13 == fr or (c // 13 == fs
                                       and abs(c % 13 - fr) <= 2))
        _, m2, se2, _ = solve_opening(g2, nrng)
        out.append(dict(v=v, f1m=bool(f1m), d1=m1, f2m=bool(f2m),
                        adj2=int(adj2), d2=m2))
    return out


def validate(deals, workers, seed):
    t0 = time.time()
    shard = 10
    jobs = [(shard, seed + 977 * i) for i in range(max(1, deals // shard))]
    rows = []
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_val_shard, jobs), 1):
            rows.extend(r)
            el = time.time() - t0
            print(f"shard {i}/{len(jobs)} ({len(rows)} deals) "
                  f"elapsed {el / 60:.0f}m "
                  f"ETA {el / i * (len(jobs) - i) / 60:.0f}m", flush=True)
    with open("results/solver_validate_rows.json", "w",
              encoding="utf-8") as fh:
        json.dump(rows, fh)
    # cell comparison vs the study ground truth
    study = json.load(open("results/opening_rows.json", encoding="utf-8"))
    sA = np.array([r["A"] for r in study])
    sB = np.array([r["B"] for r in study])
    sC = np.array([r["C"] for r in study])
    s_p2take = sC - sB
    s_f1m = np.array([r["f1"]["melds"] for r in study])
    s_f2m = np.array([r["f2"]["melds"] for r in study])
    s_adj2 = np.array([r["f2"]["adj"] for r in study])
    s_v = np.array([r["v"] for r in study])
    s_pass = np.where([r["p2_takes"] for r in study], sB, sC)
    s_d1 = sA - s_pass
    d1 = np.array([r["d1"] for r in rows])
    d2 = np.array([r["d2"] for r in rows])
    f1m = np.array([r["f1m"] for r in rows])
    f2m = np.array([r["f2m"] for r in rows])
    adj2 = np.array([r["adj2"] for r in rows])
    v = np.array([r["v"] for r in rows])

    def cell(tag, sel_s, val_s, sel_r, val_r):
        a = val_s[sel_s]
        b = val_r[sel_r]
        if len(a) < 2 or len(b) < 2:
            return
        print(f"  {tag:28s} study {a.mean():+6.2f}"
              f"+/-{1.96 * a.std(ddof=1) / math.sqrt(len(a)):.2f}  "
              f"solver {b.mean():+6.2f}"
              f"+/-{1.96 * b.std(ddof=1) / math.sqrt(len(b)):.2f}")

    print(f"validation cells (study ground truth vs solver estimate):")
    cell("P1 melding", s_f1m, s_d1, f1m, d1)
    cell("P1 non-melding", ~s_f1m, s_d1, ~f1m, d1)
    cell("P2 melding", s_f2m, s_p2take, f2m, d2)
    cell("P2 nm adj0", ~s_f2m & (s_adj2 == 0), s_p2take,
         ~f2m & (adj2 == 0), d2)
    cell("P2 nm adj1", ~s_f2m & (s_adj2 == 1), s_p2take,
         ~f2m & (adj2 == 1), d2)
    cell("P2 nm adj2+", ~s_f2m & (s_adj2 >= 2), s_p2take,
         ~f2m & (adj2 >= 2), d2)
    cell("P2 nm adj0 A-3", ~s_f2m & (s_adj2 == 0) & (s_v <= 3),
         s_p2take, ~f2m & (adj2 == 0) & (v <= 3), d2)
    cell("P2 nm adj2+ A-3", ~s_f2m & (s_adj2 >= 2) & (s_v <= 3),
         s_p2take, ~f2m & (adj2 >= 2) & (v <= 3), d2)


# ---------------------------------------------------------------------------
# --gate: paired match, solver-bot vs book-bot on identical deals.
# ---------------------------------------------------------------------------

def _gate_shard(job):
    n, seed = job
    from home_variant import HomeV6Policy
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        deal_seed = rng.randrange(1 << 30)
        ps = seed * 6113 + k * 9
        res = {}
        for arm, use_solver in (("solver", True), ("book", False)):
            game = sim.new_game_home(deal_seed)
            rnd = game.round
            hero = rnd.current_player_id
            pols = {hero: HomeV6Policy(seed=ps, solver=use_solver),
                    1 - hero: HomeV6Policy(seed=ps + 1)}
            steps = 0
            while rnd.going_out_action is None and not rnd.is_over \
                    and steps < 250:
                legal = fast_legal_actions(game)
                action = legal[0] if len(legal) == 1 else \
                    pols[rnd.current_player_id].act(game, legal)
                step_fast(game, action)
                steps += 1
            pts = sim._payoff(game) * 100.0
            res[arm] = pts if hero == 0 else -pts
        rows.append(res)
    return rows


def gate(deals, workers, seed):
    shard = 20
    jobs = [(shard, seed + 977 * i)
            for i in range(max(1, deals // shard))]
    t0 = time.time()
    rows = []
    path = "results/solver_gate_rows.jsonl"
    open(path, "w").close()
    with mp.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_gate_shard, jobs), 1):
            with open(path, "a", encoding="utf-8") as fh:
                for row in r:
                    fh.write(json.dumps(row) + "\n")
            rows.extend(r)
            el = time.time() - t0
            print(f"shard {i}/{len(jobs)} ({len(rows)} deals) "
                  f"elapsed {el / 60:.0f}m "
                  f"ETA {el / i * (len(jobs) - i) / 60:.0f}m", flush=True)
    s = np.array([r["solver"] for r in rows])
    b = np.array([r["book"] for r in rows])
    d = s - b
    n = len(rows)
    print(f"=== solver-vs-book gate, {n} paired deals ===")
    print(f"solver {s.mean():+.2f}  book {b.mean():+.2f}  paired diff "
          f"{d.mean():+.2f} +/- "
          f"{1.96 * d.std(ddof=1) / math.sqrt(n):.2f}")


def main():
    ap = argparse.ArgumentParser("opening solver")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=80200)
    args = ap.parse_args()
    if args.validate:
        validate(args.deals, args.workers, args.seed)
    if args.gate:
        gate(args.deals, args.workers, args.seed)


if __name__ == "__main__":
    main()
