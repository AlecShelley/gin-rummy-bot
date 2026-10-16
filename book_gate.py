"""
Opening-book gate (2026-09-02): paired-deal match of v7home WITH the
opening book (take the flip iff it melds) vs WITHOUT (the pre-book
searcher decides), same deals, same v7home-book opponent, hero = first
player. The study predicts a small positive edge concentrated on the
~55-60%% of deals where the two disagree.

Sharded per the study-harness rule: 20-deal shards via imap_unordered,
each appended to a JSONL immediately, progress + ETA printed per shard.

    .venv/Scripts/python.exe book_gate.py --deals 2500 --workers 12
    .venv/Scripts/python.exe book_gate.py --report
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np

import sim
from home_variant import HomeV6Policy
from sim import fast_legal_actions, step_fast

ROWS_PATH = "results/book_gate_rows.jsonl"


def _play(deal_seed, hero_book, pol_seed):
    game = sim.new_game_home(deal_seed)
    rnd = game.round
    hero = rnd.current_player_id
    pols = {hero: HomeV6Policy(seed=pol_seed, book=hero_book),
            1 - hero: HomeV6Policy(seed=pol_seed + 1, book=True)}
    steps = 0
    first = None
    while rnd.going_out_action is None and not rnd.is_over and steps < 250:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        if first is None:
            first = type(action).__name__ == "PickUpDiscardAction"
        step_fast(game, action)
        steps += 1
    pts = sim._payoff(game) * 100.0
    return (pts if hero == 0 else -pts), first


def _shard(job):
    n, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        deal_seed = rng.randrange(1 << 30)
        game = sim.new_game_home(deal_seed)
        v = game.settings.going_out_deadwood_count
        ps = seed * 4409 + k * 7
        with_book, took_b = _play(deal_seed, True, ps)
        without, took_n = _play(deal_seed, False, ps)
        rows.append(dict(v=v, book=with_book, nobook=without,
                         took_book=bool(took_b),
                         took_nobook=bool(took_n)))
    return rows


def report(rows):
    n = len(rows)
    b = np.array([r["book"] for r in rows])
    nb = np.array([r["nobook"] for r in rows])
    d = b - nb
    se = d.std(ddof=1) / math.sqrt(n)
    print(f"=== opening-book gate, {n} paired deals ===")
    print(f"book take rate {100 * np.mean([r['took_book'] for r in rows]):.1f}%"
          f"  no-book {100 * np.mean([r['took_nobook'] for r in rows]):.1f}%")
    print(f"with book   : {b.mean():+.2f} +/- "
          f"{1.96 * b.std(ddof=1) / math.sqrt(n):.2f}")
    print(f"without book: {nb.mean():+.2f} +/- "
          f"{1.96 * nb.std(ddof=1) / math.sqrt(n):.2f}")
    print(f"paired book - nobook: {d.mean():+.2f} +/- {1.96 * se:.2f}")
    diff = np.array([r["took_book"] != r["took_nobook"] for r in rows])
    m = int(diff.sum())
    if m:
        dd = d[diff]
        print(f"openings that DISAGREE ({m} deals): paired diff "
              f"{dd.mean():+.2f} +/- "
              f"{1.96 * dd.std(ddof=1) / math.sqrt(m):.2f}")


def main():
    ap = argparse.ArgumentParser("opening-book paired gate")
    ap.add_argument("--deals", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=78500)
    ap.add_argument("--shard", type=int, default=20)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        rows = [json.loads(l) for l in open(ROWS_PATH, encoding="utf-8")]
        report(rows)
        return
    open(ROWS_PATH, "w").close()
    n_shards = max(1, args.deals // args.shard)
    jobs = [(args.shard, args.seed + 977 * i) for i in range(n_shards)]
    t0 = time.time()
    done_rows = []
    with mp.Pool(args.workers) as pool:
        for i, rows in enumerate(
                pool.imap_unordered(_shard, jobs), 1):
            with open(ROWS_PATH, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            done_rows.extend(rows)
            el = time.time() - t0
            eta = el / i * (n_shards - i)
            print(f"shard {i}/{n_shards} ({len(done_rows)} deals) "
                  f"elapsed {el / 60:.0f}m ETA {eta / 60:.0f}m",
                  flush=True)
    report(done_rows)


if __name__ == "__main__":
    main()
