"""
Search the StrongTeacher parameter space against a fixed opponent panel.

This is *not* reinforcement learning -- RL from scratch was measured to fail in
this environment six different ways (see RESULTS.md). It is black-box search over
a handful of integers in a hand-written policy, scored directly on the true
points scorer. Teacher self-play runs at ~200 hands/s, so a config costs seconds.

Panel design matters more than the search algorithm. The obvious panel -- stock
rule, SmartDraw(rule), random -- shares one weakness: all three knock the instant
it is legal. Tuning against only those would reward patience for exploiting that
single habit and would happily produce an agent that never knocks at all. So the
current best teacher is always in the panel as a fourth, patient opponent, and
the worst-case score across the panel is logged next to the mean.

    python tune_teacher.py --hands 600 --workers 8
"""

import argparse
import csv
import multiprocessing as mp
import os
import time

from rlcard.agents import RandomAgent
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent

from smart_draw import SmartDraw, match
from teacher import DEFAULTS, StrongTeacher

# A config scoring below this against a random opponent has stopped going out at
# all. SmartDraw(rule) scores ~+51 and the prototype ~+78, so this is a floor,
# not a target.
RANDOM_FLOOR = 40.0

# Search grid, centred on the measured prototype.
GRID = {
    "knock_at": [0, 1, 2, 3, 5],
    "endgame_stock": [0, 2, 5, 8, 12],
    "draw_gain": [1, 2, 3],
    "w_live": [0, 1],
    "w_safe": [0, 1],
    "w_value": [0, 1],
}


def _opponents(incumbent):
    """The scoring panel. `incumbent` is the current best teacher config."""
    return [
        ("smart", lambda: SmartDraw(GinRummyNoviceRuleAgent())),
        ("rule", GinRummyNoviceRuleAgent),
        ("random", lambda: RandomAgent(num_actions=110)),
        # The guard against tuning into an anti-early-knocker exploit.
        ("patient", lambda: StrongTeacher(**incumbent)),
    ]


def score(job):
    """Play one config against the whole panel. Runs in a worker process."""
    params, incumbent, hands, seed = job
    agent = StrongTeacher(**params)
    row = {}
    for name, build in _opponents(incumbent):
        mean, ci = match(agent, build(), hands, seed)
        row[name] = mean
        row[name + "_ci"] = ci
    # `random` is deliberately excluded from the objective: every config beats
    # it by 50-80 points, so including it would swamp the two-to-four point
    # differences that actually decide the bar. It stays in the panel as a
    # tripwire -- a config that collapses against random is dead-handing, which
    # is the known failure mode of a policy that has stopped knocking.
    scored = [n for n, _ in _opponents(incumbent) if n != "random"]
    row["mean"] = sum(row[n] for n in scored) / len(scored)
    row["worst"] = min(row[n] for n in scored)
    row["ok"] = bool(row["random"] >= RANDOM_FLOOR)   # numpy bool has no __round__
    return params, row


def coordinate_descent(args, pool):
    """Sweep one parameter at a time, keeping the best value found so far.

    Cheaper than a full grid (2250 configs) and enough for six loosely coupled
    integers. Two passes, because the later parameters can change what is
    optimal for the earlier ones.
    """
    best = dict(DEFAULTS)
    incumbent = dict(DEFAULTS)
    history = []
    evaluated = {}

    for sweep in range(args.sweeps):
        for param, values in GRID.items():
            candidates = []
            for value in values:
                cfg = dict(best)
                cfg[param] = value
                key = tuple(sorted(cfg.items()))
                if key in evaluated:
                    continue
                candidates.append(cfg)
            if not candidates:
                continue
            jobs = [(cfg, incumbent, args.hands, args.seed) for cfg in candidates]
            for cfg, row in pool.imap_unordered(score, jobs):
                key = tuple(sorted(cfg.items()))
                evaluated[key] = row
                history.append((cfg, row))
                print(f"  sweep {sweep + 1} {param:<14} "
                      f"{cfg[param]:>3}  mean {row['mean']:+7.2f}  "
                      f"worst {row['worst']:+7.2f}  smart {row['smart']:+6.2f}  "
                      f"patient {row['patient']:+6.2f}  random {row['random']:+6.1f}"
                      f"{'' if row['ok'] else '   REJECTED (random floor)'}",
                      flush=True)
            # Re-read every config seen so far for this coordinate, so a value
            # scored in an earlier sweep still competes.
            options = [(k, v) for k, v in evaluated.items()
                       if v["ok"]
                       and all(dict(k)[p] == best[p] for p in GRID if p != param)]
            if not options:
                print(f"  -> every {param} value failed the random floor; "
                      f"keeping {param}={best[param]}", flush=True)
                continue
            winner = max(options, key=lambda kv: kv[1][args.objective])
            best = dict(winner[0])
            print(f"  -> best {param}={best[param]} "
                  f"({args.objective} {winner[1][args.objective]:+.2f})", flush=True)
    return best, history


def main():
    ap = argparse.ArgumentParser("Tune the StrongTeacher heuristic")
    ap.add_argument("--hands", type=int, default=600,
                    help="hands per opponent per config (seats swapped)")
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=909,
                    help="common random numbers: every config sees the same deals")
    ap.add_argument("--validate_hands", type=int, default=2500)
    ap.add_argument("--validate_seed", type=int, default=31337,
                    help="held-out seed, different from --seed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--objective", choices=["mean", "worst"], default="mean")
    ap.add_argument("--out", default="results/v2_tuning.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        best, history = coordinate_descent(args, pool)

        print(f"\nsearch done in {time.time() - t0:.0f}s; best config: {best}")

        # Winner's curse: the search seed picked these, so re-score the top few
        # on deals none of them were selected against.
        ranked = sorted([ch for ch in history if ch[1]["ok"]],
                        key=lambda ch: ch[1][args.objective], reverse=True)
        seen, finalists = set(), []
        for cfg, _ in ranked:
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                finalists.append(cfg)
            if len(finalists) == 3:
                break
        if all(f != DEFAULTS for f in finalists):
            finalists.append(dict(DEFAULTS))   # always re-check the prototype

        print(f"\nvalidating {len(finalists)} finalists on held-out seed "
              f"{args.validate_seed}, {args.validate_hands} hands/opponent",
              flush=True)
        jobs = [(cfg, dict(DEFAULTS), args.validate_hands, args.validate_seed)
                for cfg in finalists]
        results = list(pool.imap(score, jobs))

    for cfg, row in results:
        tag = "  (prototype)" if cfg == DEFAULTS else ""
        print(f"  {cfg}{tag}\n     mean {row['mean']:+.2f}  worst {row['worst']:+.2f}  "
              f"vs smart {row['smart']:+.2f} +/- {row['smart_ci']:.2f}  "
              f"vs patient {row['patient']:+.2f} +/- {row['patient_ci']:.2f}",
              flush=True)

    winner = max(results, key=lambda cr: cr[1][args.objective])
    print(f"\nWINNER (held out): {winner[0]}  "
          f"{args.objective} {winner[1][args.objective]:+.2f}")

    fields = ["stage"] + list(GRID) + ["ok", "mean", "worst", "smart", "smart_ci",
                                       "rule", "rule_ci", "random", "random_ci",
                                       "patient", "patient_ci"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for stage, rows in (("search", history), ("validate", results)):
            for cfg, row in rows:
                writer.writerow({"stage": stage, **cfg,
                                 **{k: (v if isinstance(v, bool) else round(v, 4))
                                    for k, v in row.items()}})
    print(f"wrote {args.out} ({len(history) + len(results)} rows)")


if __name__ == "__main__":
    main()
