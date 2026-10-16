import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Pick the deliverable by a high-power benchmark, not by a noisy training eval.

DAgger's per-round evaluations use 400 hands (95% CI about +/-2.2 points/hand).
Last run they swung from -0.88 to -6.27, and the round that looked best at -0.88
measured -2.74 at 1500 hands -- selecting on that number is mostly selecting
noise. This re-scores every candidate over many more hands, in both sampled and
greedy modes.

Selecting the maximum of many noisy estimates then reporting that maximum
overstates it (the winner's curse), so `--holdout_seed` re-scores the winner on
deals it was not selected against. That second number is the honest one.

    python select_final.py --dir results/v2_dagger --vs smart --num_games 2000
"""

import argparse
import glob
import os
import re

from rlcard.utils import set_seed

from legacy.head_to_head import run_match


def candidates(directory, extra):
    """Every saved policy in `directory`, round order first, plus any extras."""
    paths = glob.glob(os.path.join(directory, "*.policy.pt"))

    def order(path):
        match = re.search(r"round(\d+)", os.path.basename(path))
        return (0, int(match.group(1))) if match else (1, 0)

    return [p for p in sorted(paths, key=order)
            if "best.policy.pt" not in os.path.basename(p)] + list(extra)


def main():
    ap = argparse.ArgumentParser("High-power model selection")
    ap.add_argument("--dir", default="results/v2_dagger")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="further candidate paths, e.g. the initial clone")
    ap.add_argument("--vs", default="smart",
                    help="the opponent to select against; 'smart' is the bar")
    ap.add_argument("--num_games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--holdout_seed", type=int, default=99991,
                    help="different deals, for the honest number")
    ap.add_argument("--holdout_games", type=int, default=2000)
    args = ap.parse_args()

    paths = candidates(args.dir, args.extra)
    if not paths:
        raise SystemExit(f"no *.policy.pt candidates in {args.dir}")

    set_seed(args.seed)
    print(f"selecting against '{args.vs}' over {args.num_games} hands, "
          f"seed {args.seed}\n")
    print(f"{'candidate':<34} {'mode':<8} {'points/hand':>18}   dead", flush=True)
    rows = []
    for path in paths:
        for greedy in (False, True):
            r = run_match(path, args.vs, args.num_games, args.seed, greedy_a=greedy)
            rows.append((path, greedy, r))
            print(f"{os.path.basename(path):<34} "
                  f"{'greedy' if greedy else 'sampled':<8} "
                  f"{r['points_per_hand']:+8.2f} +/- {r['ci95']:.2f}   "
                  f"{r['mix']['dead']:.1%}", flush=True)

    path, greedy, r = max(rows, key=lambda t: t[2]["points_per_hand"])
    mode = "greedy" if greedy else "sampled"
    print(f"\nSELECTED: {path} ({mode}) "
          f"{r['points_per_hand']:+.2f} +/- {r['ci95']:.2f} on the selection seed")

    h = run_match(path, args.vs, args.holdout_games, args.holdout_seed,
                  greedy_a=greedy)
    print(f"HELD OUT (seed {args.holdout_seed}, {h['n']} hands): "
          f"{h['points_per_hand']:+.2f} +/- {h['ci95']:.2f}")
    print("The held-out number is the honest one; the gap is the winner's curse.")


if __name__ == "__main__":
    main()
