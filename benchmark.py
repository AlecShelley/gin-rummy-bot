"""
Run the full benchmark battery against a trained model and emit a markdown table.

    python benchmark.py results/stage2/final.policy.pt \
        --vs random rule results/stage1/final.policy.pt \
        --ladder results/stage2 --num_games 1000 --out results/benchmark.md

`--vs` takes any mix of `random`, `rule`, and model paths. `--ladder DIR` adds
every `snap_ep*.policy.pt` in DIR, which shows whether self-play strength is
improving monotonically or cycling.
"""

import argparse
import glob
import os
import re

from rlcard.utils import set_seed

from head_to_head import run_match, print_match


def ladder_paths(log_dir):
    snaps = glob.glob(os.path.join(log_dir, "snap_ep*.policy.pt"))
    return sorted(snaps, key=lambda p: int(re.search(r"snap_ep(\d+)", p).group(1)))


def main():
    ap = argparse.ArgumentParser("Gin rummy benchmark battery")
    ap.add_argument("champion", help="the model under test")
    ap.add_argument("--vs", nargs="*", default=["random", "rule"],
                    help="opponents: random / rule / paths to models")
    ap.add_argument("--ladder", default=None,
                    help="log dir whose snap_ep*.policy.pt files to also play")
    ap.add_argument("--num_games", type=int, default=1000)
    ap.add_argument("--ladder_games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--greedy", action="store_true",
                    help="champion plays its argmax action (measured ~2-3 "
                         "points/hand stronger than sampling)")
    ap.add_argument("--no_smart_draw", action="store_true",
                    help="champion decides its own draws instead of using the "
                         "draw_decision() heuristic -- measures the network alone")
    ap.add_argument("--out", default=None, help="write a markdown table here")
    args = ap.parse_args()

    set_seed(args.seed)

    jobs = [(spec, args.num_games) for spec in args.vs]
    if args.ladder:
        jobs += [(p, args.ladder_games) for p in ladder_paths(args.ladder)]

    rows = []
    for spec, games in jobs:
        result = run_match(args.champion, spec, games, args.seed,
                           greedy_a=args.greedy,
                           smart_draw_a=not args.no_smart_draw)
        print_match(result)
        rows.append(result)

    header = ("| Opponent | Hands | Points/hand (95% CI) | Decisive | Win rate | "
              "Gin | Knock | Undercut | Dead | Verdict |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    lines = []
    for r in rows:
        label = os.path.basename(r["name_b"]) if os.path.sep in r["name_b"] \
            else r["name_b"]
        verdict = ("**bot wins**" if r["winner"] == r["name_a"]
                   else "opponent wins" if r["significant"] else "tie (n.s.)")
        m = r["mix"]
        lines.append(
            f"| {label} | {r['n']} | {r['points_per_hand']:+.2f} "
            f"± {r['ci95']:.2f} | {r['decisive_rate']:.0%} | {r['win_rate']:.1%} | "
            f"{m['gin']:.1%} | {m['knock']:.1%} | {m['undercut']:.1%} | "
            f"{m['dead']:.1%} | {verdict} |")
    table = header + "\n".join(lines) + "\n"

    print("\n\n" + table)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(f"# Benchmark: `{args.champion}`\n\n"
                     f"Points/hand is from the bot's perspective; seats are swapped "
                     f"at the halfway point of every match.\n\n{table}")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
