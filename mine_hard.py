"""
Miner for HUMAN-HARD evaluator positions (v7 session, 2026-08-30).

Replays engine-generated home games (home_examples.play_and_record)
and runs a fast evaluator pass (96 worlds, deep, 2-filter particle
ensemble) over every "tension" node -- speculative takes, knock
margins, informed-opponent discards, endgames. Keeps nodes where the
verdict is either COUNTERINTUITIVE (the search overrules the teacher
prior decisively, z >= 2.5 -- if the strong heuristic gets it wrong,
most humans will too) or a genuine NEAR-TIE (top alternatives within
~1 point). Finalists are then re-verified at full resolution across
seeds before shipping.

    .venv/Scripts/python.exe mine_hard.py --seeds 600 --workers 12
"""

import argparse
import multiprocessing as mp
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, DrawCardAction, GinAction, KnockAction,
    PickUpDiscardAction)

import sim
from sim import cid


def _desc(action):
    if isinstance(action, DrawCardAction):
        return "draw"
    if isinstance(action, PickUpDiscardAction):
        return "take"
    if isinstance(action, KnockAction):
        return f"KNOCK-{action.card}"
    if isinstance(action, DiscardAction):
        return f"drop-{action.card}"
    if isinstance(action, GinAction):
        return "GIN"
    return str(action)


def analyze_snap(snap, worlds=96, filters=2, seed=0):
    """Run the evaluator machinery on one snapshot; return a summary
    dict (None for forced/degenerate nodes)."""
    from adaptive import AdaptiveExpert
    from evaluate import (FilterEnsemble, build_particle_belief,
                          build_state, history_check, parse_position)
    from home_examples import pos_text

    pos = parse_position(pos_text(snap, "# mined"))
    pos["history_status"], pos["history_problems"] = history_check(pos)
    if pos["history_status"] != "complete":
        return None
    pos["opp_model"] = (pos["knock_max"], 1.0)
    game, belief = build_state(pos)
    reps = [build_particle_belief(pos, game, seed + 7919 * r)[0]
            for r in range(filters)]
    reps = [b for b in reps if b is not None]
    if not reps:
        return None
    ens = FilterEnsemble(reps)
    ex = AdaptiveExpert(stages=(worlds,), deep=True, deep_from=0,
                        kbranch=3, truncate=16, z=1.5,
                        opp_model=pos["opp_model"], seed=seed,
                        sample_boost=0.0)
    info = ex.analyze(game, ens)
    if "payoff_means" not in info:
        return None
    cands = [_desc(a) for a in info["candidates"]]
    chosen_k = next(k for k, a in enumerate(info["candidates"])
                    if a.action_id == info["chosen"].action_id)
    return dict(
        seed=snap["seed"], dno=snap["decision_no"],
        threshold=snap["threshold"], stock=snap["stock"],
        phase=snap["phase"], opp_pickups=snap["opp_pickups"],
        up_adj=snap["up_adj"], hero_dw=snap["hero_dw"],
        opp_name=snap["opp_name"], worlds=info["worlds"],
        cands=cands, t_idx=info["t_idx"], chosen=chosen_k,
        ev=[round(m * 100, 2) for m in info["payoff_means"]],
        diff=[round(m * 100, 2) for m in info["diff_means"]],
        se=[round(s * 100, 2) for s in info["se"]],
        reason=info["reason"],
    )


def _interesting(s):
    """Cheap pre-filter: nodes with real tension."""
    if s["phase"] == "draw":
        return s["up_adj"] >= 1 or s["stock"] <= 8
    return (s["knock_legal"] or s["opp_pickups"] >= 2
            or s["stock"] <= 8)


def _work(job):
    seed, worlds = job
    sim.set_rules("full")
    from home_examples import play_and_record
    out = []
    for snap in play_and_record(seed):
        if not _interesting(snap):
            continue
        if snap.get("gin_legal"):
            continue
        try:
            r = analyze_snap(snap, worlds=worlds)
        except Exception as e:                    # noqa: BLE001
            out.append({"error": f"{seed} d{snap['decision_no']}: {e}"})
            continue
        if r is not None:
            r["knock_legal"] = snap["knock_legal"]
            out.append(r)
    return out


def classify(r):
    """Which hard-position families does this node hit?"""
    tags = []
    z = [d / s if s > 0 else 0.0 for d, s in zip(r["diff"], r["se"])]
    ch, ti = r["chosen"], r["t_idx"]
    if ch != ti and z[ch] >= 2.5:
        tags.append("overrule")                   # beats the teacher
    if r["phase"] == "draw" and len(r["cands"]) == 2:
        gap = abs(r["diff"][1 - ti])
        if gap <= 0.8:
            tags.append("tie_take")               # coin-flip take
        if r["up_adj"] >= 2 and r["cands"][ch] == "draw" and \
                (ti != ch and z[ch] >= 2.0 or
                 ti == ch and min(r["diff"][k] for k in range(2)
                                  if k != ti) < -2.0):
            tags.append("refuse_spec")            # tempting take refused
    kn = [k for k, c in enumerate(r["cands"]) if c.startswith("KNOCK")]
    if kn:
        k = kn[0]
        margin = r["diff"][k] if k != ti else \
            -max(r["diff"][j] for j in range(len(r["cands"])) if j != ti)
        if abs(margin) <= 1.5:
            tags.append("razor_knock")            # knock ~ play-on
    if r["stock"] <= 6 and ch != ti and z[ch] >= 2.0:
        tags.append("endgame_overrule")
    if r["phase"] == "discard" and r["opp_pickups"] >= 3 and \
            ch != ti and z[ch] >= 2.0:
        tags.append("safety_overrule")
    return tags


def main():
    ap = argparse.ArgumentParser("hard-position miner")
    ap.add_argument("--base", type=int, default=911000)
    ap.add_argument("--seeds", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--worlds", type=int, default=96)
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(args.base + i, args.worlds) for i in range(args.seeds)]
    with mp.Pool(args.workers) as pool:
        results = pool.map(_work, jobs)
    rows = [r for chunk in results for r in chunk]
    errs = [r["error"] for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    import json
    with open("results/mine_hard_rows.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    print(f"{len(rows)} analyzed nodes, {len(errs)} errors, "
          f"{time.time() - t0:.0f}s -> results/mine_hard_rows.json")
    for e in errs[:5]:
        print("  ERR", e)
    hits = {}
    for r in rows:
        for tag in classify(r):
            hits.setdefault(tag, []).append(r)
    for tag, rs in sorted(hits.items()):
        # rank: overrules by z of chosen, ties by smallness of gap
        if "tie" in tag or "razor" in tag:
            rs.sort(key=lambda r: min(
                abs(d) for k, d in enumerate(r["diff"]) if k != r["t_idx"]))
        else:
            rs.sort(key=lambda r: -(r["diff"][r["chosen"]] /
                                    max(r["se"][r["chosen"]], 1e-9)))
        print(f"--- {tag}: {len(rs)}")
        for r in rs[:14]:
            z = r["diff"][r["chosen"]] / max(r["se"][r["chosen"]], 1e-9)
            tbl = "  ".join(
                f"{'*' if k == r['chosen'] else ''}"
                f"{'(T)' if k == r['t_idx'] else ''}{c}:"
                f"{r['ev'][k]:+.1f}"
                for k, c in enumerate(r["cands"]))
            print(f"  s{r['seed']} d{r['dno']:>2} thr{r['threshold']:>2} "
                  f"st{r['stock']:>2} pk{r['opp_pickups']} "
                  f"adj{r['up_adj']} z{z:+.1f} [{r['opp_name'][:7]}] {tbl}")


if __name__ == "__main__":
    main()
