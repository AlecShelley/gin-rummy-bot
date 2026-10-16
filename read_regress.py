"""
v7 likelihood campaign, offline gate 3 (HANDOFF_V7 item 4): targeted
regression suite of READ positions. The truth key
(results/home_examples_truth.txt) makes the belief checkable: for each
position we report the raw 4-filter-ensemble marginal (the same
statistic the evaluator displays and the search samples from) on the
opponent's true hidden cards, under the v6 default PF config and the
candidate v7 config, side by side.

    .venv/Scripts/python.exe read_regress.py
    .venv/Scripts/python.exe read_regress.py --tau_p 0.05

Pass = the v7 config's mean truth-card probability is >= v6's on every
position (the reads sharpen or hold; they must not regress), with the
named incident cards (hard3's 5S/6S, home10's kings, hard1's queens)
printed for eyeballing.
"""

import argparse

import numpy as np

import sim
from evaluate import (FilterEnsemble, build_particle_belief, build_state,
                      history_check, parse_position)
from sim import CARD_ID

# (file, true hidden hand BEFORE removing opp-known, headline cards)
CASES = [
    ("pos_hard3_minus1.txt",
     "4S 5S 6S 7S AH 7H 8H 9H TH AD",
     ["5S", "6S"]),
    ("pos_home10_noknock.txt",
     "2S 5S 6S 7S KS KH AD KD AC KC",
     ["KS", "KH", "KD", "KC"]),
    ("pos_hard1_blocker.txt",
     "AS 3S 4S 5S QS QH 2D 3D QD 2C",
     ["QS", "QH", "QD"]),
]


def ensemble_marginal(pos, game, pf_cfg, filters=4, seed=0):
    reps = []
    for r in range(filters):
        pb, note = build_particle_belief(pos, game, seed + 7919 * r,
                                         pf_cfg=pf_cfg)
        if pb is None:
            raise SystemExit(f"particle belief unavailable: {note}")
        reps.append(pb)
    ens = FilterEnsemble(reps)
    core = ens.beliefs[0].core
    seen = set(core.hero_hand) | set(core.pile) | set(core.opp_known)
    unseen = [c for c in range(52) if c not in seen]
    return ens.raw_marginals(unseen, core.h), core


def run(pf_new, filters=4, seed=0):
    sim.set_rules("full")
    all_ok = True
    for fname, truth_str, headline in CASES:
        with open(fname, encoding="utf-8") as fh:
            pos = parse_position(fh.read())
        status, problems = history_check(pos)
        pos["history_status"] = status
        pos["history_problems"] = problems
        if pos["opp_model"] is None:
            pos["opp_model"] = (pos["knock_max"], 1.0)
        game, _ = build_state(pos)
        truth = [CARD_ID[t] for t in truth_str.split()]
        v6_cfg = dict(pickup_mode="logistic", tau_p=1.0,
                      mh_all_takes=False, eps=0.05, tau_po=None)
        # v6 as gated (frozen; eps/tau_po pinned pre-v10)
        rows = {}
        for tag, cfg in (("v6", v6_cfg), ("v7", pf_new)):
            marg, core = ensemble_marginal(pos, game, cfg, filters, seed)
            hidden_truth = [c for c in truth if c not in core.opp_known]
            rows[tag] = (marg, hidden_truth)
        marg6, hidden_truth = rows["v6"]
        marg7, _ = rows["v7"]
        m6 = float(np.mean([marg6[c] for c in hidden_truth]))
        m7 = float(np.mean([marg7[c] for c in hidden_truth]))
        ok = m7 >= m6 - 0.02              # hold-or-sharpen, small slack
        all_ok = all_ok and ok
        print(f"{fname}: mean P(true hidden card) v6 {m6:.3f} -> "
              f"v7 {m7:.3f}  [{'OK' if ok else 'REGRESSION'}]")
        for name in headline:
            c = CARD_ID[name]
            if c not in hidden_truth:
                print(f"    {name}: publicly known (picked up)")
                continue
            print(f"    {name}: v6 {marg6[c]:.3f} -> v7 {marg7[c]:.3f}")
    print(f"read regression: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser("v7 read-position regression gate")
    ap.add_argument("--tau_p", type=float, default=0.1)
    ap.add_argument("--filters", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = dict(pickup_mode="rule", tau_p=args.tau_p,
               mh_all_takes=True)          # == DEFAULT_PF since promotion
    import sys
    sys.exit(0 if run(cfg, args.filters, args.seed) else 1)
