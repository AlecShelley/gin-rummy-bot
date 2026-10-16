"""
V11 conditional-logit fit of the search's discard behavior
(registered results/v11_gates.log). Model: P(drop j | hand11, ctx)
= softmax_j(w_ctx . f_j) over the 11 candidates, per-context weights
(pickup / draw), features
  f1 dw_after - best_dw          f2 live(drop, actor-unseen)
  f3 DEADWOOD_VALUE[drop]        f4 in-hand adjacency
  f5 is-the-drawn-card (draw ctx only)
MLE by plain numpy gradient descent (convex; scipy unusable on this
machine). Held-out split BY HAND. G1 gate: beats the frozen energy
baseline (tau_d=2, mu=0.3 => w=(-0.5,-0.15,0,0,0)) on held-out
log-loss in EVERY context. Also emits the take/decline tau_p
calibration table from the harvested free-draw decisions.

    .venv/Scripts/python.exe discard_fit.py            # fit + report
    .venv/Scripts/python.exe discard_fit.py --take     # tau_p table only
"""

import argparse
import json
import time

import numpy as np

from sim import DEADWOOD_VALUE, _live_count, min_deadwood_ids

ROWS = "results/discard_harvest.jsonl"
NF = 6


def _features(row):
    """(11, NF) feature matrix + target index for one discard row.
    f5 (v11b): MELD PROTECTION = VALUE[drop] + dw_after - dw_before
    -- 0 for pure deadwood, large for meld members whose removal
    strands partners (the home10 four-king diagnosis)."""
    hand = row["hand"]
    seen = set(hand) | set(row["pile"]) | set(row["opp_known"])
    unseen = [c for c in range(52) if c not in seen]
    drawn = row.get("drawn")
    dw11 = min_deadwood_ids(list(hand))
    feats = np.zeros((len(hand), NF))
    dws = np.empty(len(hand))
    for j, drop in enumerate(hand):
        dws[j] = min_deadwood_ids([i for i in hand if i != drop])
        feats[j, 1] = _live_count(drop, unseen)
        feats[j, 2] = DEADWOOD_VALUE[drop]
        feats[j, 3] = sum(1 for c in hand if c != drop and (
            c % 13 == drop % 13 or (c // 13 == drop // 13
                                    and abs(c % 13 - drop % 13) <= 2)))
        feats[j, 4] = 1.0 if drop == drawn else 0.0
        feats[j, 5] = DEADWOOD_VALUE[drop] + dws[j] - dw11
    feats[:, 0] = dws - dws.min()
    try:
        tgt = hand.index(row["disc"])
    except ValueError:
        return None
    return feats, tgt


def _logloss(w, F, T):
    """Mean negative log-likelihood over rows (F: n x 11 x NF)."""
    z = F @ w
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n = len(T)
    return -float(np.mean(np.log(np.maximum(p[np.arange(n), T],
                                            1e-12)))), p


def _fit(F, T, iters=3000, lr=0.05):
    w = np.zeros(NF)
    n = len(T)
    for it in range(iters):
        z = F @ w
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(p)
        onehot[np.arange(n), T] = 1.0
        grad = np.einsum("nj,njf->f", p - onehot, F) / n
        w -= lr * grad
    return w


def load(path=ROWS):
    disc, take = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        (disc if r["kind"] == "discard" else take).append(r)
    return disc, take


def take_table(take):
    print(f"--- take/decline calibration ({len(take)} free-draw "
          f"decisions) ---")
    gains = np.array([r["gain"] for r in take])
    took = np.array([r["took"] for r in take])
    for tag, s in (("gain <= 0", gains <= 0), ("gain == 1", gains == 1),
                   ("gain 2-3", (gains >= 2) & (gains <= 3)),
                   ("gain >= 4", gains >= 4)):
        m = int(s.sum())
        if m:
            print(f"  {tag:10s} n={m:6d}  P(take) {took[s].mean():.3f}")
    spec = took[gains < 1].mean() if (gains < 1).any() else float("nan")
    dec = 1 - took[gains >= 1].mean() if (gains >= 1).any() \
        else float("nan")
    print(f"  rule-mismatch: speculative-take rate {spec:.3f} "
          f"(model 0.5*tau_p -> tau_p {2 * spec:.3f}); "
          f"decline-despite-gain rate {dec:.3f} "
          f"(model tau_p -> tau_p {dec:.3f})")


def main():
    ap = argparse.ArgumentParser("v11 discard fit")
    ap.add_argument("--rows", default=ROWS)
    ap.add_argument("--take", action="store_true")
    args = ap.parse_args()
    disc, take = load(args.rows)
    take_table(take)
    if args.take:
        return
    t0 = time.time()
    baseline = np.array([-0.5, -0.15, 0.0, 0.0, 0.0, 0.0])
    print(f"--- discard fit ({len(disc)} decisions) ---")
    for ctx in ("pickup", "draw"):
        rows = [r for r in disc if r["ctx"] == ctx]
        feats, tgts, hands = [], [], []
        for r in rows:
            out = _features(r)
            if out is None:
                continue
            f, t = out
            if len(r["hand"]) != 11:
                continue
            feats.append(f)
            tgts.append(t)
            hands.append(r["deal_seed"])
        F = np.stack(feats)
        T = np.array(tgts)
        H = np.array(hands)
        split = H % 2 == 0               # held-out split by hand
        Ftr, Ttr = F[split], T[split]
        Fho, Tho = F[~split], T[~split]
        w = _fit(Ftr, Ttr)
        ll_fit, p_fit = _logloss(w, Fho, Tho)
        ll_base, _ = _logloss(baseline, Fho, Tho)
        ll_uni = np.log(11.0)
        verdict = "PASS" if ll_fit < ll_base else "FAIL"
        print(f"ctx {ctx:7s} n={len(T)} (train {split.sum()}, "
              f"held-out {(~split).sum()})")
        print(f"  held-out logloss: fit {ll_fit:.4f}  energy-baseline "
              f"{ll_base:.4f}  uniform {ll_uni:.4f}  [G1 {verdict}]")
        print(f"  weights: dwcost {w[0]:+.3f}  live {w[1]:+.3f}  "
              f"value {w[2]:+.3f}  inhand {w[3]:+.3f}  "
              f"isdrawn {w[4]:+.3f}  meldprot {w[5]:+.3f}")
        if ctx == "draw":
            drawn_t = np.array([bool(F[i, T[i], 4])
                                for i in range(len(T))])
            print(f"  P(discard == drawn): empirical "
                  f"{drawn_t.mean():.3f}")
        print(f"  weights-json: {json.dumps([round(x, 4) for x in w])}")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
