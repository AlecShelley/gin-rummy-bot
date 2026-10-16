import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Read-only reslice of results/nat2_rows.jsonl by fine outs bins
(turn-3+ campaign phase 0, 2026-09-04): calibrates the outs<=K
threshold of the registered C2 clause before registration is locked.
Rows are the 430 naturally-conditioned turn-2 states (low card <=3,
non-melding, adj>=2); outs here is the PILE-ONLY convention.

    .venv/Scripts/python.exe nat2_reslice.py
"""

import json
import math

import numpy as np

ROWS = "results/nat2_rows.jsonl"


def cell(tag, d, s):
    m = int(s.sum())
    if m < 2:
        print(f"  {tag:22s} n={m:4d}  (too small)")
        return
    x = d[s]
    print(f"  {tag:22s} n={m:4d}  dEV(take) {x.mean():+6.2f} +/- "
          f"{1.96 * x.std(ddof=1) / math.sqrt(m):.2f}")


def main():
    rows = [json.loads(l) for l in open(ROWS, encoding="utf-8")]
    d = np.array([r["T"] - r["D"] for r in rows])
    outs = np.array([r["outs"] for r in rows])
    adj = np.array([r["adj"] for r in rows])
    v = np.array([r["v"] for r in rows])
    print(f"=== nat2 reslice, {len(rows)} rows (outs = pile-only) ===")
    cell("ALL", d, np.ones(len(d), bool))
    for o in range(0, 7):
        cell(f"outs == {o}", d, outs == o)
    cell("outs 7+", d, outs >= 7)
    for k in (2, 3, 4, 5, 6):
        cell(f"outs <= {k}", d, outs <= k)
        cell(f"outs > {k}", d, outs > k)
    cell("adj == 2", d, adj == 2)
    cell("adj >= 3", d, adj >= 3)
    cell("thresh 8-10", d, v >= 8)
    cell("thresh A-7", d, v <= 7)
    # dose-response: OLS slope of d on outs
    n = len(d)
    sl, ic = np.polyfit(outs, d, 1)
    resid = d - (sl * outs + ic)
    se = math.sqrt(resid.var(ddof=2) / (n * outs.var()))
    print(f"  OLS d ~ outs: slope {sl:+.2f} +/- {1.96 * se:.2f} "
          f"(intercept {ic:+.2f})")


if __name__ == "__main__":
    main()
