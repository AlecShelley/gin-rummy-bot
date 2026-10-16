"""
Likelihood-sharpening offline sweep (v10 campaign, registered
results/v10_gates.log 2026-09-06): score PF configs against the
STORED posterior ground-truth rows (results/post_s{1,2,3}_rows.jsonl
— the empirical conditionals measured 2026-09-05). Per (cfg,
scenario): information captured (Brier of the raw ensemble marginal
vs uniform vs achievable), registered evidence-channel absolute
gaps, and the M4 deadwood gap. The sweep SELECTS a candidate; it
confirms nothing (3 scenarios are a tiny validation set — phase B
holds it out on fresh scenarios + the standing gates).

    .venv/Scripts/python.exe post_sweep.py --stage 1
    .venv/Scripts/python.exe post_sweep.py --stage 2
    .venv/Scripts/python.exe post_sweep.py --cfg "tau_p=0.01,lik_floor=0.005"
"""

import argparse
import itertools
import json
import time

import numpy as np

from sim import DEADWOOD_VALUE, min_deadwood_ids
import posterior_check as pc


def _load(name):
    sc = pc.SCEN[name]
    rows = [json.loads(l) for l in open(sc["out"], encoding="utf-8")]
    n = len(rows)
    freq = np.zeros(52)
    for r in rows:
        for c in r["opp"]:
            freq[c] += 1
    freq /= n
    return sc, rows, freq


def _channels(name, sc):
    def mates(card):
        r, s = card % 13, card // 13
        rank = [x for x in range(52) if x % 13 == r and x != card]
        suit = [s * 13 + q for q in range(max(0, r - 2), min(13, r + 3))
                if q != r]
        return [("rank", rank), ("suit", suit)]
    kind = sc.get("kind", name)
    if kind in ("s1", "decline"):
        cards = [sc["flip"]]
    elif kind in ("s2", "take"):
        cards = [sc["flip"], sc["X"]]
    else:
        cards = [sc["flip"], sc["C"], sc["Y"]]
    out = []
    for card in cards:
        out.extend(mates(card))
    return out


def score(name, cache, pf_cfg, filters=4):
    sc, rows, freq = cache[name]
    n = len(rows)
    raw, _, unseen, core, ens = pc._pf_marginals(
        name, sc, filters=filters, pf_cfg=pf_cfg)
    p = freq[unseen]
    ach = float(np.mean(p * (1 - p)))
    base = core.h / len(unseen)
    uni = float(np.mean((base - p) ** 2 + p * (1 - p)))
    q = np.array([float(raw[c]) for c in unseen])
    brier = float(np.mean((q - p) ** 2 + p * (1 - p)))
    cap = (uni - brier) / (uni - ach) if uni > ach else float("nan")
    gap = 0.0
    for _, cards in _channels(name, sc):
        cards = [c for c in cards if c in unseen]
        if cards:
            gap += abs(sum(raw[c] for c in cards)
                       - sum(freq[c] for c in cards))
    thresh = DEADWOOD_VALUE[sc["flip"]]
    emp_dw = np.mean([min_deadwood_ids(list(r["opp"])) for r in rows])
    cl_dw, cl_w = [], []
    for b in ens.beliefs:
        w = b.core.weights()
        for wi, part in zip(w, b.core.particles):
            cl_dw.append(min_deadwood_ids(sorted(core.opp_known)
                                          + [int(x) for x in part]))
            cl_w.append(wi / len(ens.beliefs))
    dw_gap = float(np.sum(np.array(cl_dw) * np.array(cl_w))
                   / np.sum(cl_w) - emp_dw)
    return dict(cap=cap, gap=gap, dw_gap=dw_gap, brier=brier)


def show(tag, res):
    caps = " ".join(f"{k}:{v['cap']*100:5.1f}%" for k, v in res.items())
    gaps = " ".join(f"{v['gap']:.3f}" for v in res.values())
    dws = " ".join(f"{v['dw_gap']:+5.1f}" for v in res.values())
    mean_cap = np.mean([v["cap"] for v in res.values()])
    print(f"{tag:44s} cap[{caps}] mean {mean_cap*100:5.1f}%  "
          f"chgap[{gaps}]  dwgap[{dws}]", flush=True)
    return mean_cap


def parse_cfg(s):
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        k = k.strip()
        if k in ("discard_mode", "pickup_mode"):
            out[k] = v.strip()
        else:
            out[k] = int(v) if k in ("kdraw", "n_particles") else float(v)
    return out


def main():
    ap = argparse.ArgumentParser("posterior sweep")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--scenarios", default="s1,s2,s3")
    ap.add_argument("--filters", type=int, default=4)
    args = ap.parse_args()
    names = args.scenarios.split(",")
    cache = {n: _load(n) for n in names}
    t0 = time.time()

    def run(tag, cfg):
        res = {n: score(n, cache, cfg, filters=args.filters)
               for n in names}
        return show(tag, res)

    if args.cfg:
        run(args.cfg, parse_cfg(args.cfg))
    elif args.stage == 1:
        run("CHAMPION (DEFAULT_PF)", {})
        for tau_p, lf in itertools.product((0.0, 0.01, 0.03, 0.1),
                                           (0.02, 0.005, 0.001)):
            run(f"tau_p={tau_p} lik_floor={lf}",
                dict(tau_p=tau_p, lik_floor=lf))
    elif args.stage == 2:
        run("CHAMPION (DEFAULT_PF)", {})
        for tau_d, mu, eps in itertools.product((0.75, 1.0, 1.5, 2.0),
                                                (0.3, 0.5),
                                                (0.05, 0.02)):
            run(f"tau_d={tau_d} mu={mu} eps={eps}",
                dict(tau_d=tau_d, mu=mu, eps=eps))
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
