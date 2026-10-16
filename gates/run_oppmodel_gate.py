import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Value of opponent-model conditioning vs a weak early-knocker (full rules).

adv_t10_p100 = v3search conditioned on 'opponent knocks at deadwood<=10'
(correct for smart); adv_t0_p100 = wrong model control; v3search = no model;
teacher = the patient heuristic whose extraction we want to recover.
"""
import multiprocessing as mp

import meta

if __name__ == "__main__":
    with mp.Pool(12) as pool:
        for name in ("adv_t10_p100", "adv_t0_p100", "v3search", "teacher"):
            m, ci, n = meta.match(name, "smart", 1500, 12, pool,
                                  seed=45100, rules="full")
            print(f"{name:>13} vs smart [full]: {m:+.2f} +/- {ci:.2f} ({n})",
                  flush=True)
