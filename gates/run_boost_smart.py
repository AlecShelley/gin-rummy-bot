import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Does world realism matter vs an opponent who is actually near a knock?

Prediction: the meldless world bias (sampled 41.6 deadwood vs true 23.3)
makes deviations look safe vs smart; boost=2.5 should recover points here
even though it tied in strong-vs-strong play.
"""
import multiprocessing as mp

import meta

if __name__ == "__main__":
    with mp.Pool(12) as pool:
        for name in ("v3boost", "adv10_boost"):
            m, ci, n = meta.match(name, "smart", 1500, 12, pool,
                                  seed=45100, rules="full")
            print(f"{name:>12} vs smart [full]: {m:+.2f} +/- {ci:.2f} ({n})",
                  flush=True)
