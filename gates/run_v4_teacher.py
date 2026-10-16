import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Nail down the one open number: v4 vs the patient teacher, 4000 hands."""
import multiprocessing as mp

import meta

if __name__ == "__main__":
    with mp.Pool(12) as pool:
        m, ci, n = meta.match("v4", "teacher", 4000, 12, pool,
                              seed=48100, rules="full")
        print(f"v4 vs teacher [full]: {m:+.2f} +/- {ci:.2f} ({n} hands)",
              flush=True)
