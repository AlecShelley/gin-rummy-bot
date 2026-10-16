import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Selection run, roadmap #3 knob: top_discards=8 vs the shipped bot."""
import multiprocessing as mp

import meta

if __name__ == "__main__":
    with mp.Pool(12) as pool:
        m, ci, n = meta.match("v3wide", "v3search", 1500, 12, pool,
                              seed=44100, rules="full")
        print(f"v3wide vs v3search [full]: {m:+.2f} +/- {ci:.2f} ({n})",
              flush=True)
        m, ci, n = meta.match("v3wide", "teacher", 1500, 12, pool,
                              seed=44200, rules="full")
        print(f"v3wide vs teacher  [full]: {m:+.2f} +/- {ci:.2f} ({n})",
              flush=True)
