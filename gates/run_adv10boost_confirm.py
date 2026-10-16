import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Thread-1 close-out (fresh seeds):
1. Confirm adv10_boost's extraction vs smart at 4000 hands.
2. Wrong-model safety: what does assuming 'knocks at 10' cost against
   opponents who never knock (teacher, v3search)?
"""
import multiprocessing as mp

import meta

if __name__ == "__main__":
    with mp.Pool(12) as pool:
        m, ci, n = meta.match("adv10_boost", "smart", 4000, 12, pool,
                              seed=46100, rules="full")
        print(f"CONFIRM adv10_boost vs smart    [full]: {m:+.2f} +/- {ci:.2f} "
              f"({n})", flush=True)
        m, ci, n = meta.match("adv10_boost", "teacher", 1500, 12, pool,
                              seed=46200, rules="full")
        print(f"SAFETY  adv10_boost vs teacher  [full]: {m:+.2f} +/- {ci:.2f} "
              f"({n})", flush=True)
        m, ci, n = meta.match("adv10_boost", "v3search", 1500, 12, pool,
                              seed=46300, rules="full")
        print(f"SAFETY  adv10_boost vs v3search [full]: {m:+.2f} +/- {ci:.2f} "
              f"({n})", flush=True)
