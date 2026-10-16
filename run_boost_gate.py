"""Gate roadmap #2: correlated worlds (boost=2.5) vs the shipped sampler."""
import multiprocessing as mp

import meta

if __name__ == "__main__":
    with mp.Pool(12) as pool:
        m, ci, n = meta.match("v3boost", "v3search", 1500, 12, pool,
                              seed=43100, rules="full")
        print(f"v3boost vs v3search [full]: {m:+.2f} +/- {ci:.2f} ({n})",
              flush=True)
        m, ci, n = meta.match("v3boost", "teacher", 1500, 12, pool,
                              seed=43200, rules="full")
        print(f"v3boost vs teacher  [full]: {m:+.2f} +/- {ci:.2f} ({n})",
              flush=True)
