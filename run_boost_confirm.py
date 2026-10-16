"""Confirmation run: v3boost vs v3search, 4500 hands, fresh seed."""
import multiprocessing as mp

import meta

if __name__ == "__main__":
    with mp.Pool(12) as pool:
        m, ci, n = meta.match("v3boost", "v3search", 4500, 12, pool,
                              seed=43700, rules="full")
        print(f"CONFIRM v3boost vs v3search [full]: {m:+.2f} +/- {ci:.2f} "
              f"({n} hands)", flush=True)
