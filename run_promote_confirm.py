"""Promotion gate: adv10_boost vs v3search, 4000 hands, fresh seed."""
import multiprocessing as mp

import meta

if __name__ == "__main__":
    with mp.Pool(12) as pool:
        m, ci, n = meta.match("adv10_boost", "v3search", 4000, 12, pool,
                              seed=47100, rules="full")
        print(f"PROMOTE? adv10_boost vs v3search [full]: {m:+.2f} +/- "
              f"{ci:.2f} ({n} hands)", flush=True)
