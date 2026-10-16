"""Re-measure the conditioned searcher vs smart after the continuation fix."""
import multiprocessing as mp

import meta

if __name__ == "__main__":
    with mp.Pool(12) as pool:
        m, ci, n = meta.match("adv_t10_p100", "smart", 1500, 12, pool,
                              seed=45100, rules="full")
        print(f"adv_t10_p100 (fixed continuation) vs smart [full]: "
              f"{m:+.2f} +/- {ci:.2f} ({n})", flush=True)
