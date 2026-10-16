import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Full-rules exploitability probe of the v5 candidate (gate b)."""
import meta

if __name__ == "__main__":
    meta.probe_exploitability("v5", hands=400, workers=12, seed=20260826,
                              rules="full")
