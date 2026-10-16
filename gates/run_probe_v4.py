import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Gate (b) for the v4 promotion: exploitability probe, full rules."""
import meta

if __name__ == "__main__":
    meta.probe_exploitability("v4", hands=400, workers=12,
                              seed=20260824, rules="full")
