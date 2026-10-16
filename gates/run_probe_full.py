import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Re-probe exploitability of the CURRENT (leaf-fixed) v3search, full rules.

Closes the open measurement gap from the handoff: the existing
exploit_*_full.json files probed the pre-leaf-fix version. Fresh seed
(selection != reporting discipline).
"""
import meta

if __name__ == "__main__":
    meta.probe_exploitability("v3search", hands=400, workers=12,
                              seed=20260823, rules="full")
