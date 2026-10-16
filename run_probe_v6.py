"""Full-rules exploitability probe of the v6 candidate (promotion gate)."""
import meta

if __name__ == "__main__":
    meta.probe_exploitability("v6", hands=400, workers=12, seed=20260827,
                              rules="full")
