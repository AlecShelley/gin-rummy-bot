import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Build the v3 results figure from the measurement JSONs -- nothing recomputed.

    python make_plots_v3.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META = "results/v3_meta"
OUT = os.path.join(META, "v3_results.png")


def main():
    battery = json.load(open(os.path.join(META, "final_battery.json"),
                             encoding="utf-8"))
    hypo = json.load(open(os.path.join(META, "hypothesis.json"),
                          encoding="utf-8"))
    exploits = {}
    for name in ("v2", "v3search"):
        rows = json.load(open(os.path.join(META, f"exploit_{name}.json"),
                              encoding="utf-8"))
        exploits[name] = max(rows, key=lambda r: r["gain"])

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [5, 3, 3]})

    # -- left: the field ------------------------------------------------------
    order = ["random", "rule", "v1", "v3", "knock6_p100", "v2", "teacher",
             "smart", "knock10_p100"]
    labels = {"random": "random", "rule": "rlcard rule agent",
              "v1": "v1 bot", "v3": "v3 clone (failed distill)",
              "knock6_p100": "knock-at-6", "v2": "v2 bot",
              "teacher": "patient teacher", "smart": "smart-draw agent",
              "knock10_p100": "knock-at-10"}
    names = [n for n in order if n in battery]
    vals = [battery[n][0] for n in names]
    errs = [battery[n][1] for n in names]
    colors = ["#1e8449" if v - e > 0 else "#5d6d7e" for v, e in zip(vals, errs)]
    y = range(len(names))
    ax1.barh(y, vals, xerr=errs, capsize=3, color=colors)
    ax1.axvline(0, color="black", lw=1)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels([labels[n] for n in names], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("v3search's points/hand (2000 hands, 95% CI)")
    ax1.set_title("The search bot vs the field")
    ax1.grid(alpha=0.3, axis="x")

    # -- middle: exploitability ----------------------------------------------
    names2 = ["v2", "v3search"]
    vals2 = [exploits[n]["gain"] for n in names2]
    errs2 = [exploits[n]["ci"] for n in names2]
    ax2.bar(range(2), vals2, yerr=errs2, capsize=4,
            color=["#b03a2e", "#1e8449"])
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["v2 bot", "v3search"])
    ax2.set_ylabel("best adversary's gain (pts/hand)")
    ax2.set_title("Exploitability: best of 21\ndedicated adversaries")
    ax2.grid(alpha=0.3, axis="y")

    # -- right: the range-balancing hypothesis --------------------------------
    ps = ["0.0", "0.5", "1.0"]
    vals3 = [hypo[p][0] for p in ps]
    errs3 = [hypo[p][1] for p in ps]
    ax3.bar(range(3), vals3, yerr=errs3, capsize=4, color="#5d6d7e")
    ax3.axhline(0, color="black", lw=1)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(["never\nknock", "50%\nmixed", "always\nknock"])
    ax3.set_ylabel("hand-reader's edge (pts/hand)")
    ax3.set_title("Range-balancing hypothesis:\nmixing does not protect")
    ax3.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
