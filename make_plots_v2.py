"""
Build the v2 results figure: why the teacher swap worked, and what it bought.

Reads the logs the pipeline already wrote -- nothing is recomputed here, so the
figure cannot drift from the numbers in RESULTS.md.

    python make_plots_v2.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
OUT = os.path.join(R, "v2_final", "v2_results.png")


def bc_accuracy(path):
    """(epoch, accuracy) from a pretrain log."""
    if not os.path.exists(path):
        return [], []
    pat = re.compile(r"epoch\s+(\d+): loss [\d.]+ acc ([\d.]+)%")
    epochs, accs = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                accs.append(float(m.group(2)))
    return epochs, accs


def selection_rows(path):
    """(label, points, ci) for every greedy candidate in a select_final log."""
    if not os.path.exists(path):
        return []
    pat = re.compile(r"^(\S+\.policy\.pt)\s+greedy\s+([-+][\d.]+) \+/- ([\d.]+)")
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                rows.append((m.group(1).replace(".policy.pt", ""),
                             float(m.group(2)), float(m.group(3))))
    return rows


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # -- left: the teacher swap, seen through cloning accuracy ---------------
    for path, label, style in [
            (os.path.join(R, "pretrain.log"),
             "cloning the stock rule agent", "--"),
            (os.path.join(R, "v2_bc.log"),
             "cloning the deterministic teacher", "-")]:
        epochs, accs = bc_accuracy(path)
        if epochs:
            ax1.plot(epochs, accs, style, marker="o", ms=3, label=label)
    ax1.axhline(76.5, color="grey", lw=0.8, ls=":")
    ax1.annotate("76.5%: what a perfect imitator scores when half\n"
                 "the teacher's decisions are coin flips",
                 xy=(24, 76.5), xytext=(14.5, 43), fontsize=8, color="dimgrey",
                 arrowprops=dict(arrowstyle="->", color="dimgrey", lw=0.8,
                                 connectionstyle="arc3,rad=0.25"))
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("action accuracy (%)")
    ax1.set_title("A deterministic teacher is learnable")
    ax1.set_ylim(30, 100)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(alpha=0.3)

    # -- right: strength against the bar -------------------------------------
    rows = selection_rows(os.path.join(R, "v2_select.log"))
    order = {"v2_bc": -1}
    rows.sort(key=lambda r: order.get(r[0], int(re.sub(r"\D", "", r[0]) or 0)))
    labels = ["previous bot"] + ["BC clone" if n == "v2_bc" else n for n, _, _ in rows]
    # The shipped v1 bot, measured over 1500 hands in the handoff.
    values = [-2.25] + [v for _, v, _ in rows]
    errors = [1.14] + [c for _, _, c in rows]
    colours = ["#b03a2e"] + ["#5d6d7e" if v != max(values) else "#1e8449"
                             for v in values[1:]]

    ax2.bar(range(len(values)), values, yerr=errors, capsize=4, color=colours)
    ax2.axhline(0, color="black", lw=1)
    ax2.axhline(3.84, color="#1e8449", lw=0.8, ls=":")
    ax2.annotate("the teacher itself (+3.84)", xy=(0.2, 3.84), fontsize=8,
                 color="#1e8449", va="bottom")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("points/hand vs the smart-draw rule agent")
    ax2.set_title("Above zero beats the bar (2000 hands, 95% CI)")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
