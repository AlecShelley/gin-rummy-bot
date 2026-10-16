"""
Build the results figure: what failed, why, and what worked.

    python make_plots.py
"""

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
OUT = os.path.join(R, "final", "learning_curves.png")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def series(rows, x, y, scale=1.0):
    xs, ys = [], []
    for r in rows:
        if r.get(y) not in ("", None):
            xs.append(float(r[x]))
            ys.append(float(r[y]) * scale)
    return xs, ys


def parse_dagger(path):
    """Pull (round, vs_rule, vs_random) out of a dagger log."""
    if not os.path.exists(path):
        return [], [], []
    rounds, rule, rand = [], [], []
    pat = re.compile(r"round (\d+)[^:]*: vs_rule\s+([-+0-9.]+) .*?vs_random\s+([-+0-9.]+)")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                rounds.append(int(m.group(1)))
                rule.append(float(m.group(2)))
                rand.append(float(m.group(3)))
    return rounds, rule, rand


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax = axes.ravel()

# --- A: RL from scratch never learns -----------------------------------------
runs = [("dqn_base", "DQN lr=1e-4"), ("dqn_lr3", "DQN lr=1e-3"),
        ("dqn_g1_fast", "DQN gamma=1, fast target"), ("dqn_te1", "DQN train_every=1"),
        ("dqn_shape", "DQN + shaping"), ("nfsp_shape", "NFSP self-play + shaping")]
for tag, label in runs:
    rows = read_csv(os.path.join(R, f"exp_{tag}.csv"))
    if not rows:
        continue
    xs, ys = series(rows, "episode", "reward")
    style = "-" if "shap" in tag else "--"
    lw = 2.2 if "shap" in tag else 1.2
    ax[0].plot(xs, ys, style, lw=lw, label=label)
ax[0].axhline(-0.489, color="grey", ls=":", lw=1.5)
ax[0].annotate("untrained network", xy=(0.02, -0.489), xycoords=("axes fraction", "data"),
               fontsize=8, color="grey", va="bottom")
ax[0].set_title("A. From scratch: terminal-only reward fails outright;\n"
                "shaping partly rescues DQN (purple) but not NFSP (brown)",
                fontsize=10)
ax[0].set_xlabel("episode")
ax[0].set_ylabel("shaped reward vs random")
ax[0].legend(fontsize=7)
ax[0].grid(alpha=0.3)

# --- B: NFSP self-play destroys a good policy --------------------------------
rows = read_csv(os.path.join(R, "selfplay", "outcomes.csv"))
if rows:
    xs, ys = series(rows, "episode", "reward", 100.0)
    ax[1].plot(xs, ys, "o-", color="crimson", lw=2, label="vs random (points/hand)")
    xs2, ys2 = series(rows, "dead", "dead")  # placeholder guard
    xd, yd = series(rows, "episode", "dead", 100.0)
    ax[1].plot(xd, yd, "s--", color="darkorange", lw=1.6, label="dead hands (%)")
    xk, yk = series(rows, "episode", "knock", 100.0)
    ax[1].plot(xk, yk, "^--", color="steelblue", lw=1.6, label="knock rate (%)")
ax[1].set_title("B. NFSP self-play from the warm start makes it WORSE\n"
                "(its DQN knocks 2% of hands; the average policy imitates that)",
                fontsize=10)
ax[1].set_xlabel("episode")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

# --- C: DAgger works ---------------------------------------------------------
for log, label, colour in [("dagger.log", "DAgger, self-play roll-in only", "grey"),
                           ("dagger2.log", "DAgger, diversified roll-in", "seagreen")]:
    rounds, rule, rand = parse_dagger(os.path.join(R, log))
    if not rounds:
        continue
    ax[2].plot(rounds, rule, "o-", color=colour, lw=2, label=f"{label}: vs rule")
    ax[2].plot(rounds, rand, "s--", color=colour, lw=1.4, alpha=0.65,
               label=f"{label}: vs random")
ax[2].axhline(0, color="black", lw=0.8)
ax[2].set_title("C. DAgger: diversified roll-in improves BOTH opponents\n"
                "(self-play-only roll-in trades vs-random away)", fontsize=10)
ax[2].annotate("grey run evaluated by sampling, green by argmax",
               xy=(0.02, 0.03), xycoords="axes fraction", fontsize=7, color="dimgrey")
ax[2].set_xlabel("DAgger round")
ax[2].set_ylabel("points / hand")
ax[2].legend(fontsize=7)
ax[2].grid(alpha=0.3)

# --- D: final standing -------------------------------------------------------
labels = ["untrained\nnetwork", "clone\n(pretrained)", "network only\n(DAgger r5, greedy)",
          "final bot\n(+ smart draw)"]
vs_rule = [-48.74, -9.02, -3.15, +8.37]
colours = ["#b0b0b0", "#7fa8d4", "#5b9bd5", "#2e7d32"]
bars = ax[3].bar(labels, vs_rule, color=colours)
for b, v in zip(bars, vs_rule):
    ax[3].annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v),
                   ha="center", va="top" if v < 0 else "bottom", fontsize=10)
ax[3].axhline(0, color="black", lw=1)
ax[3].annotate("parity with the rule agent", xy=(0.03, 0), xycoords=("axes fraction", "data"),
               fontsize=8, color="black", va="bottom")
ax[3].set_title("D. Points/hand vs the novice rule agent\n"
                "(1500 hands, seats swapped; higher is better)", fontsize=10)
ax[3].set_ylabel("points / hand")
ax[3].grid(alpha=0.3, axis="y")

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
