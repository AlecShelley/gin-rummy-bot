"""
Train the v3 clone on search-labeled data, DAgger-style.

Round 0 fits the expert's BC shards; each later round rolls in with the
current clone (exit_gen --rollin), labels the visited states with the expert,
and refits on the UNION of everything so far -- dropping old data measured a
catastrophic forget in v2 (-5.43 -> -8.41), so the union is not optional.

Every round is benchmarked seat-swapped against the real bar ('smart') and
the teacher ('strong'); the best round by the smart score is kept. Final
selection at higher power belongs to select_final.py, not here.

    python exit_train.py --bc results/v3_data/bc --rounds 3 --out results/v3
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from rlcard.agents.nfsp_agent import AveragePolicyNetwork


def load_shards(patterns):
    obs, labels = [], []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with np.load(path) as data:
                if len(data["labels"]):
                    obs.append(data["obs"])
                    labels.append(data["labels"])
    x = np.concatenate(obs).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    return x.reshape(len(x), -1), y


def fit(net, x, y, device, epochs=12, batch=512, lr=1e-3, tag=""):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.NLLLoss()
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    n = len(x_t)
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        total, correct, seen = 0.0, 0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue                 # BatchNorm chokes on batches of one
            opt.zero_grad()
            log_probs = net(x_t[idx])
            loss = lossf(log_probs, y_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            correct += (log_probs.argmax(1) == y_t[idx]).sum().item()
            seen += len(idx)
        net.eval()
        if (epoch + 1) % 4 == 0 or epoch == epochs - 1:
            print(f"    [{tag}] epoch {epoch + 1:>2}: loss {total / seen:.4f} "
                  f"acc {correct / seen:.1%}", flush=True)
    net.eval()
    return correct / seen


def save_policy(net, path):
    torch.save({"num_actions": 110, "state_shape": [10, 52],
                "mlp_layers": [1024, 1024, 110],
                "state_dict": net.state_dict()}, path)


def _bench_chunk(job):
    """Worker: the clone (greedy) vs a named opponent, seat-swapped half."""
    path, opponent, hands, seat, seed, floor = job
    import torch as _torch
    _torch.set_num_threads(1)
    from train_gin_rummy import make_env
    from gin_rummy_bot import load_opponent
    from v3_bot import V3Bot

    class _Args:
        reward = "points"
        obs = "base"

    _Args.seed = seed
    env = make_env(_Args)
    bot = V3Bot(path, env=env, device=_torch.device("cpu"), greedy=True,
                teacher_floor=floor)
    opp, _ = load_opponent(opponent, env.num_actions, greedy=True)
    env.set_agents([bot, opp] if seat == 0 else [opp, bot])
    env.seed(seed)
    return [float(env.run(is_training=False)[1][seat] * 100.0)
            for _ in range(hands)]


def bench(path, opponent, hands=800, workers=12, seed=41, floor=None):
    import multiprocessing as mp
    per = max(1, hands // (2 * workers))
    jobs = [(path, opponent, per, seat, seed + 97 * w + seat, floor)
            for w in range(workers) for seat in (0, 1)]
    with mp.Pool(workers) as pool:
        points = np.concatenate(pool.map(_bench_chunk, jobs))
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    return mean, ci, len(points)


def main():
    ap = argparse.ArgumentParser("ExIt training loop")
    ap.add_argument("--bc", default="results/v3_data/bc",
                    help="shard prefix from exit_gen (BC round)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="DAgger rounds after the BC fit")
    ap.add_argument("--dagger_hands", type=int, default=9000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bc_epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bench_hands", type=int, default=800)
    ap.add_argument("--out", default="results/v3")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    patterns = [f"{args.bc}_*.npz"]
    x, y = load_shards(patterns)
    print(f"BC dataset: {len(x)} samples", flush=True)
    # mlp_layers includes the output size (rlcard convention: hidden + [out]).
    net = AveragePolicyNetwork(num_actions=110, state_shape=[10, 52],
                               mlp_layers=[1024, 1024, 110]).to(device)
    acc = fit(net, x, y, device, epochs=args.bc_epochs, tag="bc")
    path = os.path.join(args.out, "round0.policy.pt")
    save_policy(net, path)

    best_path, best_score = path, None
    for rnd in range(args.rounds + 1):
        m_smart, ci_s, n_s = bench(path, "smart", args.bench_hands,
                                   args.workers, seed=41 + rnd)
        m_strong, ci_t, _ = bench(path, "strong", args.bench_hands,
                                  args.workers, seed=61 + rnd)
        print(f"  round {rnd}: acc {acc:.1%} | vs smart {m_smart:+.2f} "
              f"+/- {ci_s:.2f} | vs strong {m_strong:+.2f} +/- {ci_t:.2f} "
              f"({n_s} hands each)", flush=True)
        if best_score is None or m_smart > best_score:
            best_path, best_score = path, m_smart
            save_policy(net, os.path.join(args.out, "best.policy.pt"))
            print("    -> new best", flush=True)
        if rnd == args.rounds:
            break

        # DAgger round: clone rolls in, expert labels, union refit.
        from exit_gen import generate
        prefix = f"{args.out}/dagger{rnd + 1}"
        t0 = time.time()
        generate(args.dagger_hands, args.workers, prefix,
                 rollin_path=path, seed=args.seed + 1000 * (rnd + 1))
        patterns.append(f"{prefix}_*.npz")
        x, y = load_shards(patterns)
        print(f"  union dataset now {len(x)} samples "
              f"(+{time.time() - t0:.0f}s roll-in)", flush=True)
        acc = fit(net, x, y, device, epochs=args.epochs, tag=f"r{rnd + 1}")
        path = os.path.join(args.out, f"round{rnd + 1}.policy.pt")
        save_policy(net, path)

    print(f"\nbest vs smart: {best_score:+.2f} -> {best_path} "
          f"(copied to {args.out}/best.policy.pt)")


if __name__ == "__main__":
    main()
