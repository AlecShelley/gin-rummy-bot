import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
DAgger: close the gap between the clone and its teacher.

Plain behaviour cloning trains on states the *teacher* visits, then gets played
on states the *clone* visits. Those distributions differ, and every small error
compounds -- which is why the clone sits at -5.17 points/hand against the very
teacher it copied. DAgger (Ross, Gordon & Bagnell 2011) fixes this directly:
roll in with the current clone, label those states with the expert, add them to
the dataset, retrain. Each round the training distribution moves closer to the
one the clone actually encounters.

Unlike the RL attempts here this cannot silently destroy the policy: every round
is benchmarked against a held-out opponent and only improvements are kept.

`--eval_opponent` decides what "improvement" means, and it matters: scoring
against rlcard's stock rule agent measures the wrong thing, because that agent
draws by coin flip and every candidate beats it comfortably. Score against the
opponent that is actually the bar.

    python dagger.py --teacher strong --eval_opponent smart
        --rounds 5 --diverse_rollin
        --init results/v2_bc.policy.pt --out results/v2_dagger
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from rlcard.agents import RandomAgent
from rlcard.agents.nfsp_agent import AveragePolicyNetwork
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
from rlcard.utils import get_device, set_seed
from rlcard.utils.utils import remove_illegal

from teacher import make_teacher
from legacy.train_gin_rummy import make_env, evaluate


class _Args:
    seed = 33
    reward = "points"


class CloneRollin:
    """Plays the current clone, but records what the expert would have done."""

    use_raw = False

    def __init__(self, net, device, expert, teacher_mix=0.0, epsilon=0.0):
        self.net, self.device, self.expert = net, device, expert
        self.teacher_mix = teacher_mix
        self.epsilon = epsilon
        self.obs, self.labels = [], []

    def step(self, state):
        legal = list(state["legal_actions"].keys())
        if len(legal) == 1:
            return legal[0]
        expert_action = int(self.expert.step(state))
        self.obs.append(np.asarray(state["obs"], dtype=np.uint8))
        self.labels.append(expert_action)
        if np.random.rand() < self.teacher_mix:
            return expert_action
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(legal))
        obs = torch.from_numpy(np.expand_dims(state["obs"], 0)).float().to(self.device)
        with torch.no_grad():
            probs = np.exp(self.net(obs).cpu().numpy())[0]
        probs = remove_illegal(probs, legal)
        return int(np.random.choice(len(probs), p=probs))

    def eval_step(self, state):
        return self.step(state), {}


class Clone:
    use_raw = False

    def __init__(self, net, device, greedy=False):
        self.net, self.device, self.greedy = net, device, greedy

    def _p(self, state):
        obs = torch.from_numpy(np.expand_dims(state["obs"], 0)).float().to(self.device)
        with torch.no_grad():
            p = np.exp(self.net(obs).cpu().numpy())[0]
        return remove_illegal(p, list(state["legal_actions"].keys()))

    def step(self, state):
        p = self._p(state)
        return int(np.argmax(p)) if self.greedy else int(
            np.random.choice(len(p), p=p))

    def eval_step(self, state):
        return self.step(state), {}


def fit(net, X, y, device, epochs, batch, lr, tag):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.NLLLoss()
    n = len(X)
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        tot, correct, seen = 0.0, 0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logp = net(X[idx])
            loss = lossf(logp, y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += (logp.argmax(1) == y[idx]).sum().item()
            seen += len(idx)
        net.eval()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"    [{tag}] epoch {epoch + 1:>2}: loss {tot / seen:.4f} "
                  f"acc {correct / seen:.1%}", flush=True)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser("DAgger refinement of the cloned policy")
    ap.add_argument("--init", default="results/pretrained.pth")
    ap.add_argument("--teacher", choices=["rule", "smart", "strong"],
                    default="strong", help="the expert that labels every state")
    ap.add_argument("--eval_opponent", choices=["rule", "smart", "strong"],
                    default="smart",
                    help="who each round is scored against; 'smart' is the bar")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed_episodes", type=int, default=6000,
                    help="pure-teacher episodes seeding D_expert (DAgger keeps "
                         "the union of expert and clone-visited data; without "
                         "them retraining catastrophically forgets)")
    ap.add_argument("--episodes_per_round", type=int, default=4000)
    ap.add_argument("--diverse_rollin", action="store_true",
                    help="split roll-in across self / random / teacher opponents "
                         "so the clone also sees the long hands that only occur "
                         "against weak play")
    ap.add_argument("--rollin_epsilon", type=float, default=0.05,
                    help="random deviation during roll-in, for state coverage")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num_eval_games", type=int, default=400)
    ap.add_argument("--max_samples", type=int, default=1_500_000)
    ap.add_argument("--out", default="results/dagger")
    args = ap.parse_args()

    device = get_device()
    set_seed(_Args.seed)
    os.makedirs(args.out, exist_ok=True)

    blob = torch.load(args.init, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        meta = (blob["num_actions"], blob["state_shape"], blob["mlp_layers"])
        net = AveragePolicyNetwork(*[meta[0], meta[1], meta[2]])
        net.load_state_dict(blob["state_dict"])
    else:
        meta = (blob._num_actions, blob._state_shape, blob._layer_sizes)
        net = blob.policy_network
    net = net.to(device).eval()

    env = make_env(_Args)
    eval_bar = make_env(_Args, seed=_Args.seed + 1)
    eval_rand = make_env(_Args, seed=_Args.seed + 2)
    clone = Clone(net, device, greedy=True)   # match deployment
    eval_bar.set_agents([clone, make_teacher(args.eval_opponent)])
    # vs random is not a strength measure -- everything beats random. It is the
    # tripwire for the one failure mode that matters here: a clone that has
    # stopped going out at all dead-hands the long games only random produces.
    eval_rand.set_agents([clone, RandomAgent(num_actions=meta[0])])

    def bench(tag):
        r_bar, s_bar = evaluate(eval_bar, args.num_eval_games, 11)
        r_rand, s_rand = evaluate(eval_rand, args.num_eval_games, 7)
        print(f"  {tag}: vs_{args.eval_opponent} {r_bar * 100:+7.2f} pts/hand "
              f"(win {s_bar['win_rate']:.1%}, knock {s_bar['knock']:.1%}, "
              f"gin {s_bar['gin']:.1%}, dead {s_bar['dead']:.1%}) | "
              f"vs_random {r_rand * 100:+7.2f} (dead {s_rand['dead']:.1%})",
              flush=True)
        return r_bar

    best = bench("round 0 (initial clone)")
    torch.save({"num_actions": meta[0], "state_shape": meta[1],
                "mlp_layers": meta[2], "state_dict": net.state_dict()},
               os.path.join(args.out, "best.policy.pt"))

    all_obs, all_lab = [], []

    # D_expert: the clone's original training distribution. pretrain.py did not
    # keep its dataset, and without this the first retrain forgets it entirely
    # (measured: -5.43 -> -8.41 points/hand).
    t0 = time.time()
    seeders = [CloneRollin(net, device, make_teacher(args.teacher), 1.0)
               for _ in range(2)]
    env.set_agents(seeders)
    for _ in range(args.seed_episodes):
        env.run(is_training=False)
    all_obs.append(np.concatenate([np.array(c.obs, dtype=np.uint8) for c in seeders]))
    all_lab.append(np.concatenate([np.array(c.labels, dtype=np.int64) for c in seeders]))
    print(f"seeded D_expert with {len(all_obs[0])} teacher states "
          f"in {time.time() - t0:.0f}s", flush=True)

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        # Roll in with the clone itself; mix in a little teacher play early so
        # the first rounds still see well-played hands.
        mix = max(0.0, 0.3 - 0.1 * rnd)
        def mk():
            return CloneRollin(net, device, make_teacher(args.teacher), mix,
                               args.rollin_epsilon)
        collectors = []
        if args.diverse_rollin:
            # A policy trained only on self-play never sees a hand that runs to
            # stock exhaustion, so it mis-plays long games (measured: 75% dead
            # hands vs random). Roll in against weak and strong opponents too.
            # The teacher is patient, so it produces long hands the clone would
            # never see rolling in against itself or against early knockers.
            opponents = ("self", "random", "rule", "teacher")
            share = args.episodes_per_round // len(opponents)
            for opp_name in opponents:
                c = mk()
                collectors.append(c)
                if opp_name == "self":
                    other = mk()
                    collectors.append(other)
                elif opp_name == "random":
                    other = RandomAgent(num_actions=meta[0])
                elif opp_name == "rule":
                    other = GinRummyNoviceRuleAgent()
                else:
                    other = make_teacher(args.teacher)
                env.set_agents([c, other])
                for _ in range(share):
                    env.run(is_training=False)
        else:
            collectors = [mk(), mk()]
            env.set_agents(collectors)
            for _ in range(args.episodes_per_round):
                env.run(is_training=False)
        obs = np.concatenate([np.array(c.obs, dtype=np.uint8)
                              for c in collectors if c.obs])
        lab = np.concatenate([np.array(c.labels, dtype=np.int64)
                              for c in collectors if c.labels])
        all_obs.append(obs)
        all_lab.append(lab)
        X_np = np.concatenate(all_obs)
        y_np = np.concatenate(all_lab)
        if len(X_np) > args.max_samples:      # keep memory bounded
            keep = np.random.choice(len(X_np), args.max_samples, replace=False)
            X_np, y_np = X_np[keep], y_np[keep]
        print(f"round {rnd}: +{len(obs)} states (teacher_mix={mix:.1f}), "
              f"dataset {len(X_np)}, rollout {time.time() - t0:.0f}s", flush=True)

        X = torch.from_numpy(X_np).float().to(device)
        y = torch.from_numpy(y_np).to(device)
        fit(net, X, y, device, args.epochs, args.batch, args.lr, f"r{rnd}")
        del X, y
        torch.cuda.empty_cache() if device.type == "cuda" else None

        torch.save({"num_actions": meta[0], "state_shape": meta[1],
                    "mlp_layers": meta[2], "state_dict": net.state_dict()},
                   os.path.join(args.out, f"round{rnd}.policy.pt"))
        score = bench(f"round {rnd}")
        if score > best:
            best = score
            torch.save({"num_actions": meta[0], "state_shape": meta[1],
                        "mlp_layers": meta[2], "state_dict": net.state_dict()},
                       os.path.join(args.out, "best.policy.pt"))
            print(f"  -> new best, saved", flush=True)

    print(f"\nbest vs {args.eval_opponent}: {best * 100:+.2f} pts/hand -> "
          f"{os.path.join(args.out, 'best.policy.pt')}")


if __name__ == "__main__":
    main()
