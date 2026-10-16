"""
Representation probe: can the network learn gin rummy at all?

RL is failing here, but that could mean two very different things:
  (a) the 5x52 observation / 110-action encoding cannot express a good policy,
      or the network cannot fit it  -> no amount of RL tuning will help;
  (b) the encoding is fine and the problem is credit assignment from a
      terminal-only reward over ~60 steps -> fix the learning signal.

Behaviour cloning separates them. We record the novice rule agent's decisions
and fit the same AveragePolicyNetwork used by NFSP to predict them. If the
clone plays well, the representation is fine and (b) is the answer.

    python bc_probe.py --episodes 1500 --epochs 8
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from rlcard.agents import RandomAgent
from rlcard.agents.nfsp_agent import AveragePolicyNetwork
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
from rlcard.utils import get_device, set_seed
from rlcard.utils.utils import remove_illegal

from train_gin_rummy import make_env, evaluate


class _Args:
    seed = 5
    reward = "shaped"


class ClonePolicy:
    """Wraps the cloned network as an rlcard agent."""
    use_raw = False

    def __init__(self, net, device):
        self.net, self.device = net, device

    def _probs(self, state):
        obs = torch.from_numpy(np.expand_dims(state["obs"], 0)).float().to(self.device)
        with torch.no_grad():
            p = np.exp(self.net(obs).cpu().numpy())[0]
        return remove_illegal(p, list(state["legal_actions"].keys()))

    def step(self, state):
        return int(np.argmax(self._probs(state)))

    def eval_step(self, state):
        return self.step(state), {}


def collect(env, num_episodes):
    """Record (observation, chosen action) pairs from rule-agent play."""
    obs, acts = [], []
    teacher = GinRummyNoviceRuleAgent()
    for _ in range(num_episodes):
        state, pid = env.reset()
        while not env.is_over():
            if len(state["legal_actions"]) > 1:      # skip forced moves
                action = teacher.step(state)
                obs.append(state["obs"])
                acts.append(action)
            else:
                action = list(state["legal_actions"].keys())[0]
            state, pid = env.step(action)
    return np.array(obs, dtype=np.float32), np.array(acts, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_eval_games", type=int, default=300)
    args = ap.parse_args()

    device = get_device()
    set_seed(_Args.seed)
    env = make_env(_Args)
    env.set_agents([GinRummyNoviceRuleAgent(), GinRummyNoviceRuleAgent()])

    t0 = time.time()
    X, y = collect(env, args.episodes)
    print(f"collected {len(X)} decisions from {args.episodes} episodes "
          f"in {time.time() - t0:.0f}s", flush=True)
    print(f"distinct actions taught: {len(np.unique(y))}", flush=True)

    net = AveragePolicyNetwork(num_actions=env.num_actions,
                               state_shape=env.state_shape[0],
                               mlp_layers=[512, 512, env.num_actions]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.NLLLoss()

    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(Xt)
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        total, correct, seen = 0.0, 0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 2:                 # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            logp = net(Xt[idx])
            loss = lossf(logp, yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            correct += (logp.argmax(1) == yt[idx]).sum().item()
            seen += len(idx)
        net.eval()
        print(f"  epoch {epoch + 1}: loss {total / seen:.4f}  "
              f"train-acc {correct / seen:.1%}", flush=True)

    net.eval()
    clone = ClonePolicy(net, device)
    eval_env = make_env(_Args, seed=_Args.seed + 1)
    eval_env.set_agents([clone, RandomAgent(num_actions=env.num_actions)])
    score, st = evaluate(eval_env, args.num_eval_games, 7)
    print(f"\nCLONE vs random : reward {score:+.4f} | dead {st['dead']:.1%} "
          f"knock {st['knock']:.1%} gin {st['gin']:.1%} win {st['win_rate']:.1%}",
          flush=True)

    rule_env = make_env(_Args, seed=_Args.seed + 2)
    rule_env.set_agents([GinRummyNoviceRuleAgent(),
                         RandomAgent(num_actions=env.num_actions)])
    rscore, rst = evaluate(rule_env, args.num_eval_games, 7)
    print(f"TEACHER vs random: reward {rscore:+.4f} | dead {rst['dead']:.1%} "
          f"knock {rst['knock']:.1%} gin {rst['gin']:.1%} win {rst['win_rate']:.1%}",
          flush=True)

    torch.save({"num_actions": env.num_actions, "state_shape": env.state_shape[0],
                "mlp_layers": [512, 512, env.num_actions],
                "state_dict": net.state_dict()}, "results/bc_clone.policy.pt")
    print("\nsaved results/bc_clone.policy.pt")


if __name__ == "__main__":
    main()
