"""
Improve the cloned policy by self-play policy gradient (REINFORCE + baseline).

Why not NFSP
------------
NFSP's average policy is trained to imitate its inner DQN's best responses.
Measured on the warm-started agent, that DQN knocks in 2.0% of hands while the
cloned average policy knocks in 74.7% -- so self-play steadily overwrote a good
policy with a bad one (+0.443 -> +0.270 vs random over 10k episodes). The DQN
never learned to knock even from 400k expert transitions: going out is ~2% of
transitions spread across 52 distinct knock action ids, which action-value
regression handles badly and cross-entropy handles fine.

Policy gradient sidesteps that entirely -- it optimises the policy directly,
never regressing 110 action values -- and starting from the clone means we are
improving a competent policy rather than exploring from noise. This is the
SL -> RL recipe AlphaGo used, for the same reason.

Safeguards, because RL has already destroyed a good policy once here:
  * the opponent is a frozen snapshot, refreshed only when the learner has
    demonstrably improved, so training targets a stable objective;
  * every evaluation benchmarks against the held-out rule agent and the best
    checkpoint so far is kept, so the deliverable can only improve;
  * a KL leash to the initial clone bounds how far a single run can drift.

    python policy_gradient.py --init results/pretrained.pth --episodes 40000
"""

import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from rlcard.agents import RandomAgent
from rlcard.agents.nfsp_agent import AveragePolicyNetwork
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
from rlcard.utils import get_device, set_seed

from shaping import state_potential
from train_gin_rummy import make_env, evaluate, classify_outcome


NEG = -1e9        # stand-in for -inf that stays finite under subtraction


class _Args:
    seed = 21
    reward = "points"


def load_policy_net(path, device):
    """Accept either a policy-only file or a full pickled NFSP agent."""
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        net = AveragePolicyNetwork(num_actions=blob["num_actions"],
                                   state_shape=blob["state_shape"],
                                   mlp_layers=blob["mlp_layers"])
        net.load_state_dict(blob["state_dict"])
        meta = (blob["num_actions"], blob["state_shape"], blob["mlp_layers"])
    else:
        net = copy.deepcopy(blob.policy_network)
        meta = (blob._num_actions, blob._state_shape, blob._layer_sizes)
    return net.to(device).eval(), meta


class NetAgent:
    """Plays a policy network; used for both the learner and the frozen foe."""
    use_raw = False

    def __init__(self, net, device, greedy=False):
        self.net, self.device, self.greedy = net, device, greedy

    def probs(self, state):
        obs = torch.from_numpy(np.expand_dims(state["obs"], 0)).float().to(self.device)
        with torch.no_grad():
            logits = self.net.mlp(obs)[0]
        legal = list(state["legal_actions"].keys())
        mask = torch.full_like(logits, float("-inf"))
        mask[legal] = logits[legal]   # sampling only: -inf is fine here
        return torch.softmax(mask, dim=-1).cpu().numpy()

    def step(self, state):
        p = self.probs(state)
        if self.greedy:
            return int(np.argmax(p))
        p = p / p.sum()
        return int(np.random.choice(len(p), p=p))

    def eval_step(self, state):
        return self.step(state), {}


def rollout(env, learner_seat, learner, gamma):
    """Play one hand; return the learner's (obs, action, legal, shaped reward)."""
    obs_l, act_l, legal_l, rew_l = [], [], [], []
    state, pid = env.reset()
    prev_phi = None
    while not env.is_over():
        agent = env.agents[pid]
        legal = list(state["legal_actions"].keys())
        action = agent.step(state)
        if pid == learner_seat and len(legal) > 1:
            phi = state_potential(state)
            if prev_phi is not None:
                # shaping for the gap since the learner's previous decision
                rew_l[-1] += gamma * phi - prev_phi
            obs_l.append(np.asarray(state["obs"], dtype=np.uint8))
            act_l.append(action)
            legal_l.append(legal)
            rew_l.append(0.0)
            prev_phi = phi
        state, pid = env.step(action)
    payoffs = env.get_payoffs()
    if rew_l:
        # terminal: true points payoff, and Phi(terminal) is 0 by definition
        rew_l[-1] += float(payoffs[learner_seat]) - prev_phi
    return obs_l, act_l, legal_l, rew_l, payoffs[learner_seat]


def discounted(rewards, gamma):
    out, run = [0.0] * len(rewards), 0.0
    for i in range(len(rewards) - 1, -1, -1):
        run = rewards[i] + gamma * run
        out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser("Self-play policy gradient from a cloned policy")
    ap.add_argument("--init", default="results/pretrained.pth")
    ap.add_argument("--episodes", type=int, default=40000)
    ap.add_argument("--batch_episodes", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--kl_coef", type=float, default=0.5,
                    help="leash to the initial clone; 0 disables")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--num_eval_games", type=int, default=400)
    ap.add_argument("--refresh_opponent", action="store_true",
                    help="promote the learner to opponent when it improves")
    ap.add_argument("--log_dir", default="results/pg")
    args = ap.parse_args()

    device = get_device()
    set_seed(_Args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    net, meta = load_policy_net(args.init, device)
    num_actions = meta[0]
    anchor, _ = load_policy_net(args.init, device)     # frozen reference for KL
    for p in anchor.parameters():
        p.requires_grad_(False)
    foe_net, _ = load_policy_net(args.init, device)
    for p in foe_net.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = make_env(_Args)
    learner = NetAgent(net, device)
    foe = NetAgent(foe_net, device)
    env.set_agents([learner, foe])

    eval_random = make_env(_Args, seed=_Args.seed + 1)
    eval_random.set_agents([learner, RandomAgent(num_actions=num_actions)])
    eval_rule = make_env(_Args, seed=_Args.seed + 2)
    eval_rule.set_agents([learner, GinRummyNoviceRuleAgent()])

    csv_path = os.path.join(args.log_dir, "pg_progress.csv")
    fh = open(csv_path, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=["episode", "vs_random", "vs_rule", "knock",
                                       "gin", "dead", "win_rate", "best_vs_rule",
                                       "elapsed_s"])
    w.writeheader()

    def benchmark(tag, episode, best):
        net.eval()
        r_rand, st = evaluate(eval_random, args.num_eval_games, 7)
        r_rule, st2 = evaluate(eval_rule, args.num_eval_games, 11)
        improved = r_rule > best
        if improved:
            best = r_rule
            torch.save({"num_actions": meta[0], "state_shape": meta[1],
                        "mlp_layers": meta[2], "state_dict": net.state_dict()},
                       os.path.join(args.log_dir, "best.policy.pt"))
        w.writerow({"episode": episode, "vs_random": round(r_rand, 5),
                    "vs_rule": round(r_rule, 5), "knock": round(st2["knock"], 4),
                    "gin": round(st2["gin"], 4), "dead": round(st2["dead"], 4),
                    "win_rate": round(st2["win_rate"], 4),
                    "best_vs_rule": round(best, 5),
                    "elapsed_s": round(time.time() - start, 1)})
        fh.flush()
        print(f"{episode:>7} vs_random {r_rand * 100:+7.2f} | vs_rule "
              f"{r_rule * 100:+7.2f} {'*BEST*' if improved else '      '} | "
              f"knock {st2['knock']:.1%} gin {st2['gin']:.1%} dead {st2['dead']:.1%}"
              f" | {tag}", flush=True)
        return best, improved

    print(f"init={args.init} lr={args.lr} entropy={args.entropy} "
          f"kl={args.kl_coef} batch={args.batch_episodes}", flush=True)
    start = time.time()
    best, _ = benchmark("init", 0, -9e9)

    buf_obs, buf_act, buf_legal, buf_ret = [], [], [], []
    episode = 0
    while episode < args.episodes:
        for _ in range(args.batch_episodes):
            o, a, lg, r, _ = rollout(env, 0, learner, args.gamma)
            if o:
                buf_obs += o
                buf_act += a
                buf_legal += lg
                buf_ret += discounted(r, args.gamma)
            episode += 1

        if not buf_obs:
            continue
        obs_t = torch.from_numpy(np.array(buf_obs)).float().to(device)
        act_t = torch.tensor(buf_act, dtype=torch.long, device=device)
        ret_t = torch.tensor(buf_ret, dtype=torch.float32, device=device)
        adv = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        # Finite, not -inf: the KL term below differences two masked log-prob
        # tensors, and (-inf) - (-inf) is NaN, which poisons the whole loss.
        mask = torch.full((len(buf_obs), num_actions), NEG, device=device)
        for i, lg in enumerate(buf_legal):
            mask[i, lg] = 0.0

        net.train()
        logits = net.mlp(obs_t) + mask
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp.gather(1, act_t.unsqueeze(1)).squeeze(1)
        probs = logp.exp()
        entropy = -(probs * logp.clamp_min(-30)).sum(1).mean()

        loss = -(chosen * adv).mean() - args.entropy * entropy
        if args.kl_coef:
            with torch.no_grad():
                ref = F.log_softmax(anchor.mlp(obs_t) + mask, dim=-1)
            kl = (ref.exp() * (ref - logp).clamp(-30, 30)).sum(1).mean()
            loss = loss + args.kl_coef * kl

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss.item()}")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        net.eval()

        buf_obs, buf_act, buf_legal, buf_ret = [], [], [], []

        if episode % args.eval_every < args.batch_episodes:
            best, improved = benchmark(f"loss {loss.item():+.4f}", episode, best)
            if improved and args.refresh_opponent:
                foe_net.load_state_dict(net.state_dict())

    benchmark("final", episode, best)
    fh.close()
    print(f"\nbest vs rule: {best * 100:+.2f} pts/hand -> "
          f"{os.path.join(args.log_dir, 'best.policy.pt')}")


if __name__ == "__main__":
    main()
