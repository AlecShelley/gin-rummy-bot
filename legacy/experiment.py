import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Short controlled runs to find a configuration that actually learns.

The 150k-episode Stage 1 was flat in BOTH the average policy and the DQN
best-response, so before spending hours we need to know which knob matters.
Each run here is a few thousand episodes and writes a compact CSV.

    python experiment.py --tag dqn_lr1e-3 --agent dqn --lr 1e-3 --episodes 4000
"""

import argparse
import contextlib
import csv
import os
import time

import torch
from rlcard.agents import DQNAgent, NFSPAgent, RandomAgent
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
from rlcard.utils import get_device, set_seed, reorganize

from legacy.train_gin_rummy import make_env, evaluate, _QuietStdout
from shaping import shape_trajectory


class _Args:
    seed = 7
    reward = "shaped"


def build(kind, env, device, args):
    if kind == "dqn":
        return DQNAgent(
            num_actions=env.num_actions, state_shape=env.state_shape[0],
            mlp_layers=[512, 512], learning_rate=args.lr,
            replay_memory_size=int(2e5), replay_memory_init_size=1000,
            update_target_estimator_every=args.target_every,
            train_every=args.train_every,
            epsilon_start=args.epsilon_start, epsilon_end=0.05,
            epsilon_decay_steps=args.episodes * 20,
            batch_size=64, discount_factor=args.gamma, device=device,
        )
    return NFSPAgent(
        num_actions=env.num_actions, state_shape=env.state_shape[0],
        hidden_layers_sizes=[512, 512], q_mlp_layers=[512, 512],
        reservoir_buffer_capacity=int(2e5), q_replay_memory_size=int(2e5),
        q_epsilon_decay_steps=args.episodes * 20,
        rl_learning_rate=args.lr, sl_learning_rate=args.sl_lr,
        anticipatory_param=args.anticipatory,
        train_every=args.train_every, q_train_every=args.train_every,
        q_update_target_estimator_every=args.target_every,
        q_discount_factor=args.gamma, device=device,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--agent", choices=["dqn", "nfsp"], default="dqn")
    ap.add_argument("--opponent", choices=["random", "rule", "self"], default="random")
    ap.add_argument("--reward", choices=["shaped", "points"], default="shaped")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sl_lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--anticipatory", type=float, default=0.1)
    ap.add_argument("--train_every", type=int, default=4)
    ap.add_argument("--target_every", type=int, default=1000)
    ap.add_argument("--epsilon_start", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--shaping", choices=["none", "deadwood"], default="none")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--num_eval_games", type=int, default=200)
    args = ap.parse_args()

    _Args.reward = args.reward
    device = get_device()
    set_seed(_Args.seed)
    env = make_env(_Args)
    eval_env = make_env(_Args, seed=_Args.seed + 1)

    learner = build(args.agent, env, device, args)
    if args.opponent == "self":
        partner = build(args.agent, env, device, args)
        train_agents = [learner, partner]
    else:
        partner = (RandomAgent(num_actions=env.num_actions)
                   if args.opponent == "random" else GinRummyNoviceRuleAgent())
        train_agents = [learner, partner]
    env.set_agents(train_agents)
    eval_env.set_agents([learner, RandomAgent(num_actions=env.num_actions)])

    out = os.path.join("results", f"exp_{args.tag}.csv")
    os.makedirs("results", exist_ok=True)
    fh = open(out, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=["episode", "reward", "dead", "knock",
                                       "gin", "win_rate", "eps_s"])
    w.writeheader()

    print(f"[{args.tag}] agent={args.agent} opp={args.opponent} reward={args.reward} "
          f"lr={args.lr} sl_lr={args.sl_lr} train_every={args.train_every} "
          f"gamma={args.gamma} anticipatory={args.anticipatory} shaping={args.shaping}", flush=True)
    print(f"{'ep':>6} {'reward':>9} {'dead%':>7} {'knock%':>7} {'gin%':>6} {'eps/s':>6}",
          flush=True)

    import sys
    real = sys.stdout
    start = time.time()
    with contextlib.redirect_stdout(_QuietStdout(real)):
        for episode in range(args.episodes + 1):
            if episode % args.eval_every == 0:
                score, st = evaluate(eval_env, args.num_eval_games, 7)
                rate = (episode + 1) / max(time.time() - start, 1e-9)
                w.writerow({"episode": episode, "reward": round(score, 5),
                            "dead": round(st["dead"], 4), "knock": round(st["knock"], 4),
                            "gin": round(st["gin"], 4),
                            "win_rate": round(st["win_rate"], 4),
                            "eps_s": round(rate, 2)})
                fh.flush()
                print(f"{episode:>6} {score:>9.4f} {st['dead']:>6.1%} "
                      f"{st['knock']:>6.1%} {st['gin']:>5.1%} {rate:>6.2f}",
                      file=real, flush=True)
            if episode == args.episodes:
                break
            for a in train_agents:
                if hasattr(a, "sample_episode_policy"):
                    a.sample_episode_policy()
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            if args.shaping == "deadwood":
                trajectories = [shape_trajectory(t, args.gamma) for t in trajectories]
            for a, traj in zip(train_agents, trajectories):
                if not hasattr(a, "feed"):
                    continue                     # rule/random opponents don't learn
                for ts in traj:
                    a.feed(ts)
    fh.close()
    print(f"[{args.tag}] wrote {out}", file=real, flush=True)


if __name__ == "__main__":
    main()
