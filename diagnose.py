"""
Is NFSP learning at all, and which half of it is lagging?

Evaluation in the main run uses the *average policy*, which imitates the DQN's
past best responses and therefore always trails it. A flat average-policy curve
is ambiguous: either the DQN is improving and the average policy just hasn't
caught up (normal, wait), or nothing is learning (bug, stop).

This trains a short NFSP run with the same hyper-parameters and evaluates BOTH
networks, plus tracks the losses the main run's stdout filter discards.

    python diagnose.py --num_episodes 6000
"""

import argparse
import contextlib
import io
import sys
import time

import numpy as np
import torch
from rlcard.agents import NFSPAgent, RandomAgent
from rlcard.utils import get_device, set_seed, reorganize

from train_gin_rummy import make_env, evaluate, _QuietStdout


class _Args:
    seed = 99
    reward = "shaped"
    num_episodes = 6000
    load_model = None
    train_every = 4
    q_train_every = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_episodes", type=int, default=6000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--num_eval_games", type=int, default=200)
    ap.add_argument("--reward", default="shaped")
    ap.add_argument("--train_every", type=int, default=4)
    args = ap.parse_args()

    _Args.reward = args.reward
    _Args.train_every = args.train_every
    _Args.q_train_every = args.train_every

    device = get_device()
    set_seed(_Args.seed)
    env = make_env(_Args)
    eval_env = make_env(_Args, seed=_Args.seed + 1)

    agents = [
        NFSPAgent(
            num_actions=env.num_actions, state_shape=env.state_shape[0],
            hidden_layers_sizes=[512, 512], q_mlp_layers=[512, 512],
            reservoir_buffer_capacity=int(2e5), q_replay_memory_size=int(2e5),
            q_epsilon_decay_steps=int(args.num_episodes * 36),
            rl_learning_rate=1e-4, sl_learning_rate=1e-4,
            train_every=args.train_every, q_train_every=args.train_every,
            device=device,
        )
        for _ in range(env.num_players)
    ]
    env.set_agents(agents)
    eval_env.set_agents([agents[0], RandomAgent(num_actions=env.num_actions)])

    print(f"reward={args.reward} train_every={args.train_every} "
          f"episodes={args.num_episodes}", flush=True)
    print(f"{'ep':>6} {'avg_policy':>11} {'best_resp':>11} {'dead%':>7} "
          f"{'knock%':>7} {'reservoir':>10} {'eps/s':>6}", flush=True)

    real = sys.stdout
    start = time.time()
    with contextlib.redirect_stdout(_QuietStdout(real)):
        for episode in range(args.num_episodes + 1):
            if episode % args.eval_every == 0:
                # Same agent, two different heads.
                agents[0].evaluate_with = "average_policy"
                avg, stats = evaluate(eval_env, args.num_eval_games, 7)
                agents[0].evaluate_with = "best_response"
                best, bstats = evaluate(eval_env, args.num_eval_games, 7)
                agents[0].evaluate_with = "average_policy"
                rate = (episode + 1) / max(time.time() - start, 1e-9)
                print(f"{episode:>6} {avg:>11.4f} {best:>11.4f} "
                      f"{stats['dead']:>6.1%} {bstats['knock']:>6.1%} "
                      f"{len(agents[0]._reservoir_buffer):>10} {rate:>6.2f}",
                      file=real, flush=True)
            if episode == args.num_episodes:
                break
            for agent in agents:
                agent.sample_episode_policy()
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for agent, traj in zip(agents, trajectories):
                for ts in traj:
                    agent.feed(ts)


if __name__ == "__main__":
    main()
