import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Sanity checks: scorer zero-sum, outcome distribution, and throughput."""
import time, sys, collections
import numpy as np, torch, rlcard
from rlcard.agents import NFSPAgent, RandomAgent
from rlcard.utils import get_device, set_seed, reorganize
from rlcard.games.gin_rummy.utils.action_event import KnockAction, GinAction

sys.argv = ["x"]
import train_gin_rummy as T

class A:  # arg stub
    seed = 42; reward = "points"; num_episodes = 10000; load_model = None

set_seed(42)
env = T.make_env(A)
env.set_agents([RandomAgent(num_actions=env.num_actions) for _ in range(2)])

outcomes = collections.Counter()
payoff_sums = []
N = 300
t0 = time.time()
for _ in range(N):
    _, payoffs = env.run(is_training=False)
    payoff_sums.append(payoffs[0] + payoffs[1])
    a = env.game.round.going_out_action
    if isinstance(a, GinAction): outcomes["gin"] += 1
    elif isinstance(a, KnockAction): outcomes["knock"] += 1
    else: outcomes["dead/none"] += 1
rand_t = time.time() - t0
print(f"[random self-play] {N} games in {rand_t:.1f}s ({N/rand_t:.1f} games/s)")
print("  outcomes:", dict(outcomes))
print(f"  zero-sum check: max |p0+p1| = {max(abs(x) for x in payoff_sums):.6f}  (must be 0)")

# throughput of actual NFSP training loop
device = get_device()
agents = T.build_agents(env, device, A)
env.set_agents(agents)
M = 60
t0 = time.time()
steps = 0
for _ in range(M):
    for ag in agents: ag.sample_episode_policy()
    traj, payoffs = env.run(is_training=True)
    traj = reorganize(traj, payoffs)
    for ag, tr in zip(agents, traj):
        for ts in tr:
            ag.feed(ts); steps += 1
dt = time.time() - t0
print(f"[NFSP train] {M} episodes in {dt:.1f}s -> {M/dt:.2f} eps/s, {steps/M:.1f} feeds/episode")
print(f"  => 50k episodes ~ {50000/(M/dt)/3600:.2f} h ; 100k ~ {100000/(M/dt)/3600:.2f} h")
