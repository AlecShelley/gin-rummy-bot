"""
Build a warm-started NFSP agent for train_gin_rummy.py to continue from.

Why
---
NFSP from scratch does not learn this game (verified across six configurations;
see results/exp_*.csv and RESULTS.md). The blocker is coverage, not capacity:
knocking is only legal once deadwood drops to 10, random exploration reaches
that state in ~7% of hands, so the DQN almost never observes the reward for
going out. Behaviour cloning reaches -0.169 vs random with the identical
network and encoding, proving the representation is fine.

So we seed all three of NFSP's stateful parts from expert play, then hand the
agent to normal self-play training:

  1. average-policy network  <- supervised fit to the teacher's decisions
  2. reservoir buffer        <- the same (state, action) pairs, so fictitious
                                play starts from the teacher's strategy
                                instead of from noise
  3. DQN replay memory       <- teacher trajectories with potential-based
                                shaping, giving Q coverage of exactly the
                                low-deadwood states random play never visits

Roll-in uses a noisy teacher (DAgger-style): the *executed* action is
occasionally random so the data covers off-expert states, but the *label* is
always what the expert would have done there.

`--bc_only` skips all three NFSP-specific steps and writes just the policy
network (1.8 MB instead of 654 MB). The BC + DAgger pipeline is what actually
ships, and it never touches the DQN or the reservoir; seeding them costs most of
the roll-out time for nothing.

`--teacher` selects who is being cloned. `rule` is rlcard's novice agent, which
decides half its moves by coin flip; `strong` is the deterministic teacher in
teacher.py. Cloning a teacher that flips coins caps action accuracy near 76%,
because the clone faithfully copies the randomness.

    python pretrain.py --teacher strong --bc_only
        --episodes 16000 --epochs 30 --out results/v2_bc.policy.pt
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from rlcard.agents import NFSPAgent
from rlcard.agents.nfsp_agent import Transition as NFSPTransition
from rlcard.utils import get_device, set_seed, reorganize

from shaping import shape_trajectory
from teacher import make_teacher
from train_gin_rummy import make_env


class _Args:
    seed = 11
    reward = "shaped"


class NoisyTeacher:
    """Plays `expert`, but sometimes deviates so the data covers more states.

    Records what the expert *would* do at every visited state (the DAgger
    label), independently of what actually gets played.
    """

    use_raw = False

    def __init__(self, epsilon, expert):
        self.expert = expert
        self.epsilon = epsilon
        self.obs, self.labels = [], []

    def step(self, state):
        legal = list(state["legal_actions"].keys())
        expert_action = int(self.expert.step(state))
        if len(legal) > 1:
            self.obs.append(np.asarray(state["obs"], dtype=np.uint8))
            self.labels.append(expert_action)
            if np.random.rand() < self.epsilon:
                return int(np.random.choice(legal))
        return expert_action

    def eval_step(self, state):
        return self.step(state), {}


def main():
    ap = argparse.ArgumentParser("Warm-start an NFSP agent from expert play")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--bc_lr", type=float, default=1e-3)
    ap.add_argument("--teacher_epsilon", type=float, default=0.15)
    ap.add_argument("--teacher", choices=["rule", "smart", "strong"],
                    default="strong", help="which expert to clone")
    ap.add_argument("--bc_only", action="store_true",
                    help="clone the policy network only: no DQN replay, no "
                         "reservoir, and a 1.8 MB policy file instead of 654 MB")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--reward", choices=["shaped", "points"], default="shaped")
    ap.add_argument("--train_every", type=int, default=4)
    ap.add_argument("--out", default="results/pretrained.pth")
    args = ap.parse_args()

    _Args.reward = args.reward
    device = get_device()
    set_seed(_Args.seed)
    env = make_env(_Args)

    agent = NFSPAgent(
        num_actions=env.num_actions, state_shape=env.state_shape[0],
        hidden_layers_sizes=[512, 512], q_mlp_layers=[512, 512],
        reservoir_buffer_capacity=int(2e5), q_replay_memory_size=int(2e5),
        q_epsilon_decay_steps=1_000_000,
        rl_learning_rate=1e-4, sl_learning_rate=1e-4,
        train_every=args.train_every, q_train_every=args.train_every,
        device=device,
    )

    teachers = [NoisyTeacher(args.teacher_epsilon, make_teacher(args.teacher))
                for _ in range(2)]
    env.set_agents(teachers)

    # ---- 1. roll out expert play, feeding the DQN as we go -------------------
    print(f"rolling out {args.episodes} noisy-teacher episodes "
          f"(teacher={args.teacher}, epsilon={args.teacher_epsilon})", flush=True)
    t0 = time.time()
    import contextlib
    from train_gin_rummy import _QuietStdout
    import sys
    real = sys.stdout
    with contextlib.redirect_stdout(_QuietStdout(real)):
        for i in range(args.episodes):
            trajectories, payoffs = env.run(is_training=True)
            if not args.bc_only:
                trajectories = reorganize(trajectories, payoffs)
                for traj in trajectories:
                    shape_trajectory(traj, args.gamma)
                    for ts in traj:
                        # DQN is off-policy, so expert transitions are valid data
                        # -- and they cover the low-deadwood states that matter.
                        agent._rl_agent.feed(ts)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{args.episodes} episodes, "
                      f"{time.time() - t0:.0f}s", file=real, flush=True)

    obs = np.concatenate([np.array(t.obs, dtype=np.uint8) for t in teachers])
    labels = np.concatenate([np.array(t.labels, dtype=np.int64) for t in teachers])
    print(f"collected {len(obs)} expert decisions in {time.time() - t0:.0f}s; "
          f"DQN replay holds {len(agent._rl_agent.memory.memory)}", flush=True)

    # ---- 2. behaviour-clone the average policy network -----------------------
    net = agent.policy_network
    opt = torch.optim.Adam(net.parameters(), lr=args.bc_lr)
    lossf = nn.NLLLoss()
    X = torch.from_numpy(obs).float().to(device)
    y = torch.from_numpy(labels).to(device)
    n = len(X)
    print(f"behaviour cloning: {n} samples, {args.epochs} epochs", flush=True)
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        tot, correct, seen = 0.0, 0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
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
        print(f"  epoch {epoch + 1:>2}: loss {tot / seen:.4f} "
              f"acc {correct / seen:.1%}", flush=True)
    net.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.bc_only:
        # Same layout as save_policy() in train_gin_rummy.py, so GinRummyBot,
        # dagger.py and head_to_head.py all load it unchanged.
        torch.save({"num_actions": agent._num_actions,
                    "state_shape": agent._state_shape,
                    "mlp_layers": agent._layer_sizes,
                    "state_dict": net.state_dict()}, args.out)
        print(f"saved cloned policy to {args.out} "
              f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
        print(f"continue with:  python dagger.py --teacher {args.teacher} "
              f"--init {args.out} ...")
        return

    # ---- 3. seed the reservoir so fictitious play starts from the teacher ----
    cap = agent._reservoir_buffer._reservoir_buffer_capacity
    keep = min(n, cap)
    pick = np.random.choice(n, keep, replace=False)
    for i in pick:
        probs = np.zeros(env.num_actions, dtype=np.float32)
        probs[labels[i]] = 1.0
        agent._reservoir_buffer.add(NFSPTransition(info_state=obs[i],
                                                   action_probs=probs))
    print(f"seeded reservoir with {len(agent._reservoir_buffer)} transitions",
          flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(agent, args.out)
    size = os.path.getsize(args.out) / 1e6
    print(f"\nsaved warm-started NFSP agent to {args.out} ({size:.0f} MB)")
    print("continue with:  python train_gin_rummy.py --load_model "
          f"{args.out} --reward shaped ...")


if __name__ == "__main__":
    main()
