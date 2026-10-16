import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Parallel ExIt data generation: search-labeled decisions for the v3 clone.

Workers drive games directly through the sim kernel (no Env). The hero at
every multi-action decision is labeled by the RolloutExpert; what actually
gets PLAYED depends on the roll-in mode:

  * --rollin expert  : the expert's own choice is played (BC data);
  * --rollin <path>  : a v3 clone policy plays (DAgger data -- states come
    from the clone's distribution, labels from the expert).

Opponent mix mirrors what v2 measured to be necessary: self-play alone never
shows the long hands weak opponents produce. Shares: self 40%, strong 20%,
v2-clone 20%, random 20%. In self mode both seats are heroes and both are
recorded (belief per seat).

Shards land in --out as compressed npz: obs float16 (N,10,52), labels int16.

    python exit_gen.py --hands 8000 --workers 12 --out results/v3_data/bc
"""

import argparse
import multiprocessing as mp
import os
import random
import time

import numpy as np

import sim
from belief import Belief, base_obs, build_obs
from expert import OPP_MODELS, RolloutExpert
from sim import FastTeacher, fast_legal_actions, step_fast


class _RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def act(self, game, legal):
        return legal[self.rng.randrange(len(legal))]


class _V2Clone:
    """The v2 network as a fast policy (base 5x52 obs, greedy)."""

    def __init__(self, path):
        import torch
        torch.set_num_threads(1)
        from gin_rummy_bot import GinRummyBot
        self.bot = GinRummyBot(path, device=torch.device("cpu"))

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        ids = [e.action_id for e in legal]
        probs = self.bot._probs({"obs": base_obs(game),
                                 "legal_actions": dict.fromkeys(ids)})
        return legal[ids.index(int(np.argmax(probs)))]


class _V3Clone:
    """A v3 policy as a fast policy: builds its own (10,52) obs. Used as the
    DAgger roll-in actor; the caller supplies the per-hand Belief."""

    def __init__(self, path):
        import torch
        torch.set_num_threads(1)
        from rlcard.agents.nfsp_agent import AveragePolicyNetwork
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = AveragePolicyNetwork(num_actions=blob["num_actions"],
                                   state_shape=blob["state_shape"],
                                   mlp_layers=blob["mlp_layers"])
        net.load_state_dict(blob["state_dict"])
        net.eval()
        self.net = net
        self.torch = torch

    def act_with_obs(self, obs, legal):
        x = self.torch.from_numpy(obs.reshape(1, -1).astype(np.float32))
        with self.torch.no_grad():
            log_probs = self.net(x).numpy()[0]
        ids = [e.action_id for e in legal]
        probs = np.exp(log_probs)[ids]
        return legal[int(np.argmax(probs))]


def gen_chunk(job):
    """Worker: play `hands` games, label hero decisions, write one shard."""
    (mode, opponent, hands, seed, out_path, worlds, truncate, z,
     rollin_path) = job
    rng = random.Random(seed)
    expert = RolloutExpert(worlds=worlds, truncate=truncate, z=z,
                           opp_model=OPP_MODELS.get(opponent, (0, 1.0)),
                           seed=seed)
    rollin = _V3Clone(rollin_path) if rollin_path else None
    if opponent == "self":
        opp_policy = None                # the other hero
    elif opponent == "strong":
        opp_policy = FastTeacher()
    elif opponent == "v2":
        opp_policy = _V2Clone("results/v2_final/gin_rummy_bot.policy.pt")
    else:
        opp_policy = _RandomPolicy(rng)

    obs_out, label_out = [], []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        hero_seats = (0, 1) if opponent == "self" else (rng.randrange(2),)
        beliefs = {s: Belief(s) for s in hero_seats}
        while game.round.going_out_action is None and not game.round.is_over:
            pid = game.round.current_player_id
            legal = fast_legal_actions(game)
            if len(legal) == 1:
                step_fast(game, legal[0])
                continue
            if pid in beliefs:
                belief = beliefs[pid]
                label_action = expert.decide(game, belief)
                obs = build_obs(game, pid, belief,
                                [e.action_id for e in legal])
                obs_out.append(obs.astype(np.float16))
                label_out.append(label_action.action_id)
                if rollin is None:
                    action = label_action
                else:
                    action = rollin.act_with_obs(obs, legal)
            else:
                action = opp_policy.act(game, legal)
            step_fast(game, action)
    obs_arr = np.stack(obs_out) if obs_out else np.zeros((0, 10, 52), np.float16)
    np.savez_compressed(out_path, obs=obs_arr,
                        labels=np.array(label_out, dtype=np.int16))
    return len(label_out), out_path


SHARES = (("self", 0.40), ("strong", 0.20), ("v2", 0.20), ("random", 0.20))


def generate(hands, workers, out_prefix, worlds=48, truncate=16, z=1.5,
             rollin_path=None, seed=8000):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    # Small jobs: shards land incrementally and stragglers even out. Long
    # chunks measured badly -- an hour of compute with nothing on disk.
    jobs = []
    splits = 3
    per_worker = max(1, hands // workers)
    for w in range(workers):
        offset = 0
        for opponent, share in SHARES:
            for split in range(splits):
                n = max(1, int(per_worker * share / splits))
                out_path = f"{out_prefix}_w{w}_{opponent}_{split}.npz"
                jobs.append(("gen", opponent, n, seed + 100 * w + 10 * split
                             + offset, out_path, worlds, truncate, z,
                             rollin_path))
            offset += 1
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(gen_chunk, jobs)
    total = sum(n for n, _ in results)
    dt = time.time() - t0
    print(f"generated {total} labeled decisions from ~{hands} hands "
          f"in {dt:.0f}s ({total / dt:.0f}/s) -> {out_prefix}_*.npz",
          flush=True)
    return [p for _, p in results]


def main():
    ap = argparse.ArgumentParser("ExIt data generation")
    ap.add_argument("--hands", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/v3_data/bc")
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--truncate", type=int, default=16)
    ap.add_argument("--z", type=float, default=1.5)
    ap.add_argument("--rollin", default=None,
                    help="path to a v3 policy for DAgger roll-in "
                         "(default: the expert plays its own labels)")
    ap.add_argument("--seed", type=int, default=8000)
    args = ap.parse_args()
    generate(args.hands, args.workers, args.out, args.worlds, args.truncate,
             args.z, args.rollin, args.seed)


if __name__ == "__main__":
    main()
