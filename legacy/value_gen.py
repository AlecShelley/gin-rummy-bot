import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Outcome-labeled data generation for the value leaf (roadmap #1).

Each hand: roll in with a diverse prefix (teacher / mixed-knock / noisy /
random, sampled per seat) for a random number of steps, then hand BOTH seats
to FastTeacher for the rest -- the exact continuation the truncation leaf
assumes. Every multi-action decision state after the switch is recorded
(subsampled) via `valuenet.leaf_features`, and every record from the hand is
labeled with the hand's final full-rules payoff from that state's
current-player perspective.

The prefix diversity is coverage, not target drift: labels are only ever
generated under teacher-vs-teacher continuation, so V regresses on the right
expectation everywhere.

Shards land as compressed npz: planes uint8 (N,6,52), scalars float32 (N,8),
labels float32 (N,) in payoff units.

    .venv/Scripts/python.exe value_gen.py --hands 60000 --workers 12 \
        --out results/v4_value/train --seed 90001
"""

import argparse
import multiprocessing as mp
import os
import random
import time

import numpy as np

import sim
from sim import FastTeacher, MixedKnock, fast_legal_actions, step_fast
from legacy.valuenet import leaf_features


class _RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def act(self, game, legal):
        return legal[self.rng.randrange(len(legal))]


class _NoisyTeacher:
    """FastTeacher with 20% random moves -- messy but plausible states."""

    def __init__(self, rng):
        self.rng = rng
        self.teacher = FastTeacher()

    def act(self, game, legal):
        if self.rng.random() < 0.2:
            return legal[self.rng.randrange(len(legal))]
        return self.teacher.act(game, legal)


# (name, weight): sampled independently per seat per hand.
PREFIX_MIX = (("teacher", 0.35), ("knock10", 0.15), ("knock6", 0.10),
              ("noisy", 0.25), ("random", 0.15))
MAX_PREFIX = 24                          # steps before the teacher takes over


def _build_prefix(name, rng):
    if name == "teacher":
        return FastTeacher()
    if name == "knock10":
        return MixedKnock(FastTeacher(), 10, 1.0, rng=random.Random(rng.randrange(1 << 30)))
    if name == "knock6":
        return MixedKnock(FastTeacher(), 6, 1.0, rng=random.Random(rng.randrange(1 << 30)))
    if name == "noisy":
        return _NoisyTeacher(rng)
    return _RandomPolicy(rng)


def _sample_prefix(rng):
    r, acc = rng.random(), 0.0
    for name, w in PREFIX_MIX:
        acc += w
        if r < acc:
            return name
    return PREFIX_MIX[-1][0]


def gen_chunk(job):
    """Worker: `hands` games; returns (#states, out_path)."""
    hands, seed, out_path, rules, record_p = job
    sim.set_rules(rules)
    rng = random.Random(seed)
    teacher = FastTeacher()

    planes_out, scalars_out, labels_out = [], [], []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        prefix = {0: _build_prefix(_sample_prefix(rng), rng),
                  1: _build_prefix(_sample_prefix(rng), rng)}
        switch_at = rng.randrange(MAX_PREFIX + 1)
        rnd = game.round
        records = []                     # (index into out arrays, hero_id)
        steps = 0
        while rnd.going_out_action is None and not rnd.is_over \
                and steps < 250:
            legal = fast_legal_actions(game)
            in_teacher_phase = steps >= switch_at
            if len(legal) > 1 and in_teacher_phase \
                    and rng.random() < record_p:
                planes, scalars = leaf_features(game)
                planes_out.append(planes)
                scalars_out.append(scalars)
                records.append(rnd.current_player_id)
                labels_out.append(0.0)   # patched after the hand
            if len(legal) == 1:
                action = legal[0]
            elif in_teacher_phase:
                action = teacher.act(game, legal)
            else:
                action = prefix[rnd.current_player_id].act(game, legal)
            step_fast(game, action)
            steps += 1
        payoff0 = sim._payoff(game)
        for k, hero_id in enumerate(records):
            idx = len(labels_out) - len(records) + k
            labels_out[idx] = payoff0 if hero_id == 0 else -payoff0

    planes_arr = np.stack(planes_out) if planes_out \
        else np.zeros((0, 6, 52), np.uint8)
    scalars_arr = np.stack(scalars_out) if scalars_out \
        else np.zeros((0, 8), np.float32)
    np.savez_compressed(out_path, planes=planes_arr, scalars=scalars_arr,
                        labels=np.array(labels_out, dtype=np.float32))
    return len(labels_out), out_path


def generate(hands, workers, out_prefix, rules="full", record_p=0.5,
             seed=90001, splits=4):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    jobs = []
    per = max(1, hands // (workers * splits))
    for w in range(workers):
        for s in range(splits):
            out_path = f"{out_prefix}_w{w}_{s}.npz"
            jobs.append((per, seed + 1009 * w + 101 * s, out_path, rules,
                         record_p))
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(gen_chunk, jobs)
    total = sum(n for n, _ in results)
    dt = time.time() - t0
    print(f"generated {total} labeled states from ~{hands} hands in {dt:.0f}s "
          f"({total / dt:.0f} states/s) -> {out_prefix}_*.npz", flush=True)
    return [p for _, p in results]


def main():
    ap = argparse.ArgumentParser("value-leaf data generation")
    ap.add_argument("--hands", type=int, default=60000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/v4_value/train")
    ap.add_argument("--rules", default="full")
    ap.add_argument("--record_p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=90001)
    args = ap.parse_args()
    generate(args.hands, args.workers, args.out, args.rules, args.record_p,
             args.seed)


if __name__ == "__main__":
    main()
