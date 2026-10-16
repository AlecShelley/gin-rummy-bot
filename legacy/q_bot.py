import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Gate 1: the v5 Q-net as a standalone millisecond bot.

Plays argmax Q over the SAME candidate set the label expert scored --
RolloutExpert._candidates (teacher-pruned discards + best knock) plus the
teacher's action when pruning dropped it. Never argmax over all legal ids:
the 110-dim head only carries training signal on label support (--mask legal
exists as an ablation).

Own spawn-safe seat-swapped match harness (value_expert.vmatch pattern):
config rides the job tuple as a plain dict, workers set rules first,
opponents come from meta.population() by name. meta.py itself is not touched
until promotion.

    .venv/Scripts/python.exe q_bot.py --vs teacher --hands 1500 --seed 91000
    .venv/Scripts/python.exe q_bot.py --vs teacher --hands 4000 --seed 92000
"""

import argparse
import math
import multiprocessing as mp
import random
import time

import numpy as np

import sim
from belief import Belief, build_obs
from expert import RolloutExpert
from legacy.qnet import QNet
from sim import FastTeacher, playout

DEFAULT_QNET_PATH = "results/v5_q/qnet.pt"


class QPolicy:
    """Engine-facing Q policy (mirrors meta._ExpertPolicy's shape).

    dev_margin: play the TEACHER'S action unless the net's best candidate
    clears it by more than this many points -- the standalone analogue of the
    search's own prior + z-test discipline (lesson #3: never argmax a noisy
    evaluator; greedy argmax measured -5.43 +/- 1.41 vs teacher). 0 = greedy.
    """

    def __init__(self, net_path=DEFAULT_QNET_PATH, device="cpu",
                 mask="cand", top_discards=4, dev_margin=0.0,
                 dev_classes="dkr"):
        self.net = QNet(net_path, device=device)
        self.mask = mask
        self.dev_margin = dev_margin
        self.dev_classes = dev_classes   # d=draw(2/3) r=discard(6-57) k=knock
        # _candidates is a pure function of (game, legal); no search runs.
        self._pruner = RolloutExpert(top_discards=top_discards)
        self.teacher = FastTeacher()
        self._round_key = None
        self._belief = None

    def _belief_for(self, game):
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._belief = Belief(rnd.current_player_id)
        return self._belief

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        belief = self._belief_for(game)
        pid = game.round.current_player_id
        teacher_action = self.teacher.act(game, legal)
        if self.mask == "legal":
            cands = list(legal)
        else:
            cands = self._pruner._candidates(game, legal)
            if all(a.action_id != teacher_action.action_id for a in cands):
                cands.insert(0, teacher_action)
        obs = build_obs(game, pid, belief, [e.action_id for e in legal])
        q = self.net.q_values(obs)
        ids = [a.action_id for a in cands]
        best = cands[int(np.argmax(q[ids]))]
        if self.dev_margin <= 0.0:
            return best
        bid = best.action_id
        cls = "d" if bid in (2, 3) else ("k" if bid >= 58 else "r")
        if cls in self.dev_classes and \
                q[bid] - q[teacher_action.action_id] > self.dev_margin:
            return best
        return teacher_action


class QBotAgent:
    """rlcard-env wrapper (ExpertAgent pattern) for play_vs_bot."""

    use_raw = False

    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def step(self, state):
        import sim as _sim
        legal = _sim.fast_legal_actions(self.env.game)
        action = self.policy.act(self.env.game, legal)
        return action.action_id

    def eval_step(self, state):
        return self.step(state), {}


def _qmatch_chunk(job):
    cfg, opp_name, hands, seat, seed, rules = job
    sim.set_rules(rules)
    import meta
    rng = random.Random(seed)
    pol_a = QPolicy(cfg["net_path"], cfg["device"], cfg["mask"],
                    dev_margin=cfg.get("dev_margin", 0.0),
                    dev_classes=cfg.get("dev_classes", "dkr"))
    pol_b = meta.population()[opp_name]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    return points


def qmatch(cfg, opp_name, hands, workers, seed, rules="full"):
    per = max(1, hands // (2 * workers))
    jobs = [(cfg, opp_name, per, seat, seed + 31 * w + seat, rules)
            for w in range(workers) for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        points = np.concatenate(pool.map(_qmatch_chunk, jobs))
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    print(f"v5q(mask={cfg['mask']},margin={cfg.get('dev_margin', 0.0)}) "
          f"vs {opp_name} [{rules}]: "
          f"{mean:+.2f} +/- {ci:.2f} over {len(points)} hands "
          f"({time.time() - t0:.0f}s)", flush=True)
    return mean, ci, len(points)


def main():
    ap = argparse.ArgumentParser("v5 standalone Q-bot gate")
    ap.add_argument("--vs", default="teacher")
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--net_path", default=DEFAULT_QNET_PATH)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mask", default="cand", choices=("cand", "legal"))
    ap.add_argument("--dev_margin", type=float, default=0.0)
    ap.add_argument("--dev_classes", default="dkr")
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=91000)
    args = ap.parse_args()
    cfg = {"net_path": args.net_path, "device": args.device,
           "mask": args.mask, "dev_margin": args.dev_margin,
           "dev_classes": args.dev_classes}
    qmatch(cfg, args.vs, args.hands, args.workers, args.seed, args.rules)


if __name__ == "__main__":
    main()
