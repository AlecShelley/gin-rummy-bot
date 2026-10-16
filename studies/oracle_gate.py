import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Track 1 Gate 0 (v6): the full-knowledge oracle bound.

Richards & Amir (IJCAI-07) measured a +37 pts/game oracle bound before
building Scrabble rack inference, then realized +5.2 of it. Same logic
here: play v5's exact search config but with the opponent's TRUE hidden
cards as every sampled world (stock order still unknown -- that is the
part no belief can ever know). The head-to-head margin of this oracle
over stock v5 is an upper bound on what ANY opponent-hand belief
improvement (particle filter included) can be worth in-search. If it is
inside the noise floor, the Track 1 in-search swap cannot pay.

The ONLY manipulated variable is world sampling:
  * OracleBelief.sample_free_cards returns the true hidden ids
    (recorded at update() from the real game).
  * OracleExpert._world_weight returns 1.0 -- the true world must not be
    rejected by the (10, 1.0) knock-model veto (a knockable true hand
    would otherwise force the teacher fallback and corrupt the bound).
  * Everything else is the promoted v5 config, including the MixedKnock
    rollout continuation.

    .venv/Scripts/python.exe oracle_gate.py --vs v5 --hands 1500 --seed 60100
"""

import argparse
import math
import multiprocessing as mp
import random
import time

import numpy as np

import sim
from adaptive import AdaptiveExpert
from belief import Belief
from sim import cid, playout

V5_CFG = dict(stages=(16, 48, 192), deep=True, deep_from=2, kbranch=3,
              endgame_stock=0, mid_alpha=0.5, truncate=16, z=1.5)


class OracleBelief(Belief):
    """Belief that also records the TRUE opponent hidden ids at update().

    update() still runs the normal event replay (opp_discarded etc. feed
    opp_discard_count in the search loop); the peek is additive.
    """

    def __init__(self, hero_id):
        super().__init__(hero_id)
        self._true_hidden = []

    def update(self, game):
        super().update(game)
        opp = game.round.players[1 - self.hero_id]
        self._true_hidden = [cid(c) for c in opp.hand
                             if c not in opp.known_cards]

    def sample_free_cards(self, unseen_ids, num_hidden, rng, boost=0.0):
        return list(self._true_hidden)


class OracleExpert(AdaptiveExpert):
    def _world_weight(self, opp_hand_ids, opp_discard_count):
        return 1.0                       # the true world is never rejected


class OraclePolicy:
    """Engine-facing wrapper, token pattern (same as _AdaptivePolicy)."""

    def __init__(self, seed=0):
        self.expert = OracleExpert(opp_model=(10, 1.0), sample_boost=0.0,
                                   seed=seed, **V5_CFG)
        self._round_key = None
        self._belief = None

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._belief = OracleBelief(rnd.current_player_id)
        return self.expert.decide(game, self._belief)


def _chunk(job):
    opp_name, hands, seat, seed, rules = job
    sim.set_rules(rules)
    import meta
    rng = random.Random(seed)
    pol_a = OraclePolicy(seed=seed)
    pol_b = meta.population()[opp_name]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    return points


def omatch(opp_name, hands, workers, seed, rules="full"):
    per = max(1, hands // (2 * workers))
    jobs = [(opp_name, per, seat, seed + 31 * w + seat, rules)
            for w in range(workers) for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        points = np.concatenate(pool.map(_chunk, jobs))
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    print(f"oracle-v5 vs {opp_name} [{rules}]: {mean:+.2f} +/- {ci:.2f} "
          f"over {len(points)} hands ({time.time() - t0:.0f}s wall)",
          flush=True)
    return mean, ci, len(points)


def main():
    ap = argparse.ArgumentParser("v6 gate 0: oracle bound")
    ap.add_argument("--vs", default="v5")
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=60100)
    args = ap.parse_args()
    omatch(args.vs, args.hands, args.workers, args.seed, args.rules)


if __name__ == "__main__":
    main()
