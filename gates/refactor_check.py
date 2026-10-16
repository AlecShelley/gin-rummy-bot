import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""Determinism snapshot: every expert decision over fixed hands/seeds.
Run before and after the decide->analyze refactor; outputs must be identical.
"""
import random
import sys

import sim
sim.set_rules("full")
from sim import FastTeacher, fast_legal_actions, step_fast
from belief import Belief
from expert import RolloutExpert

out = []
for boost, opp_model in ((0.0, None), (2.5, (10, 1.0))):
    ex = RolloutExpert(worlds=24, truncate=16, z=1.5, opp_model=opp_model,
                       seed=11, sample_boost=boost)
    teach = FastTeacher()
    rng = random.Random(2024)
    for _ in range(6):
        game = sim.new_game(rng.randrange(1 << 30))
        rnd = game.round
        belief = Belief(0)
        while rnd.going_out_action is None and not rnd.is_over:
            legal = fast_legal_actions(game)
            if len(legal) == 1:
                step_fast(game, legal[0])
                continue
            if rnd.current_player_id == 0:
                a = ex.decide(game, belief)
                out.append(a.action_id)
            else:
                a = teach.act(game, legal)
            step_fast(game, a)
print(",".join(map(str, out)))
