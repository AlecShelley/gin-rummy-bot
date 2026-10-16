"""
HistoryGinRummyEnv: the rlcard gin rummy env with the v3 (10, 52) observation.

The stock env's observation shows only current card locations; this one adds
the belief marginals and history planes from belief.build_obs. Both training
data generation and deployment go through that one function, so the network
never sees a distribution shift between the two (gate GA1).

Also fixes an inherited inefficiency: the stock _extract_state computes the
legal actions twice per state; this one computes them once.

Instantiate directly -- rlcard.make() bypasses subclasses:

    env = HistoryGinRummyEnv({"allow_step_back": False, "seed": 42})

or through train_gin_rummy.make_env(args) with args.obs == "hist".
"""

from collections import OrderedDict

import numpy as np
from rlcard.envs.gin_rummy import GinRummyEnv

from belief import NUM_PLANES, Belief, build_obs


class HistoryGinRummyEnv(GinRummyEnv):

    def __init__(self, config):
        super().__init__(config)
        self.state_shape = [[NUM_PLANES, 52] for _ in range(self.num_players)]
        self._beliefs = {}
        self._round_key = None

    def _extract_state(self, state):
        game = self.game
        if game.is_over():
            obs = np.zeros((NUM_PLANES, 52), dtype=np.float32)
            legal = OrderedDict()
            return {"obs": obs, "legal_actions": legal,
                    "raw_legal_actions": list(legal.keys()), "raw_obs": obs}

        # One Belief per seat per hand. A new round object means a new hand;
        # id() can be reused by the allocator, so stamp the round with a token.
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._beliefs = {0: Belief(0), 1: Belief(1)}

        legal_events = game.judge.get_legal_actions()   # once, not twice
        legal = OrderedDict((e.action_id, None) for e in legal_events)
        pid = game.round.current_player_id
        obs = build_obs(game, pid, self._beliefs[pid], list(legal.keys()))
        return {"obs": obs, "legal_actions": legal,
                "raw_legal_actions": list(legal.keys()), "raw_obs": obs}


def _verify_reproducibility(num_hands=120, seed=777):
    """GA1: the observation the env serves == the one the data generator
    builds, bit for bit, at every decision of every hand.

    Runs both pipelines against the SAME live game: the env extracts its
    state (its own Belief instances), while an independent generator-side
    Belief pair plus fast_legal_actions builds the observation the ExIt
    generator would record. Any drift in belief bookkeeping, legal-action
    computation, or plane encoding shows up as a mismatch here.
    """
    from sim import FastTeacher, fast_legal_actions
    from teacher import make_teacher  # noqa: F401  (env agents need not act)

    env = HistoryGinRummyEnv({"allow_step_back": False, "seed": seed})
    teacher = FastTeacher()
    checked = 0
    for h in range(num_hands):
        env.seed(seed + h)
        state, pid = env.reset()
        beliefs = {0: Belief(0), 1: Belief(1)}
        while not env.is_over():
            legal_events = fast_legal_actions(env.game)
            gen_ids = [e.action_id for e in legal_events]
            env_ids = list(state["legal_actions"].keys())
            assert sorted(gen_ids) == sorted(env_ids), (h, gen_ids, env_ids)
            gen_obs = build_obs(env.game, pid, beliefs[pid], gen_ids)
            assert gen_obs.dtype == state["obs"].dtype
            assert np.array_equal(gen_obs, state["obs"]), \
                (h, checked, np.argwhere(gen_obs != state["obs"])[:5])
            checked += 1
            action = legal_events[0] if len(legal_events) == 1 else \
                teacher.act(env.game, legal_events)
            state, pid = env.step(action.action_id)
    print(f"GA1: {checked} observations bit-identical across "
          f"{num_hands} env-driven hands")


if __name__ == "__main__":
    _verify_reproducibility()
