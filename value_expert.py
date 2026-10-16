"""
The v4 candidate: RolloutExpert with a learned value leaf (roadmap #1).

Identical search discipline to expert.RolloutExpert -- belief-weighted
determinized worlds, paired candidates, teacher prior with the z-SE deviation
test (lesson #3: never argmax noisy rollout values). The one change: truncated
rollouts end in `valuenet` features, batched through the trained net once per
decision, instead of the handcrafted effective-deadwood leaf.

decide() is a modified copy of RolloutExpert.decide (the payoff-filling loop
defers leaf evaluation to one batch); consolidate after the gates pass.

Gating harness (seat-swapped, own registry so Windows spawn workers can
rebuild policies from names + config tuples):

    .venv/Scripts/python.exe value_expert.py --vs v3search --hands 1500
    .venv/Scripts/python.exe value_expert.py --vs teacher --hands 1500
"""

import argparse
import math
import multiprocessing as mp
import random
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, KnockAction)

import sim
from belief import Belief
from expert import RolloutExpert
from sim import (DECK, FastTeacher, apply_world, cid, fast_legal_actions,
                 ids_of, make_scratch, playout, restore, snapshot, step_fast)
from valuenet import ValueLeaf, leaf_features

DEFAULT_VALUE_PATH = "results/v4_value/value.pt"


def playout_to_leaf(game, policies, truncate):
    """sim.playout with truncation, but the truncation frontier comes back as
    features, not a handcrafted value.

    Returns (payoff0, None) when the hand ended inside the budget, else
    (None, (planes, scalars, leaf_player_id))."""
    rnd = game.round
    for _ in range(truncate):
        if rnd.going_out_action is not None or rnd.is_over:
            return sim._payoff(game), None
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 \
            else policies[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
    if rnd.going_out_action is None and not rnd.is_over:
        planes, scalars = leaf_features(game)
        return None, (planes, scalars, rnd.current_player_id)
    return sim._payoff(game), None


class ValueRolloutExpert(RolloutExpert):
    """RolloutExpert whose truncation leaf is the trained value net."""

    def __init__(self, value_path=DEFAULT_VALUE_PATH, device="cpu", **kw):
        super().__init__(**kw)
        self.leaf = ValueLeaf(value_path, device=device)

    def decide(self, game, belief):
        legal = fast_legal_actions(game)
        candidates = self._candidates(game, legal)
        if len(candidates) == 1:
            return candidates[0]
        has_discard = any(isinstance(e, DiscardAction) for e in legal)
        has_knock = any(isinstance(e, KnockAction) for e in legal)
        node = "discard" if has_discard or has_knock else "draw"
        if node not in self.classes:
            return self.teacher.act(game, legal)

        hero_id = game.round.current_player_id
        opp_id = 1 - hero_id
        belief.update(game)

        teacher_action = self.teacher.act(game, legal)
        t_idx = next((k for k, a in enumerate(candidates)
                      if a.action_id == teacher_action.action_id), None)
        if t_idx is None:
            candidates.insert(0, teacher_action)
            t_idx = 0

        scratch = make_scratch(game)
        base = snapshot(scratch)
        opp = scratch.round.players[opp_id]
        unseen = ids_of(scratch.round.dealer.stock_pile) + \
            [cid(c) for c in opp.hand if c not in opp.known_cards]
        num_hidden = len(opp.hand) - len(opp.known_cards)
        opp_discard_count = int(belief.opp_discarded.sum())

        payoffs = np.zeros((self.worlds, len(candidates)))
        weights = np.zeros(self.worlds)
        sign = 1.0 if hero_id == 0 else -1.0
        pending = []                     # (world_idx, cand_idx, leaf_pid)
        feats = []
        attempts = 0
        worlds_done = 0
        max_attempts = 3 * self.worlds
        while worlds_done < self.worlds and attempts < max_attempts:
            attempts += 1
            if num_hidden > 0:
                if self.uniform_worlds:
                    pick = self.rng.choice(len(unseen), size=num_hidden,
                                           replace=False)
                    free_ids = [unseen[i] for i in pick]
                else:
                    free_ids = belief.sample_free_cards(unseen, num_hidden,
                                                        self.rng)
            else:
                free_ids = []
            opp_hand_ids = ids_of(opp.known_cards) + list(free_ids)
            weight = self._world_weight(opp_hand_ids, opp_discard_count)
            if weight <= 0.0:
                continue
            restore(scratch, base)
            apply_world(scratch, opp_id, [DECK[i] for i in free_ids], self.rng)
            world = snapshot(scratch)
            for k, action in enumerate(candidates):
                restore(scratch, world)
                step_fast(scratch, action)
                payoff0, leaf = playout_to_leaf(scratch, self.rollout,
                                                self.truncate)
                if leaf is None:
                    payoffs[worlds_done, k] = sign * payoff0
                else:
                    planes, scalars, leaf_pid = leaf
                    pending.append((worlds_done, k, leaf_pid))
                    feats.append((planes, scalars))
            weights[worlds_done] = weight
            worlds_done += 1

        if worlds_done == 0 or weights[:worlds_done].sum() <= 0.0:
            return teacher_action
        if feats:
            values = self.leaf.evaluate(feats)
            for (wi, k, leaf_pid), v in zip(pending, values):
                payoff0 = v if leaf_pid == 0 else -v
                payoffs[wi, k] = sign * payoff0

        w = weights[:worlds_done]
        diffs = payoffs[:worlds_done] - payoffs[:worlds_done, [t_idx]]
        w_sum = w.sum()
        means = (w[:, None] * diffs).sum(axis=0) / w_sum
        centered = diffs - means
        var = (w[:, None] * centered ** 2).sum(axis=0) / w_sum
        n_eff = w_sum ** 2 / (w ** 2).sum()
        se = np.sqrt(var / max(n_eff - 1.0, 1.0))
        gain = means - self.z * se
        best = int(np.argmax(gain))
        return candidates[best] if gain[best] > 0.0 else teacher_action


class _ValuePolicy:
    """Engine-facing wrapper (mirrors meta._ExpertPolicy)."""

    def __init__(self, worlds, truncate, z, value_path, device, seed=0,
                 leaf="net"):
        if leaf == "hand":               # control: handcrafted leaf, same K/T
            self.expert = RolloutExpert(worlds=worlds, truncate=truncate,
                                        z=z, opp_model=None, seed=seed)
        else:
            self.expert = ValueRolloutExpert(value_path=value_path,
                                             device=device, worlds=worlds,
                                             truncate=truncate, z=z,
                                             opp_model=None, seed=seed)
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
            self._belief = Belief(rnd.current_player_id)
        return self.expert.decide(game, self._belief)


def _vmatch_chunk(job):
    """Worker: v4(cfg) vs a meta population member, seat-swapped points."""
    cfg, opp_name, hands, seat, seed, rules = job
    sim.set_rules(rules)
    import meta
    rng = random.Random(seed)
    pol_a = _ValuePolicy(cfg["worlds"], cfg["truncate"], cfg["z"],
                         cfg["value_path"], cfg["device"], seed=seed,
                         leaf=cfg.get("leaf", "net"))
    pol_b = meta.population()[opp_name]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    return points


def vmatch(cfg, opp_name, hands, workers, seed, rules="full"):
    per = max(1, hands // (2 * workers))
    jobs = [(cfg, opp_name, per, seat, seed + 31 * w + seat, rules)
            for w in range(workers) for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        points = np.concatenate(pool.map(_vmatch_chunk, jobs))
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    print(f"v4(K={cfg['worlds']},T={cfg['truncate']},z={cfg['z']},"
          f"leaf={cfg.get('leaf', 'net')}) vs "
          f"{opp_name} [{rules}]: {mean:+.2f} +/- {ci:.2f} over "
          f"{len(points)} hands ({time.time() - t0:.0f}s)", flush=True)
    return mean, ci, len(points)


def main():
    ap = argparse.ArgumentParser("v4 value-leaf expert gate")
    ap.add_argument("--vs", default="teacher")
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--truncate", type=int, default=16)
    ap.add_argument("--z", type=float, default=1.5)
    ap.add_argument("--value_path", default=DEFAULT_VALUE_PATH)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--leaf", default="net", choices=("net", "hand"))
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=41000)
    args = ap.parse_args()
    cfg = {"worlds": args.worlds, "truncate": args.truncate, "z": args.z,
           "value_path": args.value_path, "device": args.device,
           "leaf": args.leaf}
    vmatch(cfg, args.vs, args.hands, args.workers, args.seed, args.rules)


if __name__ == "__main__":
    main()
