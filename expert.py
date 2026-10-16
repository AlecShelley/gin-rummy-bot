"""
The determinized rollout expert: belief-weighted search over sampled worlds.

At each real decision it samples K complete worlds consistent with what the
hero can see (opponent hidden cards drawn from the Belief weights, remainder
shuffled into the stock), plays every candidate action out to the end of the
hand in each world with FastTeacher driving both seats, and picks the action
with the best weighted-mean true-points payoff. The paired design -- same K
worlds under every candidate -- removes most of the between-world variance
from the comparison.

The knock-range constraint (the user's poker analogy, operationalized): if the
opponent is known to knock at deadwood <= t with probability p, then a sampled
world in which they could already knock is inconsistent with them not having
knocked. Such worlds get importance weight (1-p)^k, k = min(observed opponent
discards, 3): hard rejection at p=1, soft at p=0.5, no-op at p=0.

This player is far too slow to ship (~0.2 s/decision) -- it exists to (a)
prove there is strength beyond the teacher (gate GB), and (b) label training
data for the ExIt clone (exit_gen.py).

    python expert.py --eval --vs strong --hands 600 --workers 12
    python expert.py --eval --vs smart --hands 600 --ablate
"""

import argparse
import math
import multiprocessing as mp
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, GinAction, KnockAction)

import sim
from belief import Belief
from sim import (DECK, FastTeacher, MixedKnock, apply_world, cid,
                 fast_legal_actions, ids_of, make_scratch, min_deadwood_ids,
                 playout, restore, snapshot, step_fast)


class RolloutExpert:
    """Search config + the per-decision procedure. Stateless across hands
    except for the RNG; the caller owns the Belief."""

    def __init__(self, worlds=16, top_discards=4, opp_model=None,
                 uniform_worlds=False, seed=0, classes=("draw", "discard"),
                 truncate=16, z=2.0, sample_boost=0.0):
        self.worlds = worlds
        self.top_discards = top_discards
        self.opp_model = opp_model       # (threshold, p) or None
        self.uniform_worlds = uniform_worlds
        self.classes = classes           # which decision classes get search
        self.truncate = truncate         # rollout decision budget (None = full)
        self.z = z                       # deviation significance margin
        self.sample_boost = sample_boost  # meld-correlated worlds (belief.py)
        self.rng = np.random.default_rng(seed)
        self.rollout = {0: FastTeacher(), 1: FastTeacher()}
        self.teacher = FastTeacher()
        # The opponent model must shape the CONTINUATION, not just the world
        # weights: with a patient-teacher opponent seat in every rollout, a
        # conditioned searcher measured the same +3.4 vs smart as the
        # unconditioned one while the plain teacher took +7.4 (2026-08-23,
        # results/v4_value/gate_oppmodel.log) -- deviations were being
        # optimized against an opponent who never knocks.
        if opp_model is not None and opp_model[1] > 0.0:
            import random as _random
            self._opp_rollout = MixedKnock(FastTeacher(), opp_model[0],
                                           opp_model[1],
                                           rng=_random.Random(seed))
        else:
            self._opp_rollout = None

    # -- candidate selection ------------------------------------------------

    def _candidates(self, game, legal):
        """Search-worthy actions: singletons pass through; discard nodes are
        pruned to the teacher's top discards plus the best knock."""
        if len(legal) == 1:
            return legal
        discards = [e for e in legal if isinstance(e, DiscardAction)]
        knocks = [e for e in legal if isinstance(e, KnockAction)]
        if not discards and not knocks:
            return legal                 # draw decision: 2 candidates
        rnd = game.round
        hand_ids = ids_of(rnd.players[rnd.current_player_id].hand)
        chosen = []
        if discards:
            scored = []
            for event in discards:
                drop = cid(event.card)
                dw = min_deadwood_ids([i for i in hand_ids if i != drop])
                scored.append((dw, len(scored), event))
            scored.sort(key=lambda t: (t[0], t[1]))
            chosen.extend(e for _, _, e in scored[:self.top_discards])
        if knocks:
            best, best_dw = None, 999
            for event in knocks:
                drop = cid(event.card)
                dw = min_deadwood_ids([i for i in hand_ids if i != drop])
                if dw < best_dw:
                    best, best_dw = event, dw
            chosen.append(best)
        return chosen

    # -- world weighting ----------------------------------------------------

    def _world_weight(self, opp_hand_ids, opp_discard_count):
        if self.opp_model is None:
            return 1.0
        threshold, p = self.opp_model
        if p <= 0.0:
            return 1.0
        if min_deadwood_ids(opp_hand_ids) <= threshold:
            k = min(opp_discard_count, 3)
            return (1.0 - p) ** k if p < 1.0 else 0.0
        return 1.0

    # -- the decision -------------------------------------------------------

    def decide(self, game, belief):
        """Return the chosen ActionEvent for the current player of `game`."""
        return self.analyze(game, belief)["chosen"]

    def analyze(self, game, belief):
        """The full per-candidate evaluation at this decision, for decide()
        and for the position evaluator (evaluate.py). Behavior-identical to
        the historical decide() -- verified by refactor_check.py decision
        parity. Returns a dict with at least "chosen" and "reason"; searched
        nodes also carry candidates/payoff_means/diff_means/se/gain/t_idx/
        worlds (means on the payoff scale, points/100, hero view)."""
        legal = fast_legal_actions(game)
        candidates = self._candidates(game, legal)
        if len(candidates) == 1:
            return {"chosen": candidates[0], "reason": "forced",
                    "candidates": candidates}
        # Decision-class scoping (for ablation and hybrid experts): nodes of a
        # class not under search go to the teacher.
        has_discard = any(isinstance(e, DiscardAction) for e in legal)
        has_knock = any(isinstance(e, KnockAction) for e in legal)
        node = "discard" if has_discard or has_knock else "draw"
        if node not in self.classes:
            return {"chosen": self.teacher.act(game, legal),
                    "reason": "class-scoped", "candidates": candidates}

        hero_id = game.round.current_player_id
        opp_id = 1 - hero_id
        belief.update(game)

        # Stage-aware world realism: true opponent-hand deadwood falls ~52
        # (turn 0) to ~6 (turn 10+), so a flat boost overrates early
        # opponents and underrates late ones (corr_sample.py --stages,
        # 2026-08-23). "auto" ramps with the opponent's observed turn count.
        boost = self.sample_boost
        if boost == "auto":
            boost = min(3.0, 0.5 * (belief.opp_pickups + belief.opp_declines))

        # The teacher's action is the prior. Argmax over noisy rollout means
        # measured WORSE than the teacher (-16 pts/hand at K=16): per-world
        # payoffs swing +/-50 while true candidate gaps are ~1-3 points, so a
        # plain argmax mostly selects noise. Instead the expert deviates from
        # the teacher only when the PAIRED per-world difference is significant
        # at z standard errors -- worst case it equals the teacher.
        teacher_action = self.teacher.act(game, legal)
        t_idx = next((k for k, a in enumerate(candidates)
                      if a.action_id == teacher_action.action_id), None)
        if t_idx is None:
            candidates.insert(0, teacher_action)
            t_idx = 0

        rollout = self.rollout if self._opp_rollout is None else \
            {hero_id: self.rollout[hero_id], opp_id: self._opp_rollout}
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
                                                        self.rng, boost)
            else:
                free_ids = []
            opp_hand_ids = ids_of(opp.known_cards) + list(free_ids)
            weight = self._world_weight(opp_hand_ids, opp_discard_count)
            if weight <= 0.0:
                continue                 # inconsistent world (p=1 rejection)
            restore(scratch, base)
            apply_world(scratch, opp_id, [DECK[i] for i in free_ids], self.rng)
            world = snapshot(scratch)
            for k, action in enumerate(candidates):
                restore(scratch, world)
                step_fast(scratch, action)
                payoffs[worlds_done, k] = sign * playout(
                    scratch, rollout, truncate=self.truncate)
            weights[worlds_done] = weight
            worlds_done += 1

        if worlds_done == 0 or weights[:worlds_done].sum() <= 0.0:
            return {"chosen": teacher_action, "reason": "no-worlds",
                    "candidates": candidates}   # opponent model contradicted
        w = weights[:worlds_done]
        diffs = payoffs[:worlds_done] - payoffs[:worlds_done, [t_idx]]
        w_sum = w.sum()
        means = (w[:, None] * diffs).sum(axis=0) / w_sum
        centered = diffs - means
        var = (w[:, None] * centered ** 2).sum(axis=0) / w_sum
        n_eff = w_sum ** 2 / (w ** 2).sum()
        se = np.sqrt(var / max(n_eff - 1.0, 1.0))
        gain = means - self.z * se       # lower confidence bound vs teacher
        best = int(np.argmax(gain))
        deviate = gain[best] > 0.0
        payoff_means = (w[:, None] * payoffs[:worlds_done]).sum(axis=0) / w_sum
        return {"chosen": candidates[best] if deviate else teacher_action,
                "reason": "deviate" if deviate else "prior",
                "candidates": candidates, "t_idx": t_idx,
                "payoff_means": payoff_means, "diff_means": means,
                "se": se, "gain": gain, "worlds": worlds_done,
                "n_eff": float(n_eff)}


class ExpertAgent:
    """Env-facing wrapper. Reads the game through the env (the observation is
    ignored), maintains one Belief per hand keyed on the round object."""

    use_raw = False

    def __init__(self, expert, env):
        self.expert = expert
        self.env = env
        self._round_key = None
        self._belief = None

    def _belief_for(self, game):
        # id(round) can be REUSED by the allocator across hands; stamp the
        # round with a token instead.
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._belief = Belief(rnd.current_player_id)
        return self._belief

    def step(self, state):
        game = self.env.game
        legal_ids = list(state["legal_actions"].keys())
        if len(legal_ids) == 1:
            return legal_ids[0]
        action = self.expert.decide(game, self._belief_for(game))
        return action.action_id

    def eval_step(self, state):
        return self.step(state), {}


# ---------------------------------------------------------------------------
# Parallel seat-swapped evaluation (gate GB).
# ---------------------------------------------------------------------------

def _play_chunk(job):
    """Worker: `hands` games vs the named opponent, expert in `seat`."""
    (opponent, hands, seat, seed, worlds, top_discards,
     opp_model, uniform_worlds, classes, truncate, z, rules) = job
    import sim as _sim
    _sim.set_rules(rules)
    from train_gin_rummy import make_env

    class _Args:
        reward = "points"
        obs = "base"                     # opponent needs the stock 5x52 obs

    _Args.seed = seed
    _Args.rules = rules
    env = make_env(_Args)
    expert = RolloutExpert(worlds=worlds, top_discards=top_discards,
                           opp_model=opp_model,
                           uniform_worlds=uniform_worlds, seed=seed,
                           classes=classes, truncate=truncate, z=z)
    agent = ExpertAgent(expert, env)
    from gin_rummy_bot import load_opponent
    opp, _ = load_opponent(opponent, env.num_actions, greedy=True)
    env.set_agents([agent, opp] if seat == 0 else [opp, agent])
    env.seed(seed)
    points = []
    for _ in range(hands):
        _, payoffs = env.run(is_training=False)
        points.append(payoffs[seat] * 100.0)
    return points


def evaluate(opponent="strong", hands=600, workers=12, worlds=16,
             top_discards=4, opp_model=None, uniform_worlds=False, seed=606,
             classes=("draw", "discard"), truncate=16, z=2.0,
             rules="classic"):
    """Seat-swapped match vs a named opponent; returns (mean, ci95)."""
    per = max(1, hands // (2 * workers))
    jobs = []
    for w in range(workers):
        for seat in (0, 1):
            jobs.append((opponent, per, seat, seed + 1000 * w + seat,
                         worlds, top_discards, opp_model, uniform_worlds,
                         classes, truncate, z, rules))
    t0 = time.time()
    with mp.Pool(workers) as pool:
        chunks = pool.map(_play_chunk, jobs)
    points = np.concatenate(chunks)
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    print(f"expert(K={worlds},A={top_discards},opp_model={opp_model},"
          f"uniform={uniform_worlds},classes={classes},T={truncate},"
          f"z={z},rules={rules}) vs {opponent}: "
          f"{mean:+.2f} +/- {ci:.2f} over {len(points)} hands "
          f"({time.time() - t0:.0f}s)", flush=True)
    return mean, ci


# Opponent models for the named opponents: what the hero may assume about
# their knocking. smart/rule knock the moment it is legal (t=10, p=1);
# strong only ever gins (t=0, p=1); v2's clone behaves like strong.
OPP_MODELS = {
    "smart": (10, 1.0),
    "rule": (10, 1.0),
    "strong": (0, 1.0),
    "random": None,
}


def main():
    ap = argparse.ArgumentParser("rollout expert evaluation")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--vs", default="strong")
    ap.add_argument("--hands", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--top_discards", type=int, default=4)
    ap.add_argument("--no_opp_model", action="store_true")
    ap.add_argument("--uniform_worlds", action="store_true")
    ap.add_argument("--ablate", action="store_true",
                    help="run belief/uniform x opp-model on/off grid")
    ap.add_argument("--seed", type=int, default=606)
    args = ap.parse_args()

    if args.ablate:
        for uniform in (False, True):
            for use_model in (True, False):
                evaluate(args.vs, args.hands, args.workers, args.worlds,
                         args.top_discards,
                         OPP_MODELS.get(args.vs) if use_model else None,
                         uniform, args.seed)
        return
    if args.eval:
        model = None if args.no_opp_model else OPP_MODELS.get(args.vs)
        evaluate(args.vs, args.hands, args.workers, args.worlds,
                 args.top_discards, model, args.uniform_worlds, args.seed)


if __name__ == "__main__":
    main()
