"""
BeliefTeacher: FastTeacher with hand-reading in the discard safety key.

The user-identified weak link (2026-08-26): v4's root decision is
belief-aware (world sampling + opp knock model), but the gameplay INSIDE
rollout worlds is FastTeacher, whose defensive discarding only avoids cards
adjacent to the opponent's PUBLICLY KNOWN cards. This policy replaces that
safety key with the expected number of opponent-held meld-mates under the
calibrated Belief:

    safe(card) = sum over meld-adjacent a of P(opp holds a)

where P comes from `belief.marginals` (1.0 for known cards, so this strictly
generalizes the old key; hidden cards contribute their inferred
probability). Everything else about FastTeacher -- deadwood-first discard
discipline, draw gain rule, knock rule, the live-count primary key -- is
byte-for-byte the same, so any measured delta is attributable to the one
change. v5's Q-net attempted the same upgrade by learning and lost to its
own value noise (lesson #10); this is the zero-noise handcrafted version.

Usage: as a rollout hero policy inside q_expert.QRolloutExpert
(hero_mode="bt", per-lane cloned beliefs), or standalone via BTPolicy
(millisecond bot, own belief per hand).

    .venv/Scripts/python.exe belief_teacher.py --vs teacher --hands 1500 --seed 98000
"""

import argparse
import math
import multiprocessing as mp
import random
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DeclareDeadHandAction, DiscardAction, DrawCardAction, GinAction,
    KnockAction, PickUpDiscardAction)

import sim
from belief import _ADJ, Belief
from expert import RolloutExpert
from sim import (DEADWOOD_VALUE, DECK, FastTeacher, _live_count, apply_world,
                 cid, fast_legal_actions, ids_of, make_scratch,
                 min_deadwood_ids, playout, restore, snapshot, step_fast)


def clone_belief(b):
    """Cheap copy (same as q_expert.clone_belief; duplicated so the two
    modules stay import-independent)."""
    c = Belief.__new__(Belief)
    c.hero_id = b.hero_id
    c.alpha, c.beta, c.gamma = b.alpha, b.beta, b.gamma
    c.w = b.w.copy()
    c.opp_declined = b.opp_declined.copy()
    c.opp_discarded = b.opp_discarded.copy()
    c.hero_discarded = b.hero_discarded.copy()
    c.opp_pickups = b.opp_pickups
    c.opp_declines = b.opp_declines
    c._processed = b._processed
    c._pile = list(b._pile)
    return c


class BeliefTeacher(FastTeacher):
    """FastTeacher whose discard SAFETY key reads the belief marginals.

    `belief` must model the current player's view (belief.hero_id == the
    seat this policy plays); the caller owns it and may swap it per call
    (rollout lanes do). With belief=None and no source, behaves exactly
    like FastTeacher.

    `reset_from(source)` arms LAZY cloning for rollouts: the belief is
    cloned from `source` only when a discard tiebreak actually consults it
    -- most rollouts never do, so the clone cost mostly vanishes.
    """

    def __init__(self, belief=None, window=0, lam=2.0, mu=0.3, **kw):
        super().__init__(**kw)
        self.belief = belief
        self._source = None
        # window > 0: REAL defensive play -- consider discards within
        # `window` deadwood points of the minimum and choose by
        #   dw_after + lam * E[opp-held meld-mates] + mu * live_count
        # (deadwood-for-safety trades, the thing FastTeacher never does).
        # window == 0: pure tiebreak among min-deadwood discards (safe).
        self.window = window
        self.lam = lam
        self.mu = mu

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        rnd = game.round
        me = rnd.players[rnd.current_player_id]
        opp = rnd.players[1 - rnd.current_player_id]
        hand_ids = ids_of(me.hand)

        gin = [e for e in legal if isinstance(e, GinAction)]
        if gin:
            return gin[0]

        knocks = [e for e in legal if isinstance(e, KnockAction)]
        if knocks:
            best, best_dw = None, 999
            for event in knocks:
                drop = cid(event.card)
                dw = min_deadwood_ids([i for i in hand_ids if i != drop])
                if dw < best_dw:
                    best, best_dw = event, dw
            if best_dw <= self.knock_at or \
                    len(rnd.dealer.stock_pile) <= self.endgame_stock:
                return best
            legal = [e for e in legal if not isinstance(e, KnockAction)]
            if not legal:
                return best

        discards = [e for e in legal if isinstance(e, DiscardAction)]
        if discards:
            scored = []
            best_dw = 999
            for event in discards:
                drop = cid(event.card)
                dw = min_deadwood_ids([i for i in hand_ids if i != drop])
                scored.append((dw, event))
                best_dw = min(best_dw, dw)
            window = self.window
            pool = [(dw, e) for dw, e in scored
                    if dw <= best_dw + (window if self.belief is not None
                                        or self._source is not None else 0)]
            if len(pool) == 1:
                return pool[0][1]
            unseen_ids = ids_of(rnd.dealer.stock_pile)
            unseen_ids += [cid(c) for c in opp.hand
                           if c not in opp.known_cards]
            known_ids = ids_of(opp.known_cards)
            if self.belief is None and self._source is not None:
                self.belief = clone_belief(self._source)   # lazy arm
            if self.belief is None:      # fall back to FastTeacher exactly
                return min((e for dw, e in pool if dw == best_dw),
                           key=lambda e: (
                    self.w_live * _live_count(cid(e.card), unseen_ids),
                    self.w_safe * _live_count(cid(e.card), known_ids),
                    -self.w_value * DEADWOOD_VALUE[cid(e.card)]))
            self.belief.update(game)
            num_hidden = len(opp.hand) - len(opp.known_cards)
            p = self.belief.marginals(unseen_ids, num_hidden, known_ids)
            if window <= 0:              # pure tiebreak (conservative mode)
                return min((e for dw, e in pool if dw == best_dw),
                           key=lambda e: (
                    self.w_live * _live_count(cid(e.card), unseen_ids),
                    self.w_safe * float(p[_ADJ[cid(e.card)]].sum()),
                    -self.w_value * DEADWOOD_VALUE[cid(e.card)]))
            return min(pool, key=lambda t: (
                t[0] + self.lam * float(p[_ADJ[cid(t[1].card)]].sum())
                + self.mu * _live_count(cid(t[1].card), unseen_ids)))[1]

        pickup = [e for e in legal if isinstance(e, PickUpDiscardAction)]
        other = [e for e in legal
                 if isinstance(e, (DrawCardAction, DeclareDeadHandAction))]
        if pickup and other:
            if not rnd.dealer.discard_pile or len(me.hand) != 10:
                return other[0]
            top = cid(rnd.dealer.discard_pile[-1])
            candidate = hand_ids + [top]
            after = min(min_deadwood_ids([i for i in candidate if i != drop])
                        for drop in hand_ids)
            gain = min_deadwood_ids(hand_ids) - after
            return pickup[0] if gain >= self.draw_gain else other[0]

        return legal[0]


class BTPolicy:
    """Standalone engine-facing wrapper: own Belief per hand (token
    pattern)."""

    def __init__(self, seed=0, window=0, lam=2.0, mu=0.3):
        self.bt = BeliefTeacher(window=window, lam=lam, mu=mu)
        self._round_key = None

    def act(self, game, legal):
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self.bt.belief = Belief(rnd.current_player_id)
        return self.bt.act(game, legal)


class BTSerialExpert(RolloutExpert):
    """The corrected-v4 search loop (pristine-knowns world weights, see
    q_expert docstring) with the rollout HERO seat played by BeliefTeacher.

    Serial like v4 (one scratch, restore per world) -- no lane deepcopies,
    no batching; the belief is lazily cloned per playout only when a
    discard tiebreak consults it. With `bt_enabled=False` the hero rollout
    is exact FastTeacher -- the machinery-parity reference.
    """

    def __init__(self, bt_enabled=True, **kw):
        super().__init__(**kw)
        self._bt = BeliefTeacher() if bt_enabled else None

    def analyze(self, game, belief):
        from rlcard.games.gin_rummy.utils.action_event import (
            DiscardAction as _DA, KnockAction as _KA)
        legal = fast_legal_actions(game)
        candidates = self._candidates(game, legal)
        if len(candidates) == 1:
            return {"chosen": candidates[0], "reason": "forced",
                    "candidates": candidates}
        has_discard = any(isinstance(e, _DA) for e in legal)
        has_knock = any(isinstance(e, _KA) for e in legal)
        node = "discard" if has_discard or has_knock else "draw"
        if node not in self.classes:
            return {"chosen": self.teacher.act(game, legal),
                    "reason": "class-scoped", "candidates": candidates}
        hero_id = game.round.current_player_id
        opp_id = 1 - hero_id
        belief.update(game)
        boost = self.sample_boost
        if boost == "auto":
            boost = min(3.0, 0.5 * (belief.opp_pickups + belief.opp_declines))
        teacher_action = self.teacher.act(game, legal)
        t_idx = next((k for k, a in enumerate(candidates)
                      if a.action_id == teacher_action.action_id), None)
        if t_idx is None:
            candidates.insert(0, teacher_action)
            t_idx = 0
        hero_pol = self._bt if self._bt is not None \
            else self.rollout[hero_id]
        opp_pol = self._opp_rollout if self._opp_rollout is not None \
            else self.rollout[opp_id]
        rollout = {hero_id: hero_pol, opp_id: opp_pol}
        scratch = make_scratch(game)
        base = snapshot(scratch)
        opp = scratch.round.players[opp_id]
        unseen = ids_of(scratch.round.dealer.stock_pile) + \
            [cid(c) for c in opp.hand if c not in opp.known_cards]
        num_hidden = len(opp.hand) - len(opp.known_cards)
        known_ids = ids_of(opp.known_cards)      # pristine (quirk-corrected)
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
                free_ids = belief.sample_free_cards(unseen, num_hidden,
                                                    self.rng, boost)
            else:
                free_ids = []
            weight = self._world_weight(known_ids + list(free_ids),
                                        opp_discard_count)
            if weight <= 0.0:
                continue
            restore(scratch, base)
            apply_world(scratch, opp_id, [DECK[i] for i in free_ids],
                        self.rng)
            world = snapshot(scratch)
            for k, action in enumerate(candidates):
                restore(scratch, world)
                step_fast(scratch, action)
                if self._bt is not None:
                    self._bt.reset_from(belief)
                payoffs[worlds_done, k] = sign * playout(
                    scratch, rollout, truncate=self.truncate)
            weights[worlds_done] = weight
            worlds_done += 1
        if worlds_done == 0 or weights[:worlds_done].sum() <= 0.0:
            return {"chosen": teacher_action, "reason": "no-worlds",
                    "candidates": candidates}
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
        deviate = gain[best] > 0.0
        payoff_means = (w[:, None] * payoffs[:worlds_done]).sum(axis=0) / w_sum
        return {"chosen": candidates[best] if deviate else teacher_action,
                "reason": "deviate" if deviate else "prior",
                "candidates": candidates, "t_idx": t_idx,
                "payoff_means": payoff_means, "diff_means": means,
                "se": se, "gain": gain, "worlds": worlds_done,
                "n_eff": float(n_eff)}


def _btmatch_chunk(job):
    opp_name, hands, seat, seed, rules, window, lam, mu = job
    sim.set_rules(rules)
    import meta
    rng = random.Random(seed)
    pol_a = BTPolicy(seed=seed, window=window, lam=lam, mu=mu)
    pol_b = meta.population()[opp_name]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    return points


def btmatch(opp_name, hands, workers, seed, rules="full", window=0,
            lam=2.0, mu=0.3):
    per = max(1, hands // (2 * workers))
    jobs = [(opp_name, per, seat, seed + 31 * w + seat, rules, window, lam,
             mu) for w in range(workers) for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        points = np.concatenate(pool.map(_btmatch_chunk, jobs))
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    print(f"belief-teacher(w={window},lam={lam},mu={mu}) vs {opp_name} "
          f"[{rules}]: {mean:+.2f} +/- {ci:.2f} over {len(points)} hands "
          f"({time.time() - t0:.0f}s)", flush=True)
    return mean, ci, len(points)


def main():
    ap = argparse.ArgumentParser("belief-aware teacher gates")
    ap.add_argument("--vs", default="teacher")
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=98000)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--mu", type=float, default=0.3)
    args = ap.parse_args()
    btmatch(args.vs, args.hands, args.workers, args.seed, args.rules,
            args.window, args.lam, args.mu)


if __name__ == "__main__":
    main()
