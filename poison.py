"""
Adversarial probes of the v6 particle belief (v7 campaign, 2026-08-29).

The v6 filter's likelihood ASSUMES the opponent is a softened teacher:
discards are its least-useful cards, so everything it kept looks better
than what it threw. That assumption is the attack surface -- the
belief-poisoning adversary results/v6_prior_art.md section 3 calls for
as a promotion gate, never built until now.

`Advertiser` is FastTeacher with ONLY the discard tie-break replaced by
a false signal ("salting" in human gin): among the DEADWOOD-OPTIMAL
discards (fundamentals untouched -- same first criterion as the
teacher), throw the card that lies about our hand:

  mode="self"  maximize meld-adjacency to the cards we KEEP: every
               discard tells the hero's belief "I don't collect near
               this" about a zone where we DO collect, so the hero
               mis-prices exactly the discards we can use.
  mode="jam"   maximize live-count vs the unseen pool (the mu-energy
               term of SoftOpponent): each discard is maximally
               unlikely under the PF's model -- likelihood jamming.

What it forfeits: the teacher's live/safety tie-break (worth little by
lesson #12); what it buys: a poisoned posterior at trajectory level.

Gates, in order (cheap -> expensive):

    .venv/Scripts/python.exe poison.py --check           # fundamentals parity
    .venv/Scripts/python.exe poison.py --stress          # PF invariants
    .venv/Scripts/python.exe poison.py --worlds ad_self  # offline world quality
    .venv/Scripts/python.exe poison.py --match ad_self --hands 1488

The match prints v6's score (positive = v6 still wins). Compare with
teacher-vs-v6 on the SAME seed (--match teacher) -- the probe's teeth
are the DIFFERENCE, not the absolute number.
"""

import argparse
import random
import time

import numpy as np

import sim
from rlcard.games.gin_rummy.utils.action_event import (
    DeclareDeadHandAction, DiscardAction, DrawCardAction, GinAction,
    KnockAction, PickUpDiscardAction)
from sim import (FastTeacher, MixedKnock, _live_count, cid,
                 fast_legal_actions, ids_of, min_deadwood_ids, step_fast)


class Advertiser(FastTeacher):
    """FastTeacher, discard tie-break reversed into a false belief
    signal. Everything else (knock rule, pickup rule, gin) identical."""

    def __init__(self, mode="self", **kw):
        super().__init__(**kw)
        self.mode = mode

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
        if knocks:                       # teacher's knock rule, verbatim
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
            best_dw, best_events = 999, []
            for event in discards:
                drop = cid(event.card)
                dw = min_deadwood_ids([i for i in hand_ids if i != drop])
                if dw < best_dw:
                    best_dw, best_events = dw, [event]
                elif dw == best_dw:
                    best_events.append(event)
            if len(best_events) == 1:
                return best_events[0]
            # THE SALT: the teacher minimizes live/safety; we advertise.
            if self.mode == "self":
                return min(best_events, key=lambda e: (
                    -_live_count(cid(e.card),
                                 [i for i in hand_ids if i != cid(e.card)]),
                    cid(e.card)))
            unseen_ids = ids_of(rnd.dealer.stock_pile) + \
                [cid(c) for c in opp.hand if c not in opp.known_cards]
            return min(best_events, key=lambda e: (
                -_live_count(cid(e.card), unseen_ids), cid(e.card)))

        pickup = [e for e in legal if isinstance(e, PickUpDiscardAction)]
        other = [e for e in legal
                 if isinstance(e, (DrawCardAction, DeclareDeadHandAction))]
        if pickup and other:             # teacher's pickup rule, verbatim
            if not rnd.dealer.discard_pile or len(me.hand) != 10:
                return other[0]
            top = cid(rnd.dealer.discard_pile[-1])
            candidate = hand_ids + [top]
            after = min(min_deadwood_ids([i for i in candidate if i != drop])
                        for drop in hand_ids)
            gain = min_deadwood_ids(hand_ids) - after
            return pickup[0] if gain >= self.draw_gain else other[0]

        return legal[0]


# ---------------------------------------------------------------------------
# Gate 1: fundamentals parity -- the salt may only move WITHIN the
# deadwood-optimal discard set. Play teacher-vs-advertiser hands and
# assert dw-equality of the two choices at every advertiser discard node.
# ---------------------------------------------------------------------------

def check(hands=300, seed=71001):
    teacher = FastTeacher()
    diverged, nodes = 0, 0
    for mode in ("self", "jam"):
        adv = Advertiser(mode=mode)
        rng = random.Random(seed)
        d, n = 0, 0
        for _ in range(hands):
            game = sim.new_game(rng.randrange(1 << 30))
            rnd = game.round
            while rnd.going_out_action is None and not rnd.is_over:
                legal = fast_legal_actions(game)
                actor = adv if rnd.current_player_id == 0 else teacher
                a = actor.act(game, legal) if len(legal) > 1 else legal[0]
                if actor is adv and len(legal) > 1 and \
                        isinstance(a, DiscardAction):
                    t = teacher.act(game, legal)
                    if isinstance(t, DiscardAction):
                        hand = ids_of(rnd.players[0].hand)
                        dw_a = min_deadwood_ids(
                            [i for i in hand if i != cid(a.card)])
                        dw_t = min_deadwood_ids(
                            [i for i in hand if i != cid(t.card)])
                        assert dw_a == dw_t, \
                            f"salt broke deadwood: {dw_a} vs {dw_t}"
                        n += 1
                        d += a.action_id != t.action_id
                step_fast(game, a)
        print(f"check[{mode}]: {n} discard nodes, salt differs from "
              f"teacher on {d} ({d / max(n, 1):.1%}), deadwood parity "
              f"PASS", flush=True)
        diverged += d
        nodes += n
    return diverged, nodes


# ---------------------------------------------------------------------------
# Gate 2: particle-filter invariant stress. Attach a ParticleBelief to
# seat 0's information set while assorted opponents (incl. the poisoner
# and the home variant) drive real games; assert the filter's public
# bookkeeping NEVER diverges from ground truth.
# ---------------------------------------------------------------------------

def _stress_policies(name, seed):
    if name == "teacher":
        return FastTeacher()
    if name == "knock10":
        return MixedKnock(FastTeacher(), 10, 0.5, rng=random.Random(seed))
    if name == "random":
        class R:
            def __init__(self, rng):
                self.rng = rng

            def act(self, game, legal):
                return legal[self.rng.randrange(len(legal))]
        return R(random.Random(seed + 1))
    if name == "ad_self":
        return Advertiser(mode="self")
    raise ValueError(name)


def stress(hands=150, seed=71501):
    from pf_expert import ParticleBelief
    t0 = time.time()
    checked = 0
    for opp_name in ("teacher", "knock10", "random", "ad_self"):
        for home in (False, True):
            hero_pol = FastTeacher()
            opp_pol = _stress_policies(opp_name, seed)
            rng = random.Random(seed + (997 if home else 0))
            for h in range(hands // 2):
                game = (sim.new_game_home if home else sim.new_game)(
                    rng.randrange(1 << 30))
                rnd = game.round
                belief = None
                while rnd.going_out_action is None and not rnd.is_over:
                    legal = fast_legal_actions(game)
                    if rnd.current_player_id == 0 and len(legal) > 1:
                        if belief is None:
                            belief = ParticleBelief(0, seed=seed + h)
                        belief.update(game)
                        core = belief.core
                        opp = rnd.players[1]
                        hidden = {cid(c) for c in opp.hand
                                  if c not in opp.known_cards}
                        hero_ids = set(ids_of(rnd.players[0].hand))
                        pile_ids = [cid(c) for c in
                                    rnd.dealer.discard_pile]
                        ctx = f"{opp_name}/home={home}/hand={h}"
                        assert belief.desyncs == 0, f"desync {ctx}"
                        assert core.h == len(hidden), \
                            f"h={core.h} != {len(hidden)} {ctx}"
                        assert core.hero_hand == hero_ids, f"hero {ctx}"
                        assert core.pile == pile_ids, f"pile {ctx}"
                        assert core.opp_known <= set(ids_of(opp.hand)), \
                            f"known {ctx}"
                        assert np.isfinite(core.logw).all(), f"logw {ctx}"
                        for p in core.particles:
                            assert len(p) == core.h, f"psize {ctx}"
                            assert not (p & hero_ids), f"p-hero {ctx}"
                            assert not (p & set(pile_ids)), f"p-pile {ctx}"
                            assert not (p & core.opp_known), \
                                f"p-known {ctx}"
                        checked += 1
                    a = legal[0] if len(legal) == 1 else \
                        (hero_pol if rnd.current_player_id == 0
                         else opp_pol).act(game, legal)
                    step_fast(game, a)
    print(f"stress: {checked} decision-node invariant checks over "
          f"4 opponents x (standard, home), ALL PASS "
          f"({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Gate 3: offline world quality on poisoned streams (particle.worlds_gate
# with the advertiser as the stream opponent). The searcher consumes
# WORLDS; if their overlap/realism/knock-Brier survive the salt, the
# poison has no mechanism to bite through.
# ---------------------------------------------------------------------------

def worlds(mode, hands=150, seed=71700):
    from particle import worlds_gate
    print(f"--- worlds gate, opponent = advertiser[{mode}] ---", flush=True)
    worlds_gate(hands, seed, opponent=lambda: Advertiser(mode=mode))
    print(f"--- worlds gate, opponent = teacher (reference) ---", flush=True)
    worlds_gate(hands, seed, opponent="teacher")


# ---------------------------------------------------------------------------
# Gate 4: the match. v6 vs the poisoner (pfmatch harness; positive =
# v6 wins). Run `--match teacher` on the same seed for the baseline.
# ---------------------------------------------------------------------------

def match(opp, hands, workers, seed):
    from pf_expert import pfmatch
    return pfmatch(opp, hands, workers, seed, rules="full")


def main():
    ap = argparse.ArgumentParser("belief-poisoning probes")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--worlds", default=None, metavar="MODE")
    ap.add_argument("--match", default=None, metavar="OPP")
    ap.add_argument("--hands", type=int, default=1488)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=71100)
    args = ap.parse_args()
    sim.set_rules("full")
    if args.check:
        check()
    if args.stress:
        stress()
    if args.worlds:
        worlds(args.worlds.replace("ad_", ""))
    if args.match:
        match(args.match, args.hands, args.workers, args.seed)


if __name__ == "__main__":
    main()
