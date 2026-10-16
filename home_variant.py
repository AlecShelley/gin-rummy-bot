"""
The home variant: 10+10 deal, one flipped card founds the pile, and the
hand's knock threshold is the flip's deadwood value (sim.new_game_home).

Two jobs here:

1. --stats: confirm the variant behaves as expected. Both seats play
   knock-whenever-legal (MixedKnock t=10 covers every legal knock); results
   are binned by flip value. Prediction: ace flips -> long hands, gin-heavy;
   ten flips -> short hands, knock-heavy.

2. --match: seat-swapped strength runs. `v4home` is the v4 search with the
   opponent knock model set PER HAND to the flip value (in this variant the
   threshold is public, so the model is exact rather than assumed).

    .venv/Scripts/python.exe home_variant.py --stats --hands 30000
    .venv/Scripts/python.exe home_variant.py --match --vs knocker --hands 1000
"""

import argparse
import math
import multiprocessing as mp
import random
import time
from collections import defaultdict

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import GinAction, KnockAction

import sim
from belief import Belief
from expert import RolloutExpert
from sim import FastTeacher, MixedKnock, fast_legal_actions, playout, step_fast


# ---------------------------------------------------------------------------
# Behavior statistics.
# ---------------------------------------------------------------------------

def _stats_chunk(job):
    hands, seed = job
    sim.set_rules("full")
    rng = random.Random(seed)
    rows = []
    for _ in range(hands):
        game = sim.new_game_home(rng.randrange(1 << 30))
        flip = game.settings.going_out_deadwood_count
        policies = {0: MixedKnock(FastTeacher(), 10, 1.0,
                                  rng=random.Random(rng.randrange(1 << 30))),
                    1: MixedKnock(FastTeacher(), 10, 1.0,
                                  rng=random.Random(rng.randrange(1 << 30)))}
        rnd = game.round
        steps = 0
        while rnd.going_out_action is None and not rnd.is_over and steps < 250:
            legal = fast_legal_actions(game)
            action = legal[0] if len(legal) == 1 else \
                policies[rnd.current_player_id].act(game, legal)
            step_fast(game, action)
            steps += 1
        action = rnd.going_out_action
        if isinstance(action, GinAction):
            kind = "gin"
        elif isinstance(action, KnockAction):
            kind = "knock"
        else:
            kind = "dead"
        points = abs(sim._payoff(game)) * 100.0
        rows.append((flip, steps, kind, points))
    return rows


def run_stats(hands=30000, workers=12, seed=61000):
    per = max(1, hands // workers)
    jobs = [(per, seed + 97 * w) for w in range(workers)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = [r for chunk in pool.map(_stats_chunk, jobs) for r in chunk]
    bins = defaultdict(list)
    for flip, steps, kind, points in rows:
        bins[flip].append((steps, kind, points))
    print(f"{len(rows)} hands, knock-whenever-legal both seats, full rules "
          f"({time.time() - t0:.0f}s)")
    print("flip | hands | avg len | gin% | knock% | dead% | avg pts")
    for flip in sorted(bins):
        b = bins[flip]
        n = len(b)
        avg_len = np.mean([s for s, _, _ in b])
        gin = 100.0 * sum(1 for _, k, _ in b if k == "gin") / n
        knock = 100.0 * sum(1 for _, k, _ in b if k == "knock") / n
        dead = 100.0 * sum(1 for _, k, _ in b if k == "dead") / n
        pts = np.mean([p for _, _, p in b])
        print(f"{flip:>4} | {n:>5} | {avg_len:7.1f} | {gin:4.1f} | "
              f"{knock:6.1f} | {dead:5.1f} | {pts:7.1f}")


# ---------------------------------------------------------------------------
# Strength matches: v4home vs teacher / knock-whenever-legal.
# ---------------------------------------------------------------------------

# The promoted champion's NAME -- update on every promotion (the
# evaluator stamps its reports with it).
CHAMPION = "v10home"


class HomeV8Policy:
    """v8home -- THE HOME-VARIANT CHAMPION (named 2026-09-02).

    v7's search + particle filter (knock model bound per hand to the
    public flip threshold, in both the rollout continuation and the
    filter's no-knock evidence), plus the home-only upgrades gated in
    the v8 campaign (results/v8_gates.log):
      * the real OPENING RULE (P1 option -> P2 option -> forced draw),
      * the seat-asymmetric OPENING BOOK (P1 takes iff melds, z=4.9;
        P2 also takes with adjacency >= 2, z=4.0) played directly,
      * option-aware opening likelihoods in the particle filter.
    Fresh expert per hand, the HomeV4Policy pattern. book/solver flags
    reproduce the ablations (solver measured -0.53 +/- 0.62 vs the
    book -- not adopted)."""

    def __init__(self, seed=0, book=True, solver=False, t2book=True,
                 t2x=True, pf_cfg=None):
        self.seed = seed
        # pf_cfg (v10 campaign, 2026-09-06): optional DEFAULT_PF
        # overrides for the particle filter (e.g. dict(tau_po=0.0,
        # eps=0.02) = the sharpened-likelihood candidate). None =
        # champion DEFAULT_PF unchanged.
        self.pf_cfg = pf_cfg
        # Opening handling at flip-option nodes (the standard search's
        # evidence there is the MEASURED bias -- truncated deadwood
        # leaf vs gin races):
        #   solver=True : per-hand CALCULATION -- full-length paired
        #     playouts, book as the z-tested prior
        #     (opening_solver.solve_opening; validated cell-by-cell
        #     against the 6000-deal ground truth).
        #   book=True   : the opening book rule directly.
        #   both False  : the pre-book searcher decides.
        self.book = book
        self.solver = solver
        # t2book (turn-2 clause, 2026-09-02): at the hand's FIRST free
        # draw decision after the opening, take iff the card melds --
        # the bot's own non-melding takes there measured -2.06 +/-
        # 1.64/take on 2400 forced-arm cell-deals
        # (results/turn2_confirm.log); melding takes it already makes
        # 100% of the time (+9.5). Later turns: unmeasured, search
        # decides.
        self.t2book = t2book
        # t2x: the user's exception in its naturally-conditioned form
        # (nat2_study.py): also take a non-melding LOW card (<=3) with
        # adj>=2 -- +3.58 +/- 2.41 under the true conditional; fires
        # ~1/15000 hands (no match gate can see it; adopted on the
        # causal evidence).
        self.t2x = t2x
        self._nrng = None
        self._round_key = None
        self._belief = None
        self._expert = None
        self._n = 0
        self._t2_used = False

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:      # new hand
            self._round_key = rnd._v3_token
            self._n += 1
            self._belief = None
            self._expert = None
            self._t2_used = False
        opening = getattr(rnd, "_home_opening", 0)
        if self.solver and opening:
            import numpy as np
            from opening_solver import solve_opening
            if self._nrng is None:
                self._nrng = np.random.default_rng(self.seed + 991)
            action, _, _, _ = solve_opening(game, self._nrng)
            return action
        if self.book and opening:
            from rlcard.games.gin_rummy.utils.action_event import (
                DrawCardAction, PickUpDiscardAction)
            from sim import cid, flip_book_take, ids_of
            hand = ids_of(rnd.players[rnd.current_player_id].hand)
            top = cid(rnd.dealer.discard_pile[-1])
            take = flip_book_take(hand, top, opening)
            want = PickUpDiscardAction if take else DrawCardAction
            for a in legal:
                if isinstance(a, want):
                    return a
        # Turn-2 clause: first free (non-opening) draw decision of the
        # hand -- take iff the card melds. See __init__ note.
        if self.t2book and not opening and not self._t2_used and \
                hasattr(rnd, "_home_flip_id"):
            kinds = {type(a).__name__ for a in legal}
            if kinds == {"DrawCardAction", "PickUpDiscardAction"}:
                from rlcard.games.gin_rummy.utils.action_event import (
                    DrawCardAction, PickUpDiscardAction)
                from sim import cid, ids_of, t2_book_take
                self._t2_used = True
                hand = ids_of(rnd.players[rnd.current_player_id].hand)
                top = cid(rnd.dealer.discard_pile[-1])
                take = t2_book_take(hand, top, exception=self.t2x)
                want = PickUpDiscardAction if take else DrawCardAction
                for a in legal:
                    if isinstance(a, want):
                        return a
        if self._expert is None:
            from pf_expert import ParticleBelief, PFExpert, V5_CFG
            threshold = game.settings.going_out_deadwood_count
            self._belief = ParticleBelief(
                rnd.current_player_id, cfg=self.pf_cfg,
                seed=self.seed * 100003 + self._n)
            self._expert = PFExpert(
                veto=True, opp_model=(threshold, 1.0), sample_boost=0.0,
                seed=self.seed + self._n, **V5_CFG)
        return self._expert.decide(game, self._belief)


HomeV6Policy = HomeV8Policy    # backward-compat alias (pre-v8 name);
#                                the harnesses below still import it


class HomeV4Policy:
    """v4 (K=48, T=16, z=1.5, boost=2.5) with the knock model bound to the
    hand's public threshold. A fresh RolloutExpert per hand (construction is
    cheap; the threshold changes every deal)."""

    def __init__(self, seed=0):
        self.seed = seed
        self._round_key = None
        self._belief = None
        self._expert = None
        self._n = 0

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._belief = Belief(rnd.current_player_id)
            self._n += 1
            threshold = game.settings.going_out_deadwood_count
            self._expert = RolloutExpert(
                worlds=48, truncate=16, z=1.5,
                opp_model=(threshold, 1.0), sample_boost=2.5,
                seed=self.seed + self._n)
        return self._expert.decide(game, self._belief)


def _opponent(kind, rng, seed=0):
    if kind == "teacher":
        return FastTeacher()
    if kind == "knocker":
        return MixedKnock(FastTeacher(), 10, 1.0,
                          rng=random.Random(rng.randrange(1 << 30)))
    if kind == "v4home":
        return HomeV4Policy(seed=seed + 7)
    raise ValueError(kind)


def _match_chunk(job):
    kind, hands, seat, seed, bot = job
    sim.set_rules("full")
    rng = random.Random(seed)
    # v10home = the current champion (new DEFAULT_PF after the v10
    # likelihood promotion, 2026-09-07); "v8home" pins the pre-v10
    # belief exactly as it was gated (results/v10_gates.log).
    if bot == "v4home":
        pol_a = HomeV4Policy(seed=seed)
    elif bot == "v8home":
        pol_a = HomeV8Policy(seed=seed,
                             pf_cfg=dict(eps=0.05, tau_po=None))
    else:                                # v10home / v6home alias
        pol_a = HomeV8Policy(seed=seed)
    pol_b = _opponent(kind, rng, seed)
    out = []
    for _ in range(hands):
        game = sim.new_game_home(rng.randrange(1 << 30))
        flip = game.settings.going_out_deadwood_count
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        out.append((flip, (payoff0 if seat == 0 else -payoff0) * 100.0))
    return out


def run_match(kind, hands=1000, workers=12, seed=62000, bot="v4home"):
    per = max(1, hands // (2 * workers))
    jobs = [(kind, per, seat, seed + 31 * w + seat, bot)
            for w in range(workers) for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = [r for chunk in pool.map(_match_chunk, jobs) for r in chunk]
    pts = np.array([p for _, p in rows])
    mean = float(pts.mean())
    ci = 1.96 * float(pts.std(ddof=1)) / math.sqrt(len(pts))
    print(f"{bot} vs {kind} [home variant, full rules]: {mean:+.2f} +/- "
          f"{ci:.2f} over {len(pts)} hands ({time.time() - t0:.0f}s)")
    lo = [p for f, p in rows if f <= 3]
    hi = [p for f, p in rows if f >= 8]
    if lo and hi:
        print(f"  flip<=3 (gin-ish games): {np.mean(lo):+.2f} "
              f"+/- {1.96 * np.std(lo, ddof=1) / math.sqrt(len(lo)):.2f} "
              f"({len(lo)})")
        print(f"  flip>=8 (knock games):   {np.mean(hi):+.2f} "
              f"+/- {1.96 * np.std(hi, ddof=1) / math.sqrt(len(hi)):.2f} "
              f"({len(hi)})")
    return mean, ci


def test_opening(hands=300, seed=78000):
    """Gate for the home OPENING rule (user decision 2026-09-01):
    P1 flip option -> P2 flip option -> forced P1 stock draw. Scripted
    invariants + bulk teacher playouts + one searched self-play hand
    with particle-belief consistency."""
    from rlcard.games.gin_rummy.utils.action_event import (
        DiscardAction, DrawCardAction, KnockAction, PickUpDiscardAction)
    from sim import HomePassMove, restore, snapshot
    sim.set_rules("full")

    def opt(game, want):
        legal = fast_legal_actions(game)
        return next(a for a in legal if isinstance(a, want))

    # -- scripted sequence 1: P1 passes, P2 takes ------------------------
    g = sim.new_game_home(12345)
    rnd = g.round
    p1 = rnd.current_player_id
    assert rnd._home_opening == 1
    legal = fast_legal_actions(g)
    assert {type(a) for a in legal} == {DrawCardAction,
                                        PickUpDiscardAction}
    step_fast(g, opt(g, DrawCardAction))          # P1 passes
    assert rnd._home_opening == 2
    assert rnd.current_player_id == 1 - p1
    assert len(rnd.players[p1].hand) == 10        # nothing drawn
    assert isinstance(rnd.move_sheet[-1], HomePassMove)
    assert rnd.move_sheet[-1].player.player_id == p1
    step_fast(g, opt(g, PickUpDiscardAction))     # P2 takes the flip
    assert rnd._home_opening == 0
    assert len(rnd.players[1 - p1].hand) == 11
    assert not rnd.dealer.discard_pile
    legal = fast_legal_actions(g)                 # P2 must discard now
    assert all(isinstance(a, (DiscardAction, KnockAction))
               for a in legal)
    step_fast(g, next(a for a in legal
                      if isinstance(a, DiscardAction)))
    assert rnd.current_player_id == p1            # play returns to P1

    # -- scripted sequence 2: both pass -> forced P1 draw ----------------
    g = sim.new_game_home(12345)
    rnd = g.round
    step_fast(g, opt(g, DrawCardAction))          # P1 passes
    step_fast(g, opt(g, DrawCardAction))          # P2 passes
    assert rnd._home_opening == 0
    assert rnd.current_player_id == p1
    legal = fast_legal_actions(g)
    assert len(legal) == 1 and isinstance(legal[0], DrawCardAction)
    step_fast(g, legal[0])                        # forced stock draw
    assert len(rnd.players[p1].hand) == 11
    assert len(rnd.dealer.discard_pile) == 1      # flip stays buried

    # -- snapshot/restore across the opening -----------------------------
    g = sim.new_game_home(999)
    rnd = g.round
    snap = snapshot(g)
    n0 = len(rnd.move_sheet)
    step_fast(g, opt(g, DrawCardAction))
    restore(g, snap)
    assert rnd._home_opening == 1
    assert len(rnd.move_sheet) == n0
    assert fast_legal_actions(g) and len(fast_legal_actions(g)) == 2

    # -- bulk: teacher self-play terminates, cards conserved, and the
    # teacher follows the OPENING BOOK (take iff the flip melds) -------
    from sim import cid, flip_book_take, ids_of, min_deadwood_ids
    rng = random.Random(seed)
    kinds = {"take1": 0, "take2": 0, "bothpass": 0}
    for _ in range(hands):
        g = sim.new_game_home(rng.randrange(1 << 30))
        rnd = g.round
        p1_hand = ids_of(rnd.players[rnd.current_player_id].hand)
        p2_hand = ids_of(rnd.players[1 - rnd.current_player_id].hand)
        flip_id = rnd._home_flip_id
        p1_melds = flip_book_take(p1_hand, flip_id, 1)
        p2_book = flip_book_take(p2_hand, flip_id, 2)
        pols = {0: MixedKnock(FastTeacher(), 10, 1.0,
                              rng=random.Random(rng.randrange(1 << 30))),
                1: MixedKnock(FastTeacher(), 10, 1.0,
                              rng=random.Random(rng.randrange(1 << 30)))}
        steps = 0
        opening_moves = []
        while rnd.going_out_action is None and not rnd.is_over \
                and steps < 250:
            legal = fast_legal_actions(g)
            action = legal[0] if len(legal) == 1 else \
                pols[rnd.current_player_id].act(g, legal)
            if len(opening_moves) < 2 and steps < 2:
                opening_moves.append(type(action).__name__)
            step_fast(g, action)
            steps += 1
        n_cards = sum(len(p.hand) for p in rnd.players) + \
            len(rnd.dealer.stock_pile) + len(rnd.dealer.discard_pile)
        # a knock/gin discard never joins the pile (engine fact), so a
        # knocked/ginned hand shows 51 cards; dead/unfinished shows 52
        want = 51 if isinstance(rnd.going_out_action,
                                (KnockAction, GinAction)) else 52
        assert n_cards == want, f"card leak: {n_cards} != {want}"
        if opening_moves[0] == "PickUpDiscardAction":
            kinds["take1"] += 1
        elif len(opening_moves) > 1 and \
                opening_moves[1] == "PickUpDiscardAction":
            kinds["take2"] += 1
        else:
            kinds["bothpass"] += 1
        # the teacher's opening choices must equal the book (P1:
        # meld-only; P2: melds OR adjacency >= 2)
        took1 = opening_moves[0] == "PickUpDiscardAction"
        assert took1 == p1_melds, \
            f"teacher P1 opening violates the book (book={p1_melds})"
        if not took1 and len(opening_moves) > 1:
            took2 = opening_moves[1] == "PickUpDiscardAction"
            assert took2 == p2_book, \
                f"teacher P2 opening violates the book (book={p2_book})"
    assert all(v > 0 for v in kinds.values()), kinds

    # -- one searched self-play hand: PF belief stays consistent ---------
    g = sim.new_game_home(777)
    rnd = g.round
    p1 = rnd.current_player_id
    pols = {p1: HomeV8Policy(seed=1), 1 - p1: HomeV8Policy(seed=2)}
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < 250:
        legal = fast_legal_actions(g)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(g, legal)
        step_fast(g, action)
        steps += 1
    for s, pol in pols.items():
        b = pol._belief
        assert b is None or b.desyncs == 0, f"seat {s} belief desync"
    print(f"opening tests: PASS ({hands} teacher hands; opening mix "
          f"P1-take {kinds['take1']}, P2-take {kinds['take2']}, "
          f"both-pass {kinds['bothpass']}; searched hand OK)")


def main():
    ap = argparse.ArgumentParser("home variant harness")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--test-opening", dest="test_opening",
                    action="store_true")
    ap.add_argument("--vs", default="knocker",
                    choices=("teacher", "knocker", "v4home"))
    ap.add_argument("--bot", default="v10home",
                    choices=("v4home", "v6home", "v8home", "v10home"))
    ap.add_argument("--hands", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=61000)
    args = ap.parse_args()
    if args.test_opening:
        test_opening(min(args.hands, 300), args.seed + 17000)
        return
    if args.stats:
        run_stats(args.hands, args.workers, args.seed)
    if args.match:
        run_match(args.vs, args.hands, args.workers, args.seed + 1000,
                  args.bot)


if __name__ == "__main__":
    main()
