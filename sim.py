"""
Fast mid-state gin rummy simulation kernel.

This is the substrate for the v3 belief-search bot: everything above it
(determinized rollouts, the ExIt expert, exploitability measurements) needs to
play thousands of hand-continuations per second. The rlcard Env pays for a full
observation build plus two legal-action computations per step; worse, every
Card comparison routes through Card.__hash__/__eq__, which call list.index().
Profiling showed 27M list.index calls in a 600-playout benchmark.

So this module drives the engine in integer card-id space:

  * `min_deadwood_ids` is a from-scratch meld solver (DFS over disjoint melds)
    equal by construction to rlcard's cluster enumeration for hands <= 11 cards
    (>= 4 disjoint melds are impossible), verified against shaping._min_deadwood.
  * `fast_legal_actions` replicates GinRummyJudge.get_legal_actions, using a
    cheap deadwood gate to skip the expensive going-out computation on the ~90%
    of discard nodes where no knock or gin is possible; when the gate opens it
    calls rlcard's real `get_going_out_cards`, so card lists and their ORDER
    (which feeds tie-breaks) are exact. Verified action-for-action, in order.
  * `make_scratch` deep-copies a game and strips the per-player meld-cache
    upkeep, which only the bypassed judge ever reads.

Three engine facts make snapshot/restore safe (verified against rlcard 1.2.0):
play is deterministic after the deal (stock order fixed, no randomness at play
time); the turn phase lives in `game.actions[-1]` plus `move_sheet[-1]`, so
truncating both restores it; nothing else caches cards.

`FastTeacher` is an exact port of `teacher.StrongTeacher`. Exactness matters:
every gate downstream is measured against the teacher, so `--verify` proves
100% action agreement (via the real judge) plus legal-action and deadwood
equivalence before anything builds on this file.

    python sim.py --verify   # teacher agreement, legal parity, dw parity, payoff parity
    python sim.py --bench    # playout throughput, target >=500/s/core
    python sim.py --stress   # 10k noisy playouts terminate, cards conserved
"""

import argparse
import copy
import random
import time
import types

from rlcard.games.gin_rummy.game import GinRummyGame
from rlcard.games.gin_rummy.utils import utils as gr_utils
from rlcard.games.gin_rummy.utils.action_event import (
    DeclareDeadHandAction, DiscardAction, DrawCardAction, GinAction,
    KnockAction, PickUpDiscardAction, ScoreNorthPlayerAction,
    ScoreSouthPlayerAction)

# ---------------------------------------------------------------------------
# Card tables. id = rank_id + 13 * suit_id, ascending ids = suit-major with
# ranks ascending inside a suit (so runs are consecutive ids within a suit).
# The string key avoids Card.__hash__, which costs two list.index calls.
# ---------------------------------------------------------------------------

DECK = gr_utils.get_deck()                      # singletons, index == card_id
CARD_ID = {str(card): cid for cid, card in enumerate(DECK)}   # str = rank+suit
DEADWOOD_VALUE = [gr_utils.get_deadwood_value(card) for card in DECK]


def flip_book_take(hand_ids, top_id, option):
    """The OPENING BOOK (6000-deal study + paired gates, 2026-09-02):
    should the player holding flip `option` (1 = first player's, 2 =
    second player's) take it? First player: iff it melds. Second
    player: also with adjacency >= 2 -- their take is an extra turn,
    so speculation pays (+3.05 +/- 1.50 paired; +9.1 in A-3 games)."""
    hand_ids = list(hand_ids)
    if min_deadwood_ids(hand_ids + [top_id]) <= \
            min_deadwood_ids(hand_ids):
        return True
    if option != 2:
        return False
    fr, fs = top_id % 13, top_id // 13
    adj = sum(1 for c in hand_ids
              if c % 13 == fr or (c // 13 == fs
                                  and abs(c % 13 - fr) <= 2))
    return adj >= 2


def t2_book_take(hand_ids, top_id, exception=True):
    """The TURN-2 book (first free draw decision): take iff the card
    melds -- plus the user's exception in its NATURALLY-CONDITIONED
    form (nat2_study.py, 2026-09-03): a non-melding take of an
    opponent-discarded LOW card (value <= 3) with adjacency >= 2 is
    +3.58 +/- 2.41 under the true conditional (opponent CHOSE the
    discard; 430 rejection-sampled states), strongest for weak hands
    (outs 0-2: +11.0 +/- 7.8). The earlier broad exception (any card
    value, dead-10 shed) gated at a tie -- the value is specific to
    LOW cards; mid/high speculative takes stay excluded. Fires ~1 in
    15,000 hands: immaterial to match strength, but it is the theory."""
    hand_ids = list(hand_ids)
    before = min_deadwood_ids(hand_ids)
    if min_deadwood_ids(hand_ids + [top_id]) <= before:
        return True
    if not exception or DEADWOOD_VALUE[top_id] > 3:
        return False
    fr, fs = top_id % 13, top_id // 13
    adj = sum(1 for c in hand_ids
              if c % 13 == fr or (c // 13 == fs
                                  and abs(c % 13 - fr) <= 2))
    return adj >= 2


class HomePassAction:
    """Home-variant OPENING pass (user rule, 2026-09-01): declining the
    flip. Moves no cards; exists so game.actions carries the phase."""

    def __repr__(self):
        return "HomePassAction()"


class HomePassMove:
    """move_sheet record of an opening pass -- public decline evidence
    on the flip. Only .player is information; no card moved."""

    __slots__ = ("player",)

    def __init__(self, player):
        self.player = player


def cid(card):
    return CARD_ID[card.rank + card.suit]


def ids_of(cards):
    return [CARD_ID[c.rank + c.suit] for c in cards]


# ---------------------------------------------------------------------------
# Deadwood: DFS over disjoint melds, cached on sorted id tuples.
# ---------------------------------------------------------------------------

_DW_CACHE = {}
_DW_CACHE_LIMIT = 2_000_000
_GO_CACHE = {}                 # sorted hand ids -> (knock card ids, gin possible)
_GO_CACHE_LIMIT = 300_000


def _melds(ids_sorted):
    """All candidate melds as (position_bitmask, deadwood_sum).

    Runs: every sub-run of length >=3 of each maximal same-suit consecutive
    sequence. Sets: each rank group of >=3, plus all 3-subsets of a 4-group.
    The same meld space rlcard's melding.py enumerates.
    """
    n = len(ids_sorted)
    melds = []
    # runs -- consecutive ids within one suit
    i = 0
    while i < n - 2:
        j = i + 1
        while j < n and ids_sorted[j] == ids_sorted[j - 1] + 1 \
                and ids_sorted[j] // 13 == ids_sorted[i] // 13:
            j += 1
        seg_len = j - i
        if seg_len >= 3:
            for a in range(i, j - 2):
                for b in range(a + 3, j + 1):
                    mask = 0
                    value = 0
                    for p in range(a, b):
                        mask |= 1 << p
                        value += DEADWOOD_VALUE[ids_sorted[p]]
                    melds.append((mask, value))
        i = j
    # sets -- same rank across suits
    by_rank = {}
    for p, card_id in enumerate(ids_sorted):
        by_rank.setdefault(card_id % 13, []).append(p)
    for positions in by_rank.values():
        if len(positions) >= 3:
            mask = 0
            value = 0
            for p in positions:
                mask |= 1 << p
                value += DEADWOOD_VALUE[ids_sorted[p]]
            melds.append((mask, value))
            if len(positions) == 4:
                for p in positions:
                    melds.append((mask & ~(1 << p),
                                  value - DEADWOOD_VALUE[ids_sorted[p]]))
    return melds


def min_deadwood_ids(ids_iterable):
    """Deadwood under the best melding, for a hand given as card ids."""
    key = tuple(sorted(ids_iterable))
    hit = _DW_CACHE.get(key)
    if hit is not None:
        return hit
    total = sum(DEADWOOD_VALUE[i] for i in key)
    melds = _melds(key)
    best = total

    def dfs(start, used, removed):
        nonlocal best
        if total - removed < best:
            best = total - removed
        for k in range(start, len(melds)):
            mask, value = melds[k]
            if not (mask & used):
                dfs(k + 1, used | mask, removed + value)

    if melds:
        dfs(0, 0, 0)
    if len(_DW_CACHE) < _DW_CACHE_LIMIT:
        _DW_CACHE[key] = best
    return best


def min_deadwood(cards):
    """Card-object convenience wrapper."""
    return min_deadwood_ids(ids_of(cards))


def _going_out_ids(key, threshold):
    """Knock card ids and gin availability for an 11-card hand (sorted ids).

    Replicates judge._get_going_out_cards exactly, in id space: iterate every
    nonempty disjoint meld subset (rlcard's "clusters"; >= 4 disjoint melds are
    impossible with 11 cards); a subset melding everything, or all but one
    card, allows gin; otherwise, subject to rlcard's quirky pre-filter
    `deadwood <= 10 + max_value` (the 10 is hardcoded there, NOT the setting),
    every deadwood card whose removal brings deadwood within `threshold` is a
    knock card.
    """
    melds = _melds(key)
    if not melds:
        return (), False
    total = sum(DEADWOOD_VALUE[i] for i in key)
    n = len(key)
    knock_ids = set()
    has_gin = False

    def dfs(start, used, removed):
        nonlocal has_gin
        if used:
            count = total - removed
            if count == 0:
                has_gin = True           # full cover always has a 4+ pile
            else:
                free = [p for p in range(n) if not (used >> p) & 1]
                if len(free) == 1:
                    has_gin = True
                else:
                    values = [DEADWOOD_VALUE[key[p]] for p in free]
                    if count <= 10 + max(values):
                        for p, value in zip(free, values):
                            if count - value <= threshold:
                                knock_ids.add(key[p])
        for k in range(start, len(melds)):
            mask, value = melds[k]
            if not (mask & used):
                dfs(k + 1, used | mask, removed + value)

    dfs(0, 0, 0)
    return tuple(sorted(knock_ids)), has_gin


def best_discard_deadwood(hand11):
    """(best_card, deadwood) over all single discards from an 11-card hand."""
    hand_ids = ids_of(hand11)
    best_card, best_dw = None, 999
    for k, card in enumerate(hand11):
        dw = min_deadwood_ids(hand_ids[:k] + hand_ids[k + 1:])
        if dw < best_dw:
            best_card, best_dw = card, dw
    return best_card, best_dw


# ---------------------------------------------------------------------------
# Snapshot / restore / determinize.
# ---------------------------------------------------------------------------

class Snap:
    """A restorable point-in-time copy of a game's mutable state.

    Cards are immutable and value-equal, so shallow list copies suffice.
    """

    __slots__ = ("stock", "discard", "hands", "knowns", "cur_id",
                 "going_out_action", "going_out_id", "is_over",
                 "n_moves", "n_actions", "home_opening")

    def __init__(self, game):
        rnd = game.round
        self.home_opening = getattr(rnd, "_home_opening", 0)
        self.stock = list(rnd.dealer.stock_pile)
        self.discard = list(rnd.dealer.discard_pile)
        self.hands = (list(rnd.players[0].hand), list(rnd.players[1].hand))
        self.knowns = (list(rnd.players[0].known_cards),
                       list(rnd.players[1].known_cards))
        self.cur_id = rnd.current_player_id
        self.going_out_action = rnd.going_out_action
        self.going_out_id = rnd.going_out_player_id
        self.is_over = rnd.is_over
        self.n_moves = len(rnd.move_sheet)
        self.n_actions = len(game.actions)


def snapshot(game):
    return Snap(game)


def restore(game, snap):
    """Rewind `game` to `snap` (same object family; moves are truncated).

    Does NOT rebuild the per-player meld caches: scratch games must only be
    driven through fast_legal_actions/playout, never the rlcard judge.
    """
    rnd = game.round
    rnd.dealer.stock_pile[:] = snap.stock
    rnd.dealer.discard_pile[:] = snap.discard
    for i in (0, 1):
        player = rnd.players[i]
        player.hand[:] = snap.hands[i]
        player.known_cards[:] = snap.knowns[i]
    rnd.current_player_id = snap.cur_id
    rnd._home_opening = snap.home_opening
    rnd.going_out_action = snap.going_out_action
    rnd.going_out_player_id = snap.going_out_id
    rnd.is_over = snap.is_over
    del rnd.move_sheet[snap.n_moves:]
    del game.actions[snap.n_actions:]


def _fast_add(self, card):
    self.hand.append(card)


def _fast_remove(self, card):
    self.hand.remove(card)


def make_scratch(game):
    """A deep copy for rollouts: per-player meld-cache upkeep is stripped
    (instance-level override), because only the bypassed judge reads those
    caches. Use with fast_legal_actions/playout exclusively."""
    scratch = copy.deepcopy(game)
    for player in scratch.round.players:
        player.add_card_to_hand = types.MethodType(_fast_add, player)
        player.remove_card_from_hand = types.MethodType(_fast_remove, player)
    return scratch


def unseen_pool(game, hero_id):
    """Cards the hero cannot see: stock + opponent's unknown cards."""
    rnd = game.round
    opp = rnd.players[1 - hero_id]
    return list(rnd.dealer.stock_pile) + \
        [c for c in opp.hand if c not in opp.known_cards]


def apply_world(game, opp_id, sampled_free_cards, rng):
    """Determinize: opponent's unknown cards become `sampled_free_cards`; the
    rest of the unseen pool is shuffled into the stock. Preserves hand size,
    known cards, the discard pile, and the phase."""
    rnd = game.round
    opp = rnd.players[opp_id]
    pool = list(rnd.dealer.stock_pile) + \
        [c for c in opp.hand if c not in opp.known_cards]
    taken = set(ids_of(sampled_free_cards))
    rest = [c for c in pool if CARD_ID[c.rank + c.suit] not in taken]
    opp.hand[:] = list(opp.known_cards) + list(sampled_free_cards)
    rng.shuffle(rest)
    rnd.dealer.stock_pile[:] = rest


# ---------------------------------------------------------------------------
# Stepping and legality without the Env or the judge.
# ---------------------------------------------------------------------------

def step_fast(game, action):
    """game.step minus its throwaway get_state() call. Dispatch mirrors game.py."""
    rnd = game.round
    opening = getattr(rnd, "_home_opening", 0)
    if opening:
        # Home OPENING rule (user decision 2026-09-01): first player may
        # take the flip; else the second player may take it (out of
        # turn, play then returns to the first player); else the first
        # player MUST draw from stock. A DrawCardAction here means PASS
        # -- no card moves, the pass is recorded as public decline
        # evidence, and the option (or the forced draw) moves on.
        if isinstance(action, PickUpDiscardAction):
            rnd._home_opening = 0
            rnd.pick_up_discard(action)
            game.actions.append(action)
            return
        if isinstance(action, DrawCardAction):
            rnd.move_sheet.append(
                HomePassMove(rnd.players[rnd.current_player_id]))
            rnd._home_opening = 2 if opening == 1 else 0
            rnd.current_player_id = 1 - rnd.current_player_id
            game.actions.append(HomePassAction())
            return
        raise ValueError(f"illegal opening action {action}")
    if isinstance(action, DrawCardAction):
        rnd.draw_card(action)
    elif isinstance(action, DiscardAction):
        rnd.discard(action)
    elif isinstance(action, PickUpDiscardAction):
        rnd.pick_up_discard(action)
    elif isinstance(action, KnockAction):
        rnd.knock(action)
    elif isinstance(action, GinAction):
        rnd.gin(action, going_out_deadwood_count=
                game.settings.going_out_deadwood_count)
    elif isinstance(action, DeclareDeadHandAction):
        rnd.declare_dead_hand(action)
    elif isinstance(action, ScoreNorthPlayerAction):
        rnd.score_player_0(action)
    elif isinstance(action, ScoreSouthPlayerAction):
        rnd.score_player_1(action)
    else:
        raise ValueError(f"unknown action {action}")
    game.actions.append(action)


def fast_legal_actions(game):
    """GinRummyJudge.get_legal_actions, replicated (same actions, same order).

    The expensive part of the original is the going-out computation on every
    discard node. A deadwood gate (cached id-space solver) proves the common
    case -- no knock, no gin -- without it; when a knock or gin IS possible,
    the real rlcard `get_going_out_cards` supplies the exact card lists, so
    even the set-iteration order of knock actions matches the judge's.
    """
    settings = game.settings
    rnd = game.round
    last = game.actions[-1] if game.actions else None

    if getattr(rnd, "_home_opening", 0):
        # Opening option (home rule): take the flip or pass. Same
        # [draw, pickup] action shape as a normal draw decision, so
        # every policy/searcher sees a familiar node; step_fast gives
        # the DrawCardAction its pass semantics.
        legal = [DrawCardAction()]
        if settings.is_allowed_pick_up_discard:
            legal.append(PickUpDiscardAction())
        return legal
    if isinstance(last, HomePassAction):
        # both passed: the first player's stock draw is FORCED
        return [DrawCardAction()]

    if last is None or isinstance(last, (DrawCardAction, PickUpDiscardAction)):
        player = rnd.players[rnd.current_player_id]
        hand = player.hand
        hand_ids = ids_of(hand)
        # Going-out options depend only on the hand: cache them, including the
        # common negative result.
        key = (tuple(sorted(hand_ids)), settings.going_out_deadwood_count)
        hit = _GO_CACHE.get(key)
        if hit is None:
            hit = _going_out_ids(key[0], key[1])
            if len(_GO_CACHE) < _GO_CACHE_LIMIT:
                _GO_CACHE[key] = hit
        knock_ids, has_gin = hit
        if settings.is_allowed_gin and has_gin:
            return [GinAction()]
        cards_to_discard = list(hand)
        if isinstance(last, PickUpDiscardAction) and \
                not settings.is_allowed_to_discard_picked_up_card:
            cards_to_discard.remove(rnd.move_sheet[-1].card)
        legal = [DiscardAction(card=c) for c in cards_to_discard]
        if settings.is_allowed_knock and knock_ids and (
                player.player_id == 0 or not settings.is_south_never_knocks):
            knocks = [KnockAction(card=DECK[i]) for i in knock_ids]
            if settings.is_always_knock:
                return knocks
            legal.extend(knocks)
        return legal

    if isinstance(last, DiscardAction):
        if len(rnd.move_sheet) >= settings.max_move_count:
            return [DeclareDeadHandAction()]
        can_draw = len(rnd.dealer.stock_pile) > settings.stockpile_dead_card_count
        if settings.max_drawn_card_count < 52:
            drawn = sum(1 for a in game.actions
                        if isinstance(a, DrawCardAction))
            if drawn >= settings.max_drawn_card_count:
                can_draw = False
        if not can_draw:
            # Standard dead-hand rule (user decision 2026-08-30): once
            # the stock is down to 2 cards the hand is over 0-0 -- the
            # player to move may not draw OR pick up. This is the ONE
            # deliberate divergence from the rlcard judge, which lets
            # pickup chains continue on a dead stock; --verify knows
            # about it. (Match results gated before 2026-08-30 were
            # measured under the rlcard variant.)
            return [DeclareDeadHandAction()]
        legal = [DrawCardAction()]
        if settings.is_allowed_pick_up_discard:
            legal.append(PickUpDiscardAction())
        return legal

    if isinstance(last, (KnockAction, GinAction, DeclareDeadHandAction)):
        return [ScoreNorthPlayerAction()]
    if isinstance(last, ScoreNorthPlayerAction):
        return [ScoreSouthPlayerAction()]
    return []                            # after ScoreSouth: game over


# ---------------------------------------------------------------------------
# Payoff: true gin rummy points for player 0, same arithmetic as
# train_gin_rummy.real_points_payoff.
# ---------------------------------------------------------------------------

GIN_BONUS = 25
UNDERCUT_BONUS = 25


def fast_payoff(game):
    """Score differential for player 0, in points/100 (zero-sum). Valid as
    soon as going_out_action is set (both hands hold exactly 10 cards then) --
    the two cosmetic scoring steps need not have been played."""
    rnd = game.round
    action = rnd.going_out_action
    goer_id = rnd.going_out_player_id
    if goer_id is None:
        return 0.0
    if isinstance(action, GinAction):
        points = min_deadwood(rnd.players[1 - goer_id].hand) + GIN_BONUS
        winner = goer_id
    elif isinstance(action, KnockAction):
        knocker_dw = min_deadwood(rnd.players[goer_id].hand)
        defender_dw = min_deadwood(rnd.players[1 - goer_id].hand)
        if knocker_dw < defender_dw:
            points = defender_dw - knocker_dw
            winner = goer_id
        else:
            points = knocker_dw - defender_dw + UNDERCUT_BONUS
            winner = 1 - goer_id
    else:                                # dead hand
        return 0.0
    return points / 100.0 if winner == 0 else -points / 100.0


# ---------------------------------------------------------------------------
# Fast policies.
# ---------------------------------------------------------------------------

def _adjacent(a, b):
    """Could two card ids ever share a meld? Same rank, or same suit within
    two ranks -- the neighborhood StrongTeacher._live uses."""
    if a % 13 == b % 13 and a // 13 != b // 13:
        return True
    return a // 13 == b // 13 and abs(a % 13 - b % 13) in (1, 2)


def _live_count(card_id, pool_ids):
    return sum(1 for other in pool_ids if _adjacent(card_id, other))


class FastTeacher:
    """teacher.StrongTeacher, ported into id space.

    Decision-for-decision identical to the original (verified by --verify).
    Parameters have the same meaning as StrongTeacher's; defaults are the
    tuned v2 config.
    """

    def __init__(self, draw_gain=1, knock_at=0, endgame_stock=0,
                 w_live=1, w_safe=1, w_value=0):
        self.draw_gain = draw_gain
        self.knock_at = knock_at
        self.endgame_stock = endgame_stock
        self.w_live, self.w_safe, self.w_value = w_live, w_safe, w_value

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        rnd = game.round
        me = rnd.players[rnd.current_player_id]
        opp = rnd.players[1 - rnd.current_player_id]
        hand_ids = ids_of(me.hand)

        gin = [e for e in legal if isinstance(e, GinAction)]
        if gin:                          # engine makes gin the only action anyway
            return gin[0]

        knocks = [e for e in legal if isinstance(e, KnockAction)]
        if knocks:
            best, best_dw = None, 999
            for event in knocks:         # stable min, judge order
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
            for event in discards:       # legal-event order == judge order
                drop = cid(event.card)
                dw = min_deadwood_ids([i for i in hand_ids if i != drop])
                if dw < best_dw:
                    best_dw, best_events = dw, [event]
                elif dw == best_dw:
                    best_events.append(event)
            if len(best_events) == 1:
                return best_events[0]
            unseen_ids = ids_of(rnd.dealer.stock_pile)
            unseen_ids += [cid(c) for c in opp.hand
                           if c not in opp.known_cards]
            opp_ids = ids_of(opp.known_cards)
            return min(best_events, key=lambda e: (
                self.w_live * _live_count(cid(e.card), unseen_ids),
                self.w_safe * _live_count(cid(e.card), opp_ids),
                -self.w_value * DEADWOOD_VALUE[cid(e.card)]))

        pickup = [e for e in legal if isinstance(e, PickUpDiscardAction)]
        other = [e for e in legal
                 if isinstance(e, (DrawCardAction, DeclareDeadHandAction))]
        if pickup and other:
            if not rnd.dealer.discard_pile or len(me.hand) != 10:
                return other[0]
            top = cid(rnd.dealer.discard_pile[-1])
            opening = getattr(rnd, "_home_opening", 0)
            if opening:
                # OPENING BOOK -- see flip_book_take.
                return pickup[0] if flip_book_take(hand_ids, top,
                                                   opening) else other[0]
            candidate = hand_ids + [top]
            after = min(min_deadwood_ids([i for i in candidate if i != drop])
                        for drop in hand_ids)
            gain = min_deadwood_ids(hand_ids) - after
            return pickup[0] if gain >= self.draw_gain else other[0]

        return legal[0]                  # unreachable for legal gin rummy states


class MixedKnock:
    """Wrap any fast policy: knock with probability p when the best legal knock
    leaves deadwood <= threshold; otherwise defer to the base policy with the
    knocks stripped. p in {0.0, 0.5, 1.0} is the mixing discretization; the RNG
    is caller-owned so matches stay reproducible."""

    def __init__(self, base, threshold, p, rng=None):
        self.base = base
        self.threshold = threshold
        self.p = p
        self.rng = rng or random.Random(0)

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        knocks = [e for e in legal if isinstance(e, KnockAction)]
        if knocks and not any(isinstance(e, GinAction) for e in legal):
            rnd = game.round
            hand_ids = ids_of(rnd.players[rnd.current_player_id].hand)
            best, best_dw = None, 999
            for event in knocks:
                drop = cid(event.card)
                dw = min_deadwood_ids([i for i in hand_ids if i != drop])
                if dw < best_dw:
                    best, best_dw = event, dw
            if best_dw <= self.threshold:
                if self.p >= 1.0 or (self.p > 0.0 and self.rng.random() < self.p):
                    return best
            remaining = [e for e in legal if not isinstance(e, KnockAction)]
            if remaining:
                return self.base.act(game, remaining)
            return best
        return self.base.act(game, legal)


# ---------------------------------------------------------------------------
# Playout driver.
# ---------------------------------------------------------------------------

def _hand_value(hand_ids):
    """Deadwood after the best discard for 11 cards, plain deadwood for 10."""
    if len(hand_ids) == 11:
        return min(min_deadwood_ids(hand_ids[:k] + hand_ids[k + 1:])
                   for k in range(11))
    return min_deadwood_ids(hand_ids)


def _best10(hand_ids):
    """The kept 10 cards: for 11, drop the discard minimizing deadwood."""
    if len(hand_ids) != 11:
        return hand_ids
    best_k, best_dw = 0, 1000
    for k in range(11):
        dw = min_deadwood_ids(hand_ids[:k] + hand_ids[k + 1:])
        if dw < best_dw:
            best_k, best_dw = k, dw
    return hand_ids[:best_k] + hand_ids[best_k + 1:]


def leaf_value(game):
    """Truncation leaf: player 0's deadwood edge, on the payoff scale.

    Classic rules: (dw1 - dw0)/100 -- exactly what a knock would score now.
    Full rules: the same idea but with EFFECTIVE deadwood -- each side's
    residual after optimally laying off onto the other's melds. A raw-deadwood
    leaf under full rules is a BIASED evaluator (it ignores lay-offs), and the
    expert's z-test guards against noise, not bias: with the raw leaf the
    full-rules expert deviated systematically and measured WORSE than its own
    teacher prior (matrix: -0.73 vs teacher, field margins halved).
    """
    rnd = game.round
    ids0 = ids_of(rnd.players[0].hand)
    ids1 = ids_of(rnd.players[1].hand)
    if _RULES == "full":
        from fullrules import defender_residual
        h0, h1 = _best10(ids0), _best10(ids1)
        eff0 = defender_residual(h1, h0)     # my residual if they knocked now
        eff1 = defender_residual(h0, h1)
        return (eff1 - eff0) / 100.0
    return (_hand_value(ids1) - _hand_value(ids0)) / 100.0


# Which terminal scorer playouts use. "classic" = rlcard-style (no lay-offs);
# "full" = real rules (lay-offs + undercut after lay-offs, fullrules.py).
# Worker entry points call set_rules() from their job parameters.
_PAYOFF = None                           # resolved lazily to avoid an import cycle
_RULES = "classic"


def set_rules(rules):
    global _PAYOFF, _RULES
    _RULES = rules
    if rules == "full":
        from fullrules import full_payoff
        _PAYOFF = full_payoff
    else:
        _PAYOFF = fast_payoff


def get_rules():
    return _RULES


def _payoff(game):
    global _PAYOFF
    if _PAYOFF is None:
        _PAYOFF = fast_payoff
    return _PAYOFF(game)


def playout(game, policies, max_steps=250, truncate=None):
    """Play `game` until someone goes out (or it dies); return player 0's
    true-points payoff. The two cosmetic scoring steps are skipped -- they
    only exist to stamp the move sheet, and fast_payoff doesn't need them.

    `truncate`: stop after that many decisions and return `leaf_value` if
    nobody has gone out -- the variance-control mode used by the expert.
    """
    rnd = game.round
    budget = max_steps if truncate is None else truncate
    for _ in range(budget):
        if rnd.going_out_action is not None or rnd.is_over:
            return _payoff(game)
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 \
            else policies[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
    if truncate is not None and rnd.going_out_action is None \
            and not rnd.is_over:
        return leaf_value(game)
    return _payoff(game)


def new_game(seed=None):
    """A fresh dealt game, seeded, without any Env plumbing."""
    game = GinRummyGame()
    if seed is not None:
        import numpy as np
        game.np_random = np.random.RandomState(seed)
    game.init_game()
    return game


def new_game_home(seed=None, flip_index=None):
    """The user's home variant (2026-08-24): both players start with 10
    cards, one card is FLIPPED to found the discard pile, and the hand's
    knock threshold is that card's deadwood value (A=1 ... 9=9, 10/J/Q/K=10).

    Implemented as a transform of the standard 11/10 deal: one uniformly
    random card of the 11-card hand becomes the flip -- statistically
    identical to dealing 10/10 and flipping the next stock card. The former
    11-card seat acts first, facing a normal draw decision (it may take the
    flip). Drive the result only through fast_legal_actions/step_fast/
    playout, never the rlcard judge. `rnd._home_flip_id` lets Belief seed
    its replayed pile (the flip is nobody's discard).
    """
    game = new_game(seed)
    rnd = game.round
    player = rnd.players[rnd.current_player_id]
    assert len(player.hand) == 11
    k = int(game.np_random.randint(11)) if flip_index is None else flip_index
    flip = player.hand[k]
    del player.hand[k]
    rnd.dealer.discard_pile.append(flip)
    game.settings.going_out_deadwood_count = DEADWOOD_VALUE[cid(flip)]
    game.actions.append(DiscardAction(card=flip))
    rnd._home_flip_id = cid(flip)
    rnd._home_opening = 1                # opening rule: P1's flip option
    return game


# ---------------------------------------------------------------------------
# Verification and benchmarks.
# ---------------------------------------------------------------------------

def _obs_state(game, legal):
    """Build the (5,52) observation dict StrongTeacher expects, matching
    envs/gin_rummy.py:_extract_state. Verification-only path."""
    from collections import OrderedDict
    import numpy as np
    rnd = game.round
    discard_pile = rnd.dealer.discard_pile
    top_discard = [] if not discard_pile else [discard_pile[-1]]
    dead_cards = discard_pile[:-1]
    current = rnd.players[rnd.current_player_id]
    opponent = rnd.players[1 - rnd.current_player_id]
    known = opponent.known_cards
    unknown = rnd.dealer.stock_pile + [c for c in opponent.hand
                                       if c not in known]
    obs = np.array([gr_utils.encode_cards(x) for x in
                    (current.hand, top_discard, dead_cards, known, unknown)])
    ids = OrderedDict((e.action_id, None) for e in legal)
    return {"obs": obs, "legal_actions": ids,
            "raw_legal_actions": list(ids.keys())}


def verify(num_hands=250, seed=1234):
    """All correctness gates in one pass over real games:

    1. fast_legal_actions == judge.get_legal_actions -- same action set, and
       exactly the same order for everything except knocks. Knock order comes
       from a Python-set iteration inside the judge (insertion order depends
       on the player's incrementally-built meld caches) and is not worth
       replicating: a knock ends the hand on the spot, and tied knocks leave
       equal deadwood, so WHICH tied knock is chosen can never change the
       payoff. Discard order IS exact -- it feeds tie-breaks that shape play.
    2. FastTeacher == StrongTeacher on every decision, where knock-vs-knock
       disagreements count as agreement iff both leave the same deadwood
       (payoff-identical by the argument above). Everything else exact.
    3. min_deadwood_ids == shaping._min_deadwood (also on random hands below)
    4. fast_payoff == real_points_payoff, and zero-sum
    """
    from shaping import _min_deadwood as slow_dw
    from teacher import StrongTeacher
    from train_gin_rummy import real_points_payoff

    slow = StrongTeacher()
    fast = FastTeacher()
    decisions = agreements = legal_checks = 0
    knock_ties = knock_order_diffs = dead_hand_diffs = 0
    for h in range(num_hands):
        game = new_game(seed + h)
        while not game.round.is_over:
            legal = game.judge.get_legal_actions()
            judge_ids = [e.action_id for e in legal]
            fast_actions = fast_legal_actions(game)
            fast_ids = [e.action_id for e in fast_actions]
            if fast_ids == [DeclareDeadHandAction().action_id] and \
                    fast_ids != judge_ids:
                # The ONE deliberate rules divergence (2026-08-30):
                # dead stock ends the hand; the judge would still offer
                # the pickup. Play OUR rule and end the hand here.
                assert DeclareDeadHandAction().action_id in judge_ids, \
                    (h, fast_ids, judge_ids)
                dead_hand_diffs += 1
                legal_checks += 1
                step_fast(game, fast_actions[0])
                continue
            if fast_ids != judge_ids:
                assert [i for i in fast_ids if i < 58] == \
                    [i for i in judge_ids if i < 58] and \
                    sorted(fast_ids) == sorted(judge_ids), \
                    (h, fast_ids, judge_ids)
                knock_order_diffs += 1
            legal_checks += 1
            if len(legal) > 1:
                state = _obs_state(game, legal)
                a_slow = int(slow.step(state))
                a_fast = fast.act(game, legal).action_id
                decisions += 1
                if a_slow == a_fast:
                    agreements += 1
                elif a_slow >= 58 and a_fast >= 58:
                    hand_ids = ids_of(
                        game.round.players[game.round.current_player_id].hand)
                    dw_slow = min_deadwood_ids(
                        [i for i in hand_ids if i != a_slow - 58])
                    dw_fast = min_deadwood_ids(
                        [i for i in hand_ids if i != a_fast - 58])
                    assert dw_slow == dw_fast, (h, a_slow, a_fast)
                    agreements += 1
                    knock_ties += 1
                else:
                    print(f"  MISMATCH hand {h}: slow={a_slow} fast={a_fast}")
                action = next(e for e in legal if e.action_id == a_fast)
            else:
                action = legal[0]
            step_fast(game, action)
        ours = fast_payoff(game)
        theirs = real_points_payoff(game.round.players[0], game)
        assert abs(ours - theirs) < 1e-12, (h, ours, theirs)
        assert abs(theirs + real_points_payoff(game.round.players[1], game)) \
            < 1e-12, "not zero-sum"

    # Deadwood solver equivalence on random hands of the sizes we use.
    rng = random.Random(seed)
    for _ in range(4000):
        size = rng.choice((9, 10, 10, 11, 11))
        hand = rng.sample(DECK, size)
        assert min_deadwood(hand) == slow_dw(hand), [str(c) for c in hand]

    rate = agreements / max(decisions, 1)
    print(f"verify: {legal_checks} legal-action checks "
          f"({knock_order_diffs} knock-order-only diffs, {dead_hand_diffs} "
          f"strict-dead-hand rule diffs), {decisions} decisions "
          f"over {num_hands} hands, teacher agreement {rate:.4%} "
          f"({knock_ties} payoff-identical knock ties), "
          f"dw parity on 4000 random hands, payoff parity on {num_hands} hands")
    return rate == 1.0


def _mid_state_games(n, seed, min_steps=8, max_steps=30):
    """Sample n games advanced a random number of steps -- rollout start points."""
    rng = random.Random(seed)
    fast = FastTeacher()
    games = []
    while len(games) < n:
        game = new_game(rng.randrange(1 << 30))
        target = rng.randrange(min_steps, max_steps)
        steps = 0
        while game.round.going_out_action is None and steps < target:
            legal = fast_legal_actions(game)
            action = legal[0] if len(legal) == 1 else fast.act(game, legal)
            step_fast(game, action)
            steps += 1
        if game.round.going_out_action is None and not game.round.is_over:
            games.append(game)
    return games


def bench(num_states=40, rollouts_per_state=50, seed=99):
    """Continuations/s from mid-hand states, with determinization and
    snapshot/restore in the loop, exactly as the expert will run them."""
    import numpy as np
    rng = np.random.default_rng(seed)
    policies = {0: FastTeacher(), 1: FastTeacher()}
    states = _mid_state_games(num_states, seed)
    total = 0
    t0 = time.time()
    for game in states:
        scratch = make_scratch(game)
        base = snapshot(scratch)
        opp_id = 1 - scratch.round.current_player_id
        pool = unseen_pool(scratch, 1 - opp_id)
        m = len(scratch.round.players[opp_id].hand) - \
            len(scratch.round.players[opp_id].known_cards)
        for _ in range(rollouts_per_state):
            restore(scratch, base)
            pick = rng.choice(len(pool), size=m, replace=False)
            apply_world(scratch, opp_id, [pool[i] for i in pick], rng)
            playout(scratch, policies)
            total += 1
    dt = time.time() - t0
    print(f"bench: {total} determinized playout-continuations in {dt:.1f}s "
          f"-> {total / dt:.0f}/s/core (deadwood cache: {len(_DW_CACHE)})")
    return total / dt


def stress(num_playouts=10_000, seed=7):
    """Bulk: terminate + card conservation under noisy policies."""
    rng = random.Random(seed)
    teacher = FastTeacher()

    class Noisy:
        def act(self, game, legal):
            if rng.random() < 0.2:
                return legal[rng.randrange(len(legal))]
            return teacher.act(game, legal)

    policies = {0: Noisy(), 1: Noisy()}
    dead = 0
    for i in range(num_playouts):
        game = new_game(rng.randrange(1 << 30))
        payoff = playout(game, policies)
        rnd = game.round
        n_cards = (len(rnd.players[0].hand) + len(rnd.players[1].hand)
                   + len(rnd.dealer.stock_pile) + len(rnd.dealer.discard_pile))
        done = rnd.going_out_action is not None or rnd.is_over
        # After knock/gin the going-out discard leaves every pile.
        assert done and n_cards in (51, 52), (i, done, n_cards)
        dead += (rnd.going_out_player_id is not None
                 and not isinstance(rnd.going_out_action,
                                    (GinAction, KnockAction)))
        assert -1.0 <= payoff <= 1.0
    print(f"stress: {num_playouts} noisy playouts all terminated "
          f"({dead} dead hands, {dead / num_playouts:.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("sim kernel gates")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--hands", type=int, default=250)
    args = ap.parse_args()
    if args.verify:
        ok = verify(args.hands)
        raise SystemExit(0 if ok else 1)
    if args.bench:
        bench()
    if args.stress:
        stress()
