"""
A deterministic gin rummy teacher that is measurably stronger than the rule agent.

Why this exists
---------------
`GinRummyNoviceRuleAgent` flips coins in three places, not one:

  1. the draw decision (`DrawCardAction` vs `PickUpDiscardAction`) matches none of
     its branches, so it falls through to `np.random.choice` -- the finding that
     `smart_draw.py` already exploits;
  2. `_get_best_discards` returns *every* discard tied on resulting deadwood and
     the caller picks among them at random -- measured at 40.6% of all discard
     decisions, mean 1.64 tied options;
  3. when knocking is legal there are usually several legal `KnockAction`s (mean
     2.5) and it picks one at random, even though they leave different deadwood
     and knock points are `defender_deadwood - knocker_deadwood`.

Removing all three makes the teacher deterministic, which raises the accuracy a
behaviour clone can reach. Only the third is strictly stronger play on its own.

The large gain is elsewhere: the rule agent knocks the instant it is legal, and
that is a mistake in this engine. Against an opponent that knocks at deadwood 10,
holding for gin banks the +25 gin bonus and turns their early knocks into +25
undercuts. Measured over 1000 seat-swapped hands vs `SmartDraw(rule)`:

    knock at deadwood <= 10 (stock behaviour)   +0.90 +/- 1.39
    knock at deadwood <=  8                     +0.17 +/- 1.29
    knock at deadwood <=  6                     +0.52 +/- 1.39
    knock at deadwood <=  4                     +0.49 +/- 1.42
    knock at deadwood <=  2                     +2.12 +/- 1.52
    gin only (never knock)                      +3.77 +/- 1.59

A knock-anyway-when-the-stock-runs-low rule is available too (`endgame_stock`),
as insurance against dead hands, but the search below found it costs more than
it saves: this engine scores a dead hand as 0 for both players, so there is
nothing to insure against.

Everything is parameterised so `tune_teacher.py` can search it.
"""

from rlcard.games.gin_rummy.utils import utils as gr_utils
from rlcard.games.gin_rummy.utils.action_event import (
    ActionEvent, DeclareDeadHandAction, DiscardAction, DrawCardAction,
    GinAction, KnockAction, PickUpDiscardAction)
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent

from shaping import _min_deadwood

RANKS = "A23456789TJQK"

# Chosen by tune_teacher.py: coordinate descent over the grid below, then the
# top finalists re-scored on a held-out seed at 3000 hands/opponent. This config
# scored +3.84 +/- 0.92 vs SmartDraw(rule) there, against +3.11 +/- 0.91 for the
# hand-picked prototype it started from.
#
# knock_at=0 with endgame_stock=0 means: never knock, only gin. That reads
# extreme, and it is specific to this engine -- rlcard scores a dead hand as 0
# for both players and implements no lay-offs, so holding costs nothing but the
# knock points forgone, while the opponent's early knocks become undercuts. The
# outcome mix bears it out: against SmartDraw(rule) this teacher undercuts 32.5%
# of hands, and against a random opponent it reaches gin in 90.2% with no dead
# hands at all.
DEFAULTS = dict(draw_gain=1, knock_at=0, endgame_stock=0,
                w_live=1, w_safe=1, w_value=0)


def _rank_index(card):
    return RANKS.index(card.rank)


def _live(card, pool):
    """How many cards in `pool` could ever form a meld with `card`.

    Same rank (a set) or same suit within two ranks (a run). A crude but cheap
    stand-in for how much future there is in keeping the card.
    """
    r = _rank_index(card)
    return sum(1 for other in pool
               if (_rank_index(other) == r and other.suit != card.suit)
               or (other.suit == card.suit and abs(_rank_index(other) - r) in (1, 2)))


def stock_size(state):
    """Cards left in the stock pile, inferred from the observation planes.

    Plane 4 is stock + the opponent's *unknown* cards and plane 3 is the
    opponent's known cards, so the opponent's ten cards account for
    `10 - len(known)` of plane 4. See `envs/gin_rummy.py:_extract_state`.
    """
    unknown = len(gr_utils.decode_cards(state["obs"][4]))
    known = len(gr_utils.decode_cards(state["obs"][3]))
    return unknown - 10 + known


class StrongTeacher:
    """Deterministic rule agent: patient knocking, sorted discards, smart draws.

    API-compatible with an RLCard agent, so it drops into `env.set_agents`, into
    `pretrain.py`'s cloning roll-out and into `head_to_head.py` alike.
    """

    use_raw = False

    def __init__(self, draw_gain=1, knock_at=0, endgame_stock=0,
                 w_live=1, w_safe=1, w_value=0):
        self.draw_gain = draw_gain
        self.knock_at = knock_at
        self.endgame_stock = endgame_stock
        self.w_live, self.w_safe, self.w_value = w_live, w_safe, w_value
        self._base = GinRummyNoviceRuleAgent()

    # -- the three decisions ------------------------------------------------

    def _knock_action(self, events, hand):
        """The legal knock leaving the least deadwood, and that deadwood.

        The rule agent picks among legal knocks at random. Knock points are
        `defender_deadwood - knocker_deadwood`, so a lower own deadwood is
        strictly more points and strictly less undercut risk.
        """
        knocks = [e for e in events if isinstance(e, KnockAction)]
        if not knocks:
            return None, None
        best = min(knocks,
                   key=lambda k: _min_deadwood([c for c in hand if c != k.card]))
        return best, _min_deadwood([c for c in hand if c != best.card])

    def _discard_action(self, state, events):
        """Least-deadwood discard, ties broken deterministically."""
        discards = [e for e in events if isinstance(e, DiscardAction)]
        if not discards:
            return None
        best = GinRummyNoviceRuleAgent._get_best_discards(
            discard_action_events=discards, state=state)
        if not best:
            return None
        if len(best) > 1:
            unseen = gr_utils.decode_cards(state["obs"][4])
            opponent = gr_utils.decode_cards(state["obs"][3])
            # Throw the card with the least future in it; among those, the one
            # the opponent is least likely to want; among those, the dearest.
            best = [min(best, key=lambda c: (
                self.w_live * _live(c, unseen),
                self.w_safe * _live(c, opponent),
                -self.w_value * gr_utils.get_deadwood_value(c)))]
        return DiscardAction(card=best[0]).action_id

    def _draw_action(self, state, events):
        """Take the upcard only when it lowers deadwood by at least `draw_gain`.

        Picking up otherwise is doubly bad: it fails to help, and it tells the
        opponent exactly which card you hold.
        """
        pickup = [e for e in events if isinstance(e, PickUpDiscardAction)]
        other = [e for e in events
                 if isinstance(e, (DrawCardAction, DeclareDeadHandAction))]
        if not pickup or not other:
            return None                       # not a draw decision
        hand = gr_utils.decode_cards(state["obs"][0])
        top = gr_utils.decode_cards(state["obs"][1])
        if not top or len(hand) != 10:
            return other[0].action_id
        # The engine forbids discarding the picked-up card immediately
        # (is_allowed_to_discard_picked_up_card=False), so only the original ten
        # are discard candidates.
        candidate = hand + top
        after = min(_min_deadwood([c for c in candidate if c != drop])
                    for drop in hand)
        gain = _min_deadwood(hand) - after
        return pickup[0].action_id if gain >= self.draw_gain else other[0].action_id

    # -- agent interface ----------------------------------------------------

    def step(self, state):
        legal = list(state["legal_actions"].keys())
        if len(legal) == 1:
            return legal[0]
        events = [ActionEvent.decode_action(a) for a in legal]

        gin = [e for e in events if isinstance(e, GinAction)]
        if gin:
            return gin[0].action_id

        hand = gr_utils.decode_cards(state["obs"][0])
        knock, deadwood = self._knock_action(events, hand)
        if knock is not None:
            if deadwood <= self.knock_at or stock_size(state) <= self.endgame_stock:
                return knock.action_id
            # Hold: re-decide without the knocks. A discard is always available
            # alongside a knock, but fall back rather than assume it.
            events = [e for e in events if not isinstance(e, KnockAction)]
            if not events:
                return knock.action_id

        discard = self._discard_action(state, events)
        if discard is not None:
            return discard

        draw = self._draw_action(state, events)
        if draw is not None:
            return draw

        # Nothing above matched (e.g. a lone DeclareDeadHandAction). Defer to the
        # rule agent rather than inventing a rule for a case that barely occurs.
        return int(self._base.step(state))

    def eval_step(self, state):
        return self.step(state), {}

    def __repr__(self):
        return (f"StrongTeacher(draw_gain={self.draw_gain}, "
                f"knock_at={self.knock_at}, endgame_stock={self.endgame_stock}, "
                f"w_live={self.w_live}, w_safe={self.w_safe}, "
                f"w_value={self.w_value})")


def make_teacher(name):
    """Resolve a teacher name used by pretrain.py / dagger.py / benchmarks."""
    if name == "rule":
        return GinRummyNoviceRuleAgent()
    if name == "smart":
        from smart_draw import SmartDraw
        return SmartDraw(GinRummyNoviceRuleAgent())
    if name == "strong":
        return StrongTeacher(**DEFAULTS)
    raise ValueError(f"unknown teacher {name!r}")
