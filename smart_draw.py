"""
Exploit the rule agent's one structural blind spot: it draws at random.

`GinRummyNoviceRuleAgent.step` branches on gin / knock / discard. The draw
decision (DrawCardAction vs PickUpDiscardAction) matches none of those, so
`actions` falls through as the full legal set and the agent calls
`np.random.choice` on it -- a coin flip. Every turn starts with that decision,
so about half of the "expert's" moves are noise.

This wrapper changes only that decision and delegates everything else to a base
agent: take the upcard when it strictly lowers your deadwood, otherwise draw
blind. The base can be the rule agent (to measure the headroom) or our trained
network (to actually use it).

    python smart_draw.py --num_games 1500
"""

import argparse
import math

import numpy as np

from rlcard.games.gin_rummy.utils import utils as gr_utils
from rlcard.games.gin_rummy.utils.action_event import (
    ActionEvent, DrawCardAction, PickUpDiscardAction, DeclareDeadHandAction)
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
from rlcard.utils import get_device, set_seed

from shaping import _min_deadwood
from train_gin_rummy import make_env, classify_outcome


class _Args:
    seed = 777
    reward = "points"


class SmartDraw:
    """Base agent, but the draw decision uses the upcard instead of a coin flip."""

    use_raw = False

    def __init__(self, base, allow_discard_picked_up=False):
        self.base = base
        # RLCard's default is is_allowed_to_discard_picked_up_card=False, so the
        # upcard cannot be thrown straight back; evaluating as if it could would
        # overstate how much picking up helps.
        self.allow_discard_picked_up = allow_discard_picked_up

    def _is_draw_decision(self, events):
        return any(isinstance(e, (DrawCardAction, DeclareDeadHandAction))
                   for e in events) and any(
            isinstance(e, PickUpDiscardAction) for e in events)

    def step(self, state):
        legal = list(state["legal_actions"].keys())
        events = [ActionEvent.decode_action(a) for a in legal]
        if not self._is_draw_decision(events):
            return self.base.step(state)

        hand = gr_utils.decode_cards(state["obs"][0])
        top = gr_utils.decode_cards(state["obs"][1])
        pickup_id = next(e.action_id for e in events
                         if isinstance(e, PickUpDiscardAction))
        other_id = next(e.action_id for e in events
                        if isinstance(e, (DrawCardAction, DeclareDeadHandAction)))
        if not top or len(hand) != 10:
            return other_id

        current = _min_deadwood(hand)
        candidate = hand + top
        drop_pool = candidate if self.allow_discard_picked_up else hand
        after = min(_min_deadwood([c for c in candidate if c != drop])
                    for drop in drop_pool)
        # Only take a card that actually improves the hand; otherwise the upcard
        # is also information handed to the opponent for nothing.
        return pickup_id if after < current else other_id

    def eval_step(self, state):
        return self.step(state), {}


def match(a, b, num_games, seed):
    """Seat-swapped match; returns points/hand for A with a 95% CI."""
    env = make_env(_Args, seed=seed)
    half = num_games // 2
    pts = []
    for a_seat in (0, 1):
        env.seed(seed)
        env.set_agents([a, b] if a_seat == 0 else [b, a])
        n = half if a_seat == 0 else num_games - half
        for _ in range(n):
            _, payoffs = env.run(is_training=False)
            pts.append(payoffs[a_seat] * 100.0)
    arr = np.array(pts)
    ci = 1.96 * arr.std(ddof=1) / math.sqrt(len(arr))
    return arr.mean(), ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_games", type=int, default=1500)
    ap.add_argument("--model", default="results/final/gin_rummy_bot.policy.pt")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    from gin_rummy_bot import GinRummyBot

    rule = GinRummyNoviceRuleAgent()
    bot = GinRummyBot(args.model, device=device, greedy=True)

    from rlcard.agents import RandomAgent
    smart_bot = SmartDraw(bot)
    tests = [
        ("rule + smart draw", SmartDraw(GinRummyNoviceRuleAgent()),
         "vs stock rule", rule),
        ("our bot (baseline)", bot, "vs stock rule", rule),
        ("our bot + smart draw", smart_bot, "vs stock rule", rule),
        ("our bot + smart draw", smart_bot, "vs random    ",
         RandomAgent(num_actions=110)),
        ("our bot + smart draw", smart_bot, "vs plain bot ", bot),
    ]
    print(f"{args.num_games} hands each, seats swapped\n")
    for name, agent, tag, opp in tests:
        mean, ci = match(agent, opp, args.num_games, args.seed)
        verdict = ("wins" if mean - ci > 0 else
                   "loses" if mean + ci < 0 else "statistical tie")
        print(f"{name:<22} {tag}: {mean:+7.2f} +/- {ci:.2f} pts/hand   {verdict}",
              flush=True)


if __name__ == "__main__":
    main()
