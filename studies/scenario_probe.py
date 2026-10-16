import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Scenario probe (user question 2026-09-03): the BEST-CASE non-melding
turn-2 take -- an adjacency-2 ace (A-heart with A-spade + 2-heart
held), dumping a dead K-diamond, in a 10-point game. Is THAT take
positive EV?

Constructed-state paired measurement: hero at their first free draw
decision, pile = [flip(K-club), hero junk discard, A-heart], threshold
10. Randomize hero filler (no melds with the A-heart, K-diamond kept
dead) and the opponent hand/stock; per world play forced TAKE vs
forced DRAW to the end of the hand. Continuations: hero = patient
teacher, opp = MixedKnock(10, 1.0) (champion convention), plus a
symmetric-knocker robustness variant.

    .venv/Scripts/python.exe scenario_probe.py --worlds 3000
"""

import argparse
import math
import random
import types

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, DrawCardAction, PickUpDiscardAction)

import sim
from sim import (CARD_ID, DECK, FastTeacher, MixedKnock,
                 fast_legal_actions, min_deadwood_ids, step_fast)


BASE = dict(taken="AH", fixed=("AS", "2H", "KD"), flip="KC",
            threshold=10,
            ban=("AD", "AC", "3H", "KS", "KH", "QD", "JD"))

# The factor map: vary one axis at a time from the user's best case.
CONFIGS = [
    ("base: AH, adj2, dump K, t10", BASE),
    ("t5 game (flip 5C)", dict(BASE, flip="5C", threshold=5)),
    ("t1 game (flip AC)", dict(BASE, flip="AC", threshold=1,
                               ban=("AD", "3H", "KS", "KH", "QD",
                                    "JD"))),
    ("adj1 (AS only, no 2H)", dict(BASE, fixed=("AS", "KD"),
                                   ban=("AD", "AC", "2H", "3H", "KS",
                                        "KH", "QD", "JD"))),
    ("adj0 (no helpers)", dict(BASE, fixed=("KD",),
                               ban=("AD", "AC", "AS", "2H", "3H",
                                    "KS", "KH", "QD", "JD"))),
    ("small dump (6D dead, no K)", dict(BASE, fixed=("AS", "2H", "6D"),
                                        ban=("AD", "AC", "3H", "6S",
                                             "6H", "6C", "5D", "7D",
                                             "KD", "KS", "KH"))),
    ("mid card (5H w/ 5S+6H)", dict(BASE, taken="5H",
                                    fixed=("5S", "6H", "KD"),
                                    ban=("5D", "5C", "4H", "7H", "KS",
                                         "KH", "QD", "JD"))),
]


def _mk_game(rng, cfg):
    """One constructed world of a scenario config. Returns a
    scratch-safe game at hero's draw decision with cfg['taken'] on
    top."""
    game = sim.new_game(seed=0)
    rnd = game.round
    hero, opp = rnd.players[0], rnd.players[1]
    AH = CARD_ID[cfg["taken"]]
    KC = CARD_ID[cfg["flip"]]
    fixed = [CARD_ID[c] for c in cfg["fixed"]]
    banned = {AH, KC} | set(fixed) | \
        {CARD_ID[c] for c in cfg["ban"]}
    pool = [c for c in range(52) if c not in banned]
    while True:
        filler = rng.sample(pool, 10 - len(fixed))
        hand = fixed + filler
        cand = hand + [AH]
        if min_deadwood_ids(cand) <= min_deadwood_ids(hand):
            continue                     # filler melded the AH: reject
        break
    rest = [c for c in range(52)
            if c not in set(hand) and c not in (AH, KC)]
    rng.shuffle(rest)
    junk = rest[0]                       # hero's earlier discard
    opp_ids = rest[1:11]
    stock_ids = rest[11:]
    hero.hand[:] = [DECK[i] for i in hand]
    hero.known_cards[:] = []
    opp.hand[:] = [DECK[i] for i in opp_ids]
    opp.known_cards[:] = []
    rnd.dealer.discard_pile[:] = [DECK[KC], DECK[junk], DECK[AH]]
    rnd.dealer.stock_pile[:] = [DECK[i] for i in stock_ids]
    rnd.current_player_id = 0
    rnd.going_out_action = None
    rnd.going_out_player_id = None
    rnd.is_over = False
    rnd._home_flip_id = KC
    rnd._home_opening = 0
    game.settings.going_out_deadwood_count = cfg["threshold"]
    for player in (hero, opp):
        player.add_card_to_hand = types.MethodType(
            lambda self, card: self.hand.append(card), player)
        player.remove_card_from_hand = types.MethodType(
            lambda self, card: self.hand.remove(card), player)
    game.actions[:] = [DiscardAction(card=DECK[AH])]
    return game


def _playout(game, symmetric, rng):
    rnd = game.round
    t = game.settings.going_out_deadwood_count
    if symmetric:
        pols = {0: MixedKnock(FastTeacher(), t, 1.0,
                              rng=random.Random(rng.randrange(1 << 30))),
                1: MixedKnock(FastTeacher(), t, 1.0,
                              rng=random.Random(rng.randrange(1 << 30)))}
    else:
        pols = {0: FastTeacher(),
                1: MixedKnock(FastTeacher(), t, 1.0,
                              rng=random.Random(rng.randrange(1 << 30)))}
    steps = 0
    while rnd.going_out_action is None and not rnd.is_over and steps < 250:
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 else \
            pols[rnd.current_player_id].act(game, legal)
        step_fast(game, action)
        steps += 1
    return sim._payoff(game) * 100.0


def run(worlds, seed, symmetric, cfg, name):
    rng = random.Random(seed)
    diffs = np.empty(worlds)
    for w in range(worlds):
        wseed = rng.randrange(1 << 30)
        res = {}
        for arm, want in (("T", PickUpDiscardAction),
                          ("D", DrawCardAction)):
            g = _mk_game(random.Random(wseed), cfg)
            legal = fast_legal_actions(g)
            step_fast(g, next(a for a in legal if isinstance(a, want)))
            res[arm] = _playout(g, symmetric, random.Random(wseed + 7))
        diffs[w] = res["T"] - res["D"]
    m = diffs.mean()
    ci = 1.96 * diffs.std(ddof=1) / math.sqrt(worlds)
    tag = "sym" if symmetric else "champ"
    print(f"  {name:30s} [{tag}]: {m:+6.2f} +/- {ci:.2f}", flush=True)
    return m, ci


def main():
    ap = argparse.ArgumentParser("best-case non-meld take probe")
    ap.add_argument("--worlds", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=84200)
    ap.add_argument("--map", dest="factor_map", action="store_true",
                    help="run the full factor map")
    args = ap.parse_args()
    sim.set_rules("full")
    configs = CONFIGS if args.factor_map else CONFIGS[:1]
    print(f"dEV(take) per scenario, {args.worlds} paired worlds each:")
    for i, (name, cfg) in enumerate(configs):
        run(args.worlds, args.seed + i, False, cfg, name)
        run(args.worlds, args.seed + 100 + i, True, cfg, name)


if __name__ == "__main__":
    main()
