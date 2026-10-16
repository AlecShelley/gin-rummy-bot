import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Illustration (user request, 2026-08-29): the SAME position evaluated by
v6home under two different flips -- an ACE (knock threshold 1: gin-ish
game) vs a TEN (threshold 10: knock-happy game).

Construction: two games with IDENTICAL hero and opponent hands and an
identical scripted first move (hero draws 4C from the stock). The ONLY
difference is the founding flip: AD vs TD (the unused twin is buried at
the stock bottom, never reached). After the draw the hero holds three
complete melds (4S-5S-6S, 9H-9S-9C, 2C-3C-4C) plus 6H and 8D: dropping
8D leaves 6 deadwood -- a LEGAL knock in the ten game, ILLEGAL in the
ace game (needs <= 1).

Everything else the two searches see also tracks the flip: the opponent
knock model is (flip, 1.0) -- exact, the threshold is public -- so
rollout continuations and world weighting reflect "they knock the
moment they can" (ten) vs "they effectively play to gin" (ace).

Population-level context (results/home_stats.log, 30k hands): ace flips
run ~44.5 steps with ~50% gin; ten flips ~25.7 steps with ~98% knock.

    .venv/Scripts/python.exe home_demo.py
"""

import numpy as np

import sim
from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, DrawCardAction, GinAction, KnockAction)

sim.set_rules("full")

from pf_expert import ParticleBelief, PFExpert, V5_CFG  # noqa: E402
from sim import (CARD_ID, DECK, DEADWOOD_VALUE, cid, fast_legal_actions,
                 min_deadwood_ids, step_fast)  # noqa: E402

# Panel A -- a GIN-SHAPED hand: three complete melds after the draw,
# one useful spare; the gin chase is real.
HERO_A = ["4S", "5S", "6S", "9H", "9S", "9C", "2C", "3C", "6H", "8D"]
OPP_A = ["7H", "8H", "JS", "JD", "QS", "KC", "2H", "5C", "TH", "AC"]
DRAW_A = "4C"                            # completes 2C-3C-4C

# Panel B -- a KNOCK-SHAPED hand: seven melded cards but the last three
# (2H 3D 5C) are scattered bricks summing to exactly 10; gin is remote.
HERO_B = ["4S", "5S", "6S", "9H", "9S", "9C", "9D", "2H", "3D", "5C"]
OPP_B = ["7H", "8H", "JS", "JD", "QH", "KC", "6C", "TS", "TH", "AC"]
DRAW_B = "QS"                            # a brick; best discard = QS


def build_game(flip_name, other_flip, hero_cards, opp_cards, draw):
    """Identical position; only the flip differs. The unused twin sits
    at the stock bottom (index 0; stock pops from the END)."""
    game = sim.new_game_home(seed=1, flip_index=0)
    rnd = game.round
    hero_id = rnd.current_player_id      # the former-11 seat acts first
    opp_id = 1 - hero_id
    used = set(hero_cards + opp_cards + [flip_name, other_flip, draw])
    assert len(used) == 23, "card overlap in the constructed position"
    rest = [k for k in CARD_ID if k not in used]
    rnd.players[hero_id].hand[:] = [DECK[CARD_ID[k]] for k in hero_cards]
    rnd.players[opp_id].hand[:] = [DECK[CARD_ID[k]] for k in opp_cards]
    rnd.players[hero_id].known_cards[:] = []
    rnd.players[opp_id].known_cards[:] = []
    # stock: bottom = unused twin + the rest; top (END) = the draw card
    rnd.dealer.stock_pile[:] = (
        [DECK[CARD_ID[other_flip]]] +
        [DECK[CARD_ID[k]] for k in rest] +
        [DECK[CARD_ID[draw]]])
    flip = DECK[CARD_ID[flip_name]]
    rnd.dealer.discard_pile[:] = [flip]
    game.settings.going_out_deadwood_count = DEADWOOD_VALUE[cid(flip)]
    rnd._home_flip_id = cid(flip)
    game.actions[-1] = DiscardAction(card=flip)
    return game, hero_id


def act_name(action):
    if isinstance(action, KnockAction):
        return f"KNOCK (discard {action.card})"
    if isinstance(action, GinAction):
        return "GIN"
    if isinstance(action, DiscardAction):
        return f"discard {action.card}"
    if isinstance(action, DrawCardAction):
        return "draw from stock"
    return type(action).__name__


def evaluate(flip_name, other_flip, hero_cards, opp_cards, draw_name,
             seed=424242):
    game, hero_id = build_game(flip_name, other_flip, hero_cards,
                               opp_cards, draw_name)
    threshold = game.settings.going_out_deadwood_count
    # scripted first move: hero draws from the stock (declining the
    # flip) -- a real engine move, so the belief's replay is genuine
    legal = fast_legal_actions(game)
    draw = next(e for e in legal if isinstance(e, DrawCardAction))
    step_fast(game, draw)

    belief = ParticleBelief(hero_id, seed=seed)
    expert = PFExpert(veto=True, opp_model=(threshold, 1.0),
                      sample_boost=0.0, seed=seed, **V5_CFG)
    hand_ids = [cid(c) for c in game.round.players[hero_id].hand]
    best_dw = min(min_deadwood_ids([i for i in hand_ids if i != d])
                  for d in hand_ids)
    result = expert.analyze(game, belief)

    flip_card = DECK[CARD_ID[flip_name]]
    print(f"\n--- flip {flip_card}  ->  knock threshold {threshold} ---")
    print(f"best possible knock leaves {best_dw} deadwood -> knock is "
          f"{'LEGAL' if best_dw <= threshold else 'ILLEGAL'} this game")
    if "payoff_means" in result:
        order = np.argsort(-result["payoff_means"])
        print(f"{'candidate':26s} {'EV pts':>8s} {'dSE':>6s}")
        for k in order:
            ev = result["payoff_means"][k] * 100.0
            se = result["se"][k] * 100.0
            mark = " <-- chosen" if result["candidates"][k] is \
                result["chosen"] else ""
            print(f"{act_name(result['candidates'][k]):26s} "
                  f"{ev:+8.1f} {se:6.1f}{mark}")
        print(f"(EV = mean hero payoff over sampled worlds; dSE = the "
              f"paired std error of that candidate's DIFF vs the "
              f"teacher prior; worlds {result['worlds']}, reason: "
              f"{result['reason']})")
    else:
        print(f"chosen: {act_name(result['chosen'])} "
              f"({result['reason']})")
    return result


def panel(title, hero_cards, opp_cards, draw_name):
    print(f"\n=== {title} ===")
    print(f"hero: {' '.join(hero_cards)}  + drawn {draw_name}")
    evaluate("AD", "TD", hero_cards, opp_cards, draw_name)
    evaluate("TD", "AD", hero_cards, opp_cards, draw_name)


if __name__ == "__main__":
    print("v6home evaluating the SAME hand under two different flips")
    print("(population context, 30k-hand stats: ace flips -> ~44-step")
    print(" hands, ~50% gin; ten flips -> ~26-step hands, ~98% knock)")
    panel("Panel A: gin-shaped hand (three melds + useful spares)",
          HERO_A, OPP_A, DRAW_A)
    panel("Panel B: knock-shaped hand (7 melded, 3 scattered bricks, "
          "gin remote)", HERO_B, OPP_B, DRAW_B)
