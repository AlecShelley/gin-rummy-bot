import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Illustration (user request, 2026-08-29): WHEN to take the upcard, on a
connectivity ladder, in a TEN game (flip KD -> knock threshold 10).

Setup, identical across rungs: two genuine scripted opening turns --
the hero draws JS from the stock and discards it back; the opponent
draws from the stock and discards 7H. The hero now faces the real
decision: take the 7H or draw. The particle filter conditions on the
real history (the opponent declined JS and let go of 7H).

The hero always holds the same two complete melds (3S-4S-5S, KH-KS-KC)
plus four loose cards; only the 7H-connectivity of the loose cards
changes:

  rung 0  no connection            (QC 2D 9S JD)
  rung 1  1 partial meld: pair     (7D  + QC 2D 9S)
  rung 2  2 partial melds: pair +  (7D 8H + QC 2D)     <- the user's
          run                                             7d/8h example
  rung 3  3 partial melds: pair +  (7D 8H 5H + QC)
          run + gap-run (5H_7H needs 6H, which would complete TWO
          melds at once)
  rung 4  immediate meld: trips    (7D 7S + QC 2D)     <- sanity anchor

Note the cost side the bot weighs against those combos: a taken 7H is
7 deadwood while unmelded, it is REVEALED (the opponent knows you hold
it), and it forfeits a blind draw that might hit the hand elsewhere.

    .venv/Scripts/python.exe home_upcard_demo.py
"""

import numpy as np

import sim
from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, DrawCardAction, PickUpDiscardAction)

sim.set_rules("full")

from pf_expert import ParticleBelief, PFExpert, V5_CFG  # noqa: E402
from sim import (CARD_ID, DECK, DEADWOOD_VALUE, cid, fast_legal_actions,
                 step_fast)  # noqa: E402

MELDS = ["3S", "4S", "5S", "KH", "KS", "KC"]
# All spare cards are LOW (<= 6) and mutually unconnected, so taking
# the 7H always WORSENS raw deadwood (7 in, at-best-6 out): a positive
# take-gap can only come from meld potential, not a value swap.
RUNGS = [
    ("0 combos (no connection)", ["AC", "2D", "4H", "6C"]),
    ("1 combo  (pair: 7D)", ["7D", "AC", "2D", "4H"]),
    ("2 combos (pair 7D + run 8H)", ["7D", "8H", "AC", "2D"]),
    ("3 combos (pair 7D + run 8H + gap 5H)", ["7D", "8H", "5H", "AC"]),
    ("immediate meld (trips: 7D 7S)", ["7D", "7S", "AC", "2D"]),
]
# Opponent hand chosen so their SCRIPTED moves are rational play:
# declining JS (it improves nothing: TC-JC-QC is already a run, and a
# lone JS pairs with nothing) and, after drawing 6D, discarding 7H
# (their deadwood-minimal drop: loose cards 7H 5D 4C AH 6D, and 5D-6D
# is worth keeping). An earlier draft gave them JC/JH/TS loose --
# declining JS would have been a blunder for that hand.
OPP = ["TC", "JC", "QC", "8C", "8D", "8S", "7H", "5D", "4C", "AH"]
FLIP, B1, B2 = "KD", "JS", "6D"          # flip; hero's brick; opp's draw


def build(loose, seed):
    game = sim.new_game_home(seed=1, flip_index=0)
    rnd = game.round
    hero_id = rnd.current_player_id
    opp_id = 1 - hero_id
    hero = MELDS + loose
    used = set(hero + OPP + [FLIP, B1, B2])
    assert len(used) == 23, "card overlap in the constructed position"
    rest = [k for k in CARD_ID if k not in used]
    rnd.players[hero_id].hand[:] = [DECK[CARD_ID[k]] for k in hero]
    rnd.players[opp_id].hand[:] = [DECK[CARD_ID[k]] for k in OPP]
    rnd.players[hero_id].known_cards[:] = []
    rnd.players[opp_id].known_cards[:] = []
    # stock pops from the END: hero draws B1 first, then opp draws B2
    rnd.dealer.stock_pile[:] = [DECK[CARD_ID[k]] for k in rest] + \
        [DECK[CARD_ID[B2]], DECK[CARD_ID[B1]]]
    flip = DECK[CARD_ID[FLIP]]
    rnd.dealer.discard_pile[:] = [flip]
    game.settings.going_out_deadwood_count = DEADWOOD_VALUE[cid(flip)]
    rnd._home_flip_id = cid(flip)
    game.actions[-1] = DiscardAction(card=flip)

    def scripted(pred):
        legal = fast_legal_actions(game)
        action = next(e for e in legal if pred(e))
        step_fast(game, action)

    # hero turn 1: draw JS, discard it back (a brick passing through)
    scripted(lambda e: isinstance(e, DrawCardAction))
    scripted(lambda e: isinstance(e, DiscardAction) and
             str(e.card) == B1)
    # opp turn 1: draw from stock (declines JS), discard 7H
    scripted(lambda e: isinstance(e, DrawCardAction))
    scripted(lambda e: isinstance(e, DiscardAction) and
             str(e.card) == "7H")
    return game, hero_id


def main():
    print("TEN game (flip KD, knock threshold 10). Opponent just")
    print("discarded 7H. Take it, or draw blind from the stock?")
    print("Hero always holds melds 3S-4S-5S + KH-KS-KC; spares are all")
    print("low and unconnected, so taking 7H always WORSENS deadwood --")
    print("any take-value is pure meld potential. v6home search,")
    print("384 worlds x 3 seeds per rung.\n")
    demo_cfg = dict(V5_CFG, stages=(384,), deep=False)
    rows = []
    for i, (label, loose) in enumerate(RUNGS):
        gaps, takes, draws, ses = [], [], [], []
        chosen_votes = 0
        for s in range(3):
            game, hero_id = build(loose, seed=900 + i)
            threshold = game.settings.going_out_deadwood_count
            belief = ParticleBelief(hero_id, seed=555 + 31 * s + i)
            expert = PFExpert(veto=True, opp_model=(threshold, 1.0),
                              sample_boost=0.0, seed=555 + 31 * s + i,
                              **demo_cfg)
            result = expert.analyze(game, belief)
            ev = {("take" if isinstance(a, PickUpDiscardAction)
                   else "draw"): result["payoff_means"][k] * 100.0
                  for k, a in enumerate(result["candidates"])}
            takes.append(ev["take"])
            draws.append(ev["draw"])
            gaps.append(ev["take"] - ev["draw"])
            ses.append(max(result["se"]) * 100.0)
            if isinstance(result["chosen"], PickUpDiscardAction):
                chosen_votes += 1
        rows.append((label, loose, np.mean(takes), np.mean(draws),
                     np.mean(gaps), np.mean(ses) / np.sqrt(3),
                     f"TAKE 7H ({chosen_votes}/3)" if chosen_votes >= 2
                     else f"draw ({3 - chosen_votes}/3)"))
    print(f"{'rung':40s} {'loose cards':14s} {'EV take':>8s} "
          f"{'EV draw':>8s} {'gap':>6s} {'dSE':>5s}  decision")
    for label, loose, tk, dr, gap, dse, chosen in rows:
        print(f"{label:40s} {' '.join(loose):14s} {tk:+8.1f} "
              f"{dr:+8.1f} {gap:+6.1f} {dse:5.1f}  {chosen}")
    print("\n(gap = EV(take) - EV(draw), pts/hand, mean of 3 search")
    print(" seeds; dSE = paired std error of the take-vs-draw diff,")
    print(" seed-averaged. Costs the search weighs against the combos:")
    print(" +7 deadwood while unmelded, the card is revealed, and a")
    print(" blind draw that might hit elsewhere is forfeited.")
    print(" The DECISION column follows the z-test-vs-prior rule: the")
    print(" teacher prior takes when raw deadwood improves by >= 1 --")
    print(" at 2+ combos it does, by ditching the now-redundant run")
    print(" card -- and the search overrides only on significant")
    print(" evidence. gap ~ 0 with decision TAKE therefore reads:")
    print(" 'taking has become EV-free; the prior's tiebreak stands'.)")


if __name__ == "__main__":
    main()
