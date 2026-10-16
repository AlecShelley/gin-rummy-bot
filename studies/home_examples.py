import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
Home-variant example-position generator (v7 session, 2026-08-29).

Plays REAL home games (sim.new_game_home) with rational policies on
both seats and cuts evaluator positions out of them, so every
transcript is rational and replay-consistent BY CONSTRUCTION (the two
standing user rules) -- no hand-constructed histories.

Hero = seat 0 (FastTeacher). Opponent rotates: patient teacher /
knock-whenever-legal / 50% knocker (thresholds = the flip, per home
rules). Every hero decision node is recorded with its full hero-view
event transcript plus selection metadata (turn, stock, deadwoods,
knock legality, upcard adjacency, opponent pickups); categories are
then filtered and written as pos_h*.txt files with a ground-truth
answer key (true opponent hand at the node) in results/.

    .venv/Scripts/python.exe home_examples.py --scan   # list candidates
    .venv/Scripts/python.exe home_examples.py --emit   # write files
"""

import argparse
import random

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, DrawCardAction, GinAction, KnockAction,
    PickUpDiscardAction)

import sim
from sim import (DECK, FastTeacher, MixedKnock, cid, fast_legal_actions,
                 ids_of, min_deadwood_ids, step_fast)

_ADJ = None  # lazy belief._ADJ


def card_tok(cid_):
    return str(DECK[cid_])                # rank+suit, e.g. "TS"


def toks(ids):
    return " ".join(card_tok(i) for i in sorted(ids))


def hand_value(ids):
    if len(ids) == 11:
        return min(min_deadwood_ids(ids[:k] + ids[k + 1:])
                   for k in range(11))
    return min_deadwood_ids(list(ids))


def play_and_record(seed):
    """One home game; returns list of hero-decision snapshots."""
    global _ADJ
    if _ADJ is None:
        from belief import _ADJ as adj
        _ADJ = adj
    game = sim.new_game_home(seed)
    rnd = game.round
    threshold = game.settings.going_out_deadwood_count
    flip_id = rnd._home_flip_id
    hero_pol = FastTeacher()
    kind = seed % 3
    if kind == 0:
        opp_pol = FastTeacher()
        opp_name = "patient teacher"
    elif kind == 1:
        opp_pol = MixedKnock(FastTeacher(), threshold, 1.0,
                             rng=random.Random(seed + 5))
        opp_name = "knock-when-legal"
    else:
        opp_pol = MixedKnock(FastTeacher(), threshold, 0.5,
                             rng=random.Random(seed + 5))
        opp_name = "50% knocker"

    events = []                           # (kind_str, card_id_or_None)
    snaps = []
    hero_decision_no = 0
    while rnd.going_out_action is None and not rnd.is_over:
        legal = fast_legal_actions(game)
        mover = rnd.current_player_id
        if mover == 0 and len(legal) > 1:
            hero_decision_no += 1
            hand = ids_of(rnd.players[0].hand)
            opp = rnd.players[1]
            pile = [cid(c) for c in rnd.dealer.discard_pile]
            up_adj = (len(set(_ADJ[pile[-1]]) & set(hand))
                      if pile and len(hand) == 10 else 0)
            snaps.append(dict(
                seed=seed, opp_name=opp_name, threshold=threshold,
                flip=flip_id, decision_no=hero_decision_no,
                events=list(events), hand=list(hand),
                opp_known=[cid(c) for c in opp.known_cards],
                my_known=[cid(c) for c in rnd.players[0].known_cards],
                pile=list(pile), stock=len(rnd.dealer.stock_pile),
                phase="draw" if len(hand) == 10 else "discard",
                knock_legal=any(isinstance(e, KnockAction)
                                for e in legal),
                gin_legal=any(isinstance(e, GinAction) for e in legal),
                hero_dw=hand_value(hand),
                opp_true=sorted(ids_of(opp.hand)),
                opp_dw=hand_value(ids_of(opp.hand)),
                opp_pickups=sum(1 for k, _ in events
                                if k == "opp-pickup"),
                up_adj=up_adj,
                last_event=events[-1] if events else None,
            ))
        action = legal[0] if len(legal) == 1 else \
            (hero_pol if mover == 0 else opp_pol).act(game, legal)
        pile_cards = rnd.dealer.discard_pile
        if isinstance(action, DrawCardAction):
            if mover == 0:
                events.append(("my-draw",
                               cid(rnd.dealer.stock_pile[-1])))
            elif pile_cards:
                events.append(("opp-decline", cid(pile_cards[-1])))
            else:
                events.append(("opp-draw", None))
        elif isinstance(action, PickUpDiscardAction):
            top = cid(pile_cards[-1])
            events.append(("my-pickup" if mover == 0 else "opp-pickup",
                           top))
        elif isinstance(action, DiscardAction):
            events.append(("my-discard" if mover == 0
                           else "opp-discard", cid(action.card)))
        step_fast(game, action)
    return snaps


def pos_text(snap, header):
    """One snapshot as evaluator position-file text."""
    lines = [header.rstrip(), ""]
    events = snap["events"]
    picked = None
    if snap["phase"] == "discard" and events and \
            events[-1][0] == "my-pickup":
        picked = events[-1][1]
        events = events[:-1]
    lines.append(f"me: {toks(snap['hand'])}")
    lines.append(f"opp-known: {toks(snap['opp_known'])}")
    lines.append("pile: " + " ".join(card_tok(i) for i in snap["pile"]))
    if picked is not None:
        lines.append(f"picked: {card_tok(picked)}")
    if snap["my_known"]:
        lines.append(f"my-known: {toks(snap['my_known'])}")
    for kind, card in events:
        lines.append(f"event: {kind}"
                     + (f" {card_tok(card)}" if card is not None else ""))
    lines.append(f"flip: {card_tok(snap['flip'])}")
    lines.append(f"knock-max: {snap['threshold']}")
    return "\n".join(lines) + "\n"


def emit(snap, path, header):
    """Write one snapshot as an evaluator position file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(pos_text(snap, header))


def scan(seeds):
    rows = []
    for seed in seeds:
        for s in play_and_record(seed):
            rows.append(s)
    return rows


def describe(s):
    return (f"seed {s['seed']:>5} d{s['decision_no']:>2} "
            f"[{s['opp_name']:>16}] flip={card_tok(s['flip'])} "
            f"thr={s['threshold']:>2} {s['phase']:<7} "
            f"stock={s['stock']:>2} dw={s['hero_dw']:>2} "
            f"oppdw={s['opp_dw']:>2} picks={s['opp_pickups']} "
            f"upadj={s['up_adj']} knock={'Y' if s['knock_legal'] else '-'}")


def main():
    ap = argparse.ArgumentParser("home example generator")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--base", type=int, default=910000)
    args = ap.parse_args()
    sim.set_rules("full")
    rows = scan(range(args.base, args.base + args.seeds))
    print(f"{len(rows)} hero decision nodes from {args.seeds} games")

    cats = {
        "t1_flip_take": lambda s: s["decision_no"] == 1
            and s["phase"] == "draw" and len(s["pile"]) == 1
            and s["up_adj"] >= 2 and 3 <= s["threshold"] <= 9,
        "early_discard": lambda s: s["decision_no"] == 2
            and s["phase"] == "discard" and s["last_event"][0] == "my-draw"
            and s["hero_dw"] >= 25,
        "spec_take": lambda s: s["phase"] == "draw"
            and 3 <= s["decision_no"] <= 6 and s["up_adj"] >= 3
            and not s["knock_legal"],
        "reading_discard": lambda s: s["phase"] == "discard"
            and s["opp_pickups"] >= 3 and s["hero_dw"] >= 8
            and s["stock"] >= 10 and not s["knock_legal"],
        "losing_late": lambda s: s["phase"] == "discard"
            and s["stock"] <= 16 and s["hero_dw"] - s["opp_dw"] >= 15
            and s["opp_pickups"] >= 2 and not s["knock_legal"],
        "endgame_draw": lambda s: s["phase"] == "draw"
            and s["stock"] <= 6,
        "endgame_discard": lambda s: s["phase"] == "discard"
            and s["stock"] <= 5 and not s["gin_legal"],
        "ace_race": lambda s: s["threshold"] == 1
            and s["phase"] == "discard" and 4 <= s["decision_no"] <= 8
            and s["opp_pickups"] >= 1,
        "knock_option": lambda s: s["knock_legal"]
            and not s["gin_legal"] and s["threshold"] >= 5,
    }
    for name, pred in cats.items():
        hits = [s for s in rows if pred(s)]
        print(f"--- {name}: {len(hits)} hits")
        for s in hits[:12]:
            print("   ", describe(s))


if __name__ == "__main__":
    main()
