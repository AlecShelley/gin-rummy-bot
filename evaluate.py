"""
The evaluator: feed a position, get the bot's move and both players' EV.

Chess-engine-style analysis of a single gin rummy decision, driven by the
v3search architecture (belief-weighted determinized worlds, paired rollouts,
teacher prior + z-test). For every candidate action it reports the expected
final full-rules score of the hand (hero perspective, in points), the paired
comparison against the teacher prior, and whether the search's deviation
test fires. Zero-sum: opponent EV = -hero EV.

Position file format (lines in any order, '#' comments, cards like AS TD 9H;
10 also accepted for T):

    me: AS 2S 3S 7H 8H 9H KD KC 5C 2D     # 10 cards = draw decision,
                                           # 11 = discard decision
    opp-known: 5H 6H                       # cards opponent picked up (visible)
    pile: 4C 7C QD                         # discard pile, bottom -> top
    picked: QD          # optional: the card I just took from the pile this
                        # turn (11-card positions; excludes it from `pile`,
                        # forbids discarding it back)
    my-known: KD        # optional: my cards the opponent has seen me pick up
    event: opp-discard 7C   # optional, chronological; sharpens hand-reading
    event: opp-decline 4C   # opponent drew from stock, declining upcard 4C
    event: opp-pickup 5H
    event: my-draw          # optional: our own moves, for a full transcript
    event: my-discard 7C    # (my-* events feed the display; only opp-*
    event: my-pickup 5D     #  events shape the opponent-hand belief)
    opp-model: 10,1.0   # optional: opponent knocks at deadwood<=10 with p=1
                        # (DEFAULT since 2026-08-29: knock-max,1.0 -- the
                        # promoted bot's model; 'opp-model: none' disables)
    boost: 2.5          # optional: meld-correlated world sampling (only
                        # used when the classic belief is active)

    .venv/Scripts/python.exe evaluate.py position.txt
    .venv/Scripts/python.exe evaluate.py position.txt --worlds 192
    .venv/Scripts/python.exe evaluate.py --example > position.txt

Stock size is derived (52 - mine - 10 - pile). The unknown part of the
opponent's hand and the stock order are sampled from the belief exactly as in
play; EVs assume both players continue with the teacher policy (the
opponent's seat knocks per opp-model when one is given).

Since 2026-08-29 the belief defaults to the v6 PARTICLE POSTERIOR
(pf_expert.ParticleBelief) whenever the position carries a complete
transcript, falling back to the classic per-card weights otherwise --
both report which one was used. This closes the documented v7 open item
(evaluator lagging the champion's belief is how the 6C blind spot
happened).
"""

import argparse
import sys
import types

import numpy as np

import sim
from belief import Belief
from expert import RolloutExpert
from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, DrawCardAction, GinAction, KnockAction,
    PickUpDiscardAction)
from sim import CARD_ID, DECK, cid, ids_of, min_deadwood_ids

SUIT_SYM = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}


def parse_card(tok):
    t = tok.strip().upper().replace("10", "T")
    if len(t) == 2 and t[1] in "SHDC":          # rank+suit: AS, TD
        rank, suit = t[0], t[1]
    elif len(t) == 2 and t[0] in "SHDC":        # suit+rank: SA
        suit, rank = t[0], t[1]
    else:
        raise ValueError(f"can't parse card {tok!r}")
    key = rank + suit
    if key not in CARD_ID:
        raise ValueError(f"unknown card {tok!r}")
    return DECK[CARD_ID[key]]


def show_card(card):
    return card.rank.replace("T", "10") + SUIT_SYM[card.suit]


def show_cards(cards):
    return " ".join(show_card(c) for c in cards)


def parse_position(text):
    pos = {"me": [], "opp_known": [], "pile": [], "picked": None,
           "my_known": [], "events": [], "opp_model": None,
           "boost": "auto",   # stage-ramped realistic worlds; boost: 0 reverts
           "knock_max": 10}   # home variant: the flip's value
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip().lower()
        rest = rest.strip()
        if key == "me":
            pos["me"] = [parse_card(t) for t in rest.split()]
        elif key == "opp-known":
            pos["opp_known"] = [parse_card(t) for t in rest.split()]
        elif key == "pile":
            pos["pile"] = [parse_card(t) for t in rest.split()]
        elif key == "picked":
            pos["picked"] = parse_card(rest)
        elif key == "my-known":
            pos["my_known"] = [parse_card(t) for t in rest.split()]
        elif key == "event":
            parts = rest.split()
            kind = parts[0].lower()
            assert kind in ("opp-discard", "opp-decline", "opp-pickup",
                            "my-discard", "my-pickup", "my-draw",
                            "opp-draw", "my-pass", "opp-pass"), \
                f"unknown event {kind!r}"
            card = parse_card(parts[1]) if len(parts) > 1 else None
            assert card is not None or kind in ("my-draw", "opp-draw",
                                                "my-pass", "opp-pass"), \
                f"event {kind} needs a card"
            pos["events"].append((kind, card))
        elif key == "opp-model":
            if rest.lower() == "none":
                pos["opp_model"] = "none"    # explicit opt-out
            else:
                t, p = rest.split(",")
                pos["opp_model"] = (int(t), float(p))
        elif key == "boost":
            pos["boost"] = "auto" if rest.lower() == "auto" else float(rest)
        elif key == "flip":
            # home variant: the founding flip card. Needed whenever the
            # flip is NO LONGER the pile bottom (someone picked it up);
            # without it the replay seeds the pile with the wrong card.
            pos["flip"] = parse_card(rest)
        elif key == "knock-max":
            pos["knock_max"] = int(rest)
            pos["home"] = True           # explicit: works for a 10-flip too
            assert 1 <= pos["knock_max"] <= 10, "knock-max must be 1..10"
        else:
            raise ValueError(f"unknown line {raw!r}")
    return pos


def validate(pos):
    me, pile, opp_known = pos["me"], pos["pile"], pos["opp_known"]
    if len(me) not in (10, 11):
        raise ValueError(f"'me' must have 10 or 11 cards, got {len(me)}")
    if len(me) == 10 and not pile:
        raise ValueError("a 10-card (draw) position needs a non-empty pile "
                         "(the upcard you could take)")
    if pos["picked"] and len(me) != 11:
        raise ValueError("'picked' only makes sense with an 11-card hand")
    if pos["picked"] and cid(pos["picked"]) not in ids_of(me):
        raise ValueError("'picked' card must be in your hand")
    seen = {}
    for zone, cards in (("me", me), ("pile", pile),
                        ("opp-known", opp_known)):
        for c in cards:
            k = cid(c)
            if k in seen:
                raise ValueError(f"{show_card(c)} appears in both "
                                 f"{seen[k]} and {zone}")
            seen[k] = zone
    if len(opp_known) > 10:
        raise ValueError("opponent can't be known to hold > 10 cards")
    if not set(ids_of(pos["my_known"])) <= set(ids_of(me)):
        raise ValueError("'my-known' must be a subset of 'me'")
    stock = 52 - len(me) - 10 - len(pile)
    if stock < 0:
        raise ValueError("card count impossible (52 - mine - 10 - pile < 0)")
    return stock


def history_check(pos):
    """Replay the event transcript against the declared pile (user rule
    2026-08-29: positions should ALWAYS carry a full move history, or
    the verdicts cannot be audited). Returns (status, problems):
    status 'complete' iff the replayed pile reproduces `pile:` exactly
    and every pickup/decline matched the replayed top."""
    pile = []
    if pos.get("home"):
        flip = pos.get("flip") or (pos["pile"][0] if pos["pile"] else None)
        if flip is not None:
            pile.append(flip)                # the flip founds the pile
    problems = []
    for kind, card in pos["events"]:
        name = show_card(card) if card is not None else "?"
        if kind in ("my-discard", "opp-discard"):
            pile.append(card)
        elif kind in ("my-pickup", "opp-pickup"):
            if not pile:
                problems.append(f"{kind} {name}: replayed pile is empty")
            elif cid(pile[-1]) != cid(card):
                problems.append(f"{kind} {name}: replayed pile top is "
                                f"{show_card(pile[-1])}")
                pile.pop()
            else:
                pile.pop()
        elif kind == "opp-decline":
            if not pile or cid(pile[-1]) != cid(card):
                problems.append(
                    f"opp-decline {name}: replayed pile top is "
                    f"{show_card(pile[-1]) if pile else '(empty)'}")
        elif kind == "opp-draw":
            if pile:
                problems.append(
                    f"opp-draw: replayed pile is not empty (top is "
                    f"{show_card(pile[-1])}) -- when an upcard was "
                    f"declined, use an opp-decline event naming it")
        elif kind in ("my-pass", "opp-pass"):
            # opening pass (home rule 2026-09-01): legal only in a home
            # game while the flip is still the whole pile
            if not pos.get("home"):
                problems.append(f"{kind}: opening passes exist only in "
                                f"the home variant (add knock-max:)")
            elif len(pile) != 1:
                problems.append(f"{kind}: an opening pass must happen "
                                f"while the flip is the whole pile")
    if pos["picked"] is not None:            # current-turn pile pickup
        if pile and cid(pile[-1]) == cid(pos["picked"]):
            pile.pop()
        else:
            problems.append(f"picked {show_card(pos['picked'])}: not the "
                            f"replayed pile top")
    if [cid(c) for c in pile] != [cid(c) for c in pos["pile"]]:
        problems.append(
            f"replayed pile [{show_cards(pile)}] != declared "
            f"[{show_cards(pos['pile'])}] -- transcript is missing "
            f"discards or misordered")
    return ("complete" if not problems else "partial"), problems


class FilterEnsemble:
    """R independent particle-filter replications of one transcript,
    duck-typing the single belief the search consumes.

    Why (caught 2026-08-29, pos_take1_spec): a single 256-particle
    filter on a SHORT history carries spurious per-card structure from
    its random initialization (reads of 1.2-1.7x uniform on arbitrary
    cards; verified: the top-read cards differ completely across filter
    seeds while n_eff stays ~200). All worlds sampled from one filter
    inherit the same tilt, which shifts candidate EV diffs coherently
    -- the take1 verdict swung from 'draw +2.70 +/- 1.27 significant'
    to 'draw -2.98 +/- 1.29' purely with the filter seed, a ~4.5-SE
    flip the world-level z-test cannot see. Worlds drawn from the
    MIXTURE of R independent filters shrink that filter-level noise by
    ~sqrt(R), and the ensemble-mean marginal stops presenting
    initialization noise as hand-reading signal."""

    def __init__(self, beliefs):
        self.beliefs = beliefs

    def update(self, game):
        for b in self.beliefs:
            b.update(game)

    def sample_free_cards(self, unseen_ids, num_hidden, rng, boost=0.0):
        b = self.beliefs[int(rng.integers(len(self.beliefs)))]
        return b.sample_free_cards(unseen_ids, num_hidden, rng, boost)

    def marginals(self, unseen_ids, num_hidden, known_ids=()):
        stack = [b.marginals(unseen_ids, num_hidden, known_ids)
                 for b in self.beliefs]
        return np.mean(stack, axis=0)

    @property
    def opp_discarded(self):
        return self.beliefs[0].opp_discarded

    @property
    def opp_declined(self):
        return self.beliefs[0].opp_declined

    @property
    def hero_discarded(self):
        return self.beliefs[0].hero_discarded

    @property
    def opp_pickups(self):
        return self.beliefs[0].opp_pickups

    @property
    def opp_declines(self):
        return self.beliefs[0].opp_declines

    def raw_marginals(self, unseen_ids, num_hidden, known_ids=()):
        """The UNSHRUNK ensemble particle marginal -- the distribution
        the search actually samples worlds from. The display must use
        this, not shrunk_marginal: with few distinct particles the
        shrinkage collapses a near-certain read onto the classic
        belief (the hidden-6S bug, 2026-08-30)."""
        m = np.mean([b.core.marginal52() for b in self.beliefs], axis=0)
        p = np.zeros(52, dtype=np.float32)
        if unseen_ids:
            p[list(unseen_ids)] = np.minimum(1.0, m[list(unseen_ids)])
        for i in known_ids:
            p[i] = 1.0
        return p


def build_particle_belief(pos, game, seed=0, pf_cfg=None):
    """The v6 particle posterior for a position (standing rule
    2026-08-29: the evaluator must run the same belief as the promoted
    champion). Requires a COMPLETE transcript -- the filter is
    trajectory-level, so it cannot be seeded from a summary the way the
    classic per-card weights can. Returns (ParticleBelief|None, note);
    None means fall back to the classic belief, with the reason."""
    from particle import DEFAULT_PF, ParticleCore, make_model
    from pf_expert import ParticleBelief

    if pos["history_status"] != "complete":
        return None, "classic weights (transcript incomplete -- the " \
                     "particle filter needs a full replayable history)"
    events = []
    src_events = list(pos["events"])
    # Legacy home transcripts predate the opening rule (2026-09-01): a
    # first-move stock draw while the flip led the pile implies the
    # full pass sequence (me-pass, opp-pass, forced draw) -- inject the
    # opponent's pass so its decline evidence isn't lost.
    if pos.get("home") and src_events and \
            not any(k in ("my-pass", "opp-pass") for k, _ in src_events):
        if src_events[0][0] == "my-draw":
            src_events.insert(0, ("opp-pass", None))
    for kind, card in src_events:
        if kind == "opp-discard":
            events.append(("opp_discard", cid(card)))
        elif kind == "my-pass":
            events.append(("hero_pass", None))
        elif kind == "opp-pass":
            events.append(("opp_pass", None))
        elif kind in ("opp-decline", "opp-draw"):
            events.append(("opp_draw", None))    # decline is implicit;
            # bare opp-draw = stock draw with an EMPTY pile (possible in
            # home games once the flip has been taken)
        elif kind == "opp-pickup":
            events.append(("opp_pickup", cid(card)))
        elif kind == "my-draw":
            if card is None:
                return None, "classic weights (a my-draw event lacks " \
                             "its card -- particle replay impossible)"
            events.append(("hero_draw", cid(card)))
        elif kind == "my-discard":
            events.append(("hero_discard", cid(card)))
        elif kind == "my-pickup":
            events.append(("hero_pickup", cid(card)))
    if pos["picked"] is not None:
        events.append(("hero_pickup", cid(pos["picked"])))

    # Unwind the hero events to recover the dealt hand.
    hero0 = set(ids_of(pos["me"]))
    for kind, card in reversed(events):
        if kind in ("hero_draw", "hero_pickup"):
            hero0.discard(card)
        elif kind == "hero_discard":
            hero0.add(card)
    home = bool(pos.get("home"))
    if home:
        if len(hero0) != 10:
            return None, "classic weights (home deal is 10+10 but the " \
                         f"transcript unwinds to {len(hero0)} dealt cards)"
    elif len(hero0) not in (10, 11):
        return None, "classic weights (transcript unwinds to " \
                     f"{len(hero0)} dealt cards; expected 10 or 11)"
    opp_size = (20 if home else 21) - len(hero0)
    flip_id = None
    if home:
        if pos.get("flip") is not None:
            flip_id = cid(pos["flip"])
        elif pos["pile"]:
            flip_id = cid(pos["pile"][0])

    cfg = dict(DEFAULT_PF, **(pf_cfg or {}))
    model = make_model(cfg, knock_max=pos["knock_max"])
    rng = np.random.default_rng(seed)
    core = ParticleCore(hero0, opp_size, rng, model=model,
                        n_particles=cfg["n_particles"],
                        kdraw=cfg["kdraw"],
                        pile0=[flip_id] if flip_id is not None else None,
                        mh_all_takes=cfg.get("mh_all_takes", False))
    for ev in events:
        core.apply_event(*ev)

    # The replayed filter must land exactly on the declared position;
    # anything else means the transcript and the position disagree in a
    # way history_check cannot see (it only audits the pile).
    hidden_n = 10 - len(pos["opp_known"])
    checks = [
        (core.h == hidden_n,
         f"filter tracks {core.h} hidden opp cards, position says "
         f"{hidden_n}"),
        (core.hero_hand == set(ids_of(pos["me"])),
         "replayed hero hand differs from 'me'"),
        (core.pile == ids_of(pos["pile"]),
         "replayed filter pile differs from 'pile'"),
        (core.opp_known == set(ids_of(pos["opp_known"])),
         "replayed opp pickups differ from 'opp-known'"),
    ]
    for ok, why in checks:
        if not ok:
            return None, f"classic weights (transcript mismatch: {why})"

    pb = ParticleBelief(hero_id=0, seed=seed)
    pb.core = core
    pb.rng = rng
    pb._pile = ids_of(pos["pile"])
    pb._processed = len(game.round.move_sheet)
    return pb, (f"particle filter (v10 posterior, "
                f"{cfg['n_particles']} particles from the full transcript)")


def build_state(pos):
    """A game object + Belief representing the described position, hero =
    player 0 to move. Only ever drive it through fast_legal_actions /
    the expert (never the rlcard judge)."""
    stock_n = validate(pos)
    game = sim.new_game(seed=0)
    rnd = game.round
    hero, opp = rnd.players[0], rnd.players[1]

    used = set(ids_of(pos["me"]) + ids_of(pos["pile"])
               + ids_of(pos["opp_known"]))
    unseen = [DECK[i] for i in range(52) if i not in used]
    hidden_n = 10 - len(pos["opp_known"])
    assert stock_n + hidden_n == len(unseen)

    hero.hand[:] = list(pos["me"])
    hero.known_cards[:] = list(pos["my_known"])
    # Placeholder hidden cards: the expert resamples them from the belief, so
    # which ones sit here never matters -- only their count does.
    opp.hand[:] = list(pos["opp_known"]) + unseen[:hidden_n]
    opp.known_cards[:] = list(pos["opp_known"])
    rnd.dealer.discard_pile[:] = list(pos["pile"])
    rnd.dealer.stock_pile[:] = unseen[hidden_n:]
    rnd.current_player_id = 0
    rnd.going_out_action = None
    rnd.going_out_player_id = None
    rnd.is_over = False
    game.settings.going_out_deadwood_count = pos["knock_max"]
    for player in (hero, opp):                    # scratch-game discipline
        player.add_card_to_hand = types.MethodType(
            lambda self, card: self.hand.append(card), player)
        player.remove_card_from_hand = types.MethodType(
            lambda self, card: self.hand.remove(card), player)

    # Phase, encoded the way fast_legal_actions reads it (game.actions[-1]).
    if len(pos["me"]) == 10:                      # draw decision
        game.actions[:] = [DiscardAction(card=pos["pile"][-1])]
        # Home OPENING (rule 2026-09-01): hero to move, no cards drawn
        # yet, pile = the bare flip -> this is a flip option, and the
        # pass mechanic must be live in rollouts. State 1 = hero holds
        # the FIRST option; an opp-pass event means hero holds the
        # SECOND (hero's pass then sends the first player to a forced
        # stock draw).
        if pos.get("home") and len(pos["pile"]) == 1 and not any(
                k in ("my-draw", "my-pickup", "opp-draw", "opp-pickup",
                      "my-discard", "opp-discard")
                for k, _ in pos["events"]):
            rnd._home_opening = 2 if any(
                k == "opp-pass" for k, _ in pos["events"]) else 1
    elif pos["picked"] is not None:               # discard after pile pickup
        game.actions[:] = [PickUpDiscardAction()]
        rnd.move_sheet.append(types.SimpleNamespace(card=pos["picked"]))
    else:                                         # discard after stock draw
        game.actions[:] = [DrawCardAction()]

    belief = Belief(hero_id=0)
    for kind, card in pos["events"]:
        if kind == "opp-discard":
            belief._evt_opp_discard(cid(card))
        elif kind == "opp-decline":
            belief._evt_decline(cid(card))
        elif kind == "opp-pickup":
            belief._evt_opp_pickup(cid(card))
        elif kind == "my-discard":
            belief.hero_discarded[cid(card)] = 1.0
        # my-pickup / my-draw: transcript only -- no belief impact
    belief._clamp()
    belief._processed = len(rnd.move_sheet)       # update() becomes a no-op
    belief._pile = ids_of(pos["pile"])
    return game, belief


def describe_action(action, game):
    from rlcard.games.gin_rummy.utils.action_event import (
        DeclareDeadHandAction)
    if isinstance(action, DeclareDeadHandAction):
        return "DECLARE DEAD HAND (0-0)"
    if isinstance(action, DrawCardAction):
        if getattr(game.round, "_home_opening", 0):
            return "PASS on the flip"
        return "draw from stock"
    if isinstance(action, PickUpDiscardAction):
        top = game.round.dealer.discard_pile[-1]
        if getattr(game.round, "_home_opening", 0):
            return f"take the flip ({show_card(top)})"
        return f"take {show_card(top)} from pile"
    if isinstance(action, DiscardAction):
        return f"discard {show_card(action.card)}"
    if isinstance(action, KnockAction):
        return f"KNOCK (discarding {show_card(action.card)})"
    if isinstance(action, GinAction):
        return "GIN"
    return str(action)


def hand_value_line(cards):
    ids = ids_of(cards)
    if len(ids) == 11:
        dw = min(min_deadwood_ids(ids[:k] + ids[k + 1:]) for k in range(11))
        return f"deadwood after best discard: {dw}"
    return f"deadwood: {min_deadwood_ids(ids)}"


def belief_read_parts(belief, game, top_n=8):
    """Shared by the text report and viz.py: (rows, base) where rows =
    [(card_id, probability)] sorted desc, filtered to reads that beat
    chance. Probabilities are the RAW ensemble particle marginal for
    particle beliefs (what the search samples from); percent units
    cannot saturate the way x-vs-chance did (base can be 0.78 late,
    capping a CERTAIN card at '1.3x' -- the hidden-read bug)."""
    rnd = game.round
    opp = rnd.players[1]
    unseen = ids_of(rnd.dealer.stock_pile) + \
        [cid(c) for c in opp.hand if c not in opp.known_cards]
    hidden_n = len(opp.hand) - len(opp.known_cards)
    if hidden_n == 0 or not unseen:
        return [], 1.0
    if hasattr(belief, "raw_marginals"):
        p = belief.raw_marginals(unseen, hidden_n)
    else:
        p = belief.marginals(unseen, hidden_n)
    base = hidden_n / len(unseen)
    rows = [(i, float(p[i])) for i in sorted(unseen, key=lambda i: -p[i])
            if p[i] > base * 1.05 or p[i] >= 0.85]
    return rows[:top_n], base


def belief_read(belief, game, top_n=8):
    rows, base = belief_read_parts(belief, game, top_n)
    if base >= 1.0 and not rows:
        return "opponent hand fully known"
    if not rows:
        return "no strong reads (their hidden cards look near-random)"
    sure = [r for r in rows if r[1] >= 0.90]
    likely = [r for r in rows if r[1] < 0.90]
    parts = []
    if sure:
        parts.append("almost certainly holding: " + ", ".join(
            f"{show_card(DECK[i])} ({p:.0%})" for i, p in sure))
    if likely:
        parts.append("likely: " + ", ".join(
            f"{show_card(DECK[i])} ({p:.0%})" for i, p in likely))
    return "; ".join(parts) + f"  [chance level {base:.0%}]"


def evaluate_position(pos, worlds=192, truncate=16, z=1.5, seed=0,
                      engine="deep", belief_mode="particle", filters=4,
                      explain_n=0, veto=False):
    """Analyze and print; returns (info, game, belief) for rendering.

    engine="deep" (default since 2026-08-29): the v5/v6-style search --
    every world's rollout branches the hero's FIRST next decision over
    the top-2 discards + best KNOCK (exact max within the determinized
    world). The old flat engine ("v4") plays patient-teacher rollouts
    that never knock before gin, which measurably undervalues branches
    whose payoff runs through a NEXT-TURN hero knock: on the take-vs-
    draw position pos_take2_swap.txt (11/39 draw outs enable an
    immediate knock, the take enables none) the flat engine scored the
    draw +0.74 +/- 0.35 while the deep engine scores it +4.50 +/- 0.39
    -- a 6x undervaluation, caught by the user's out-counting.
    """
    sim.set_rules("full")
    status, problems = history_check(pos)
    pos["history_status"] = status
    pos["history_problems"] = problems

    # Champion-default opponent model (2026-08-29): the promoted bot
    # ALWAYS plays with opp_model=(knock threshold, 1.0); an evaluator
    # that silently analyzes against a never-knocking opponent is the
    # same class of blind spot as the flat-engine 6C miss. 'opp-model:
    # none' in the position file opts out explicitly.
    if pos["opp_model"] is None:
        pos["opp_model"] = (pos["knock_max"], 1.0)
        pos["opp_model_default"] = True
    elif pos["opp_model"] == "none":
        pos["opp_model"] = None

    game, belief = build_state(pos)

    # Belief source (standing rule: same belief as the champion). The
    # v6 particle posterior needs a full transcript; fall back to the
    # classic weights, saying so, when it cannot be built.
    boost = pos["boost"]
    pf_active = False
    if belief_mode == "particle" and engine != "v4":
        pb, note = build_particle_belief(pos, game, seed)
        pos["belief_note"] = note
        if pb is not None:
            pf_active = True
            reps = [pb] + [
                build_particle_belief(pos, game, seed + 7919 * r)[0]
                for r in range(1, filters)]
            belief = FilterEnsemble([b for b in reps if b is not None])
            boost = 0.0          # particles are already correlated worlds
            pos["belief_note"] = note.replace(
                "particle filter", f"particle filter x{len(belief.beliefs)}"
                " independent replications")
    else:
        pos["belief_note"] = "classic weights (requested)" \
            if engine != "v4" else "classic weights (v4 engine)"

    if engine == "v4":
        expert = RolloutExpert(worlds=worlds, truncate=truncate, z=z,
                               opp_model=pos["opp_model"], seed=seed,
                               sample_boost=boost)
    elif pf_active:
        # No world veto on the particle path (2026-08-30): the hard
        # "opponent could knock => impossible world" filter discards
        # exactly the melded hands the pickup evidence proves (the
        # hidden-5S6S bug); the filter's own compounding no-knock
        # evidence already carries that information, and the no-veto
        # search variant match-gated as a tie. The knock model still
        # shapes the rollout continuation. --veto restores the filter.
        from pf_expert import PFExpert
        expert = PFExpert(veto=veto, stages=(worlds,), deep=True,
                          deep_from=0, kbranch=3, truncate=truncate,
                          z=z, opp_model=pos["opp_model"], seed=seed,
                          sample_boost=0.0)
    else:
        from adaptive import AdaptiveExpert
        expert = AdaptiveExpert(stages=(worlds,), deep=True,
                                deep_from=0, kbranch=3,
                                truncate=truncate, z=z,
                                opp_model=pos["opp_model"], seed=seed,
                                sample_boost=boost)
    info = expert.analyze(game, belief)

    # OPENING BOOK (6000-deal study, results/opening_report.log): at a
    # flip-option node the bot plays take-iff-it-melds; the search's
    # own EVs here carry the MEASURED truncation bias (deadwood gain
    # visible, tempo cost beyond the horizon), so the recommendation
    # follows the book. The candidate table still shows the search EVs.
    opening = getattr(game.round, "_home_opening", 0)
    if opening and info.get("candidates"):
        from sim import flip_book_take
        hand = ids_of(pos["me"])
        top = cid(pos["pile"][-1])
        take = flip_book_take(hand, top, opening)
        want = PickUpDiscardAction if take else DrawCardAction
        book = next((a for a in info["candidates"]
                     if isinstance(a, want)), None)
        if book is not None:
            if book.action_id != info["chosen"].action_id:
                info.pop("followup_desc", None)
            info["chosen"] = book
            rule = ("take iff it melds" if opening == 1 else
                    "take iff it melds OR adjacency >= 2 (second "
                    "player's take is an extra turn)")
            info["reason"] = (
                f"opening book: {rule} -- measured (6000-deal study + "
                "paired gates; opening-node search EVs overweight "
                "deadwood)")
    # Turn-2 clause (evaluator tracks the bot, 2026-09-02): at the
    # hand's FIRST free draw decision the champion takes iff the card
    # melds (non-melding takes there measured -2.06 +/- 1.64 each,
    # results/turn2_confirm.log; match gate +1.84 +/- 1.78). Applied
    # when the transcript proves no earlier free draw decision (no
    # my-draw / my-pickup events); a hero who took the flip earlier is
    # not covered by this conservative check and keeps the search's
    # answer.
    elif pos.get("home") and len(pos["me"]) == 10 and \
            pos["history_status"] == "complete" and \
            info.get("candidates") and \
            not any(k in ("my-draw", "my-pickup")
                    for k, _ in pos["events"]) and \
            pos["picked"] is None:
        from sim import t2_book_take
        hand = ids_of(pos["me"])
        top = cid(pos["pile"][-1])
        take = t2_book_take(hand, top, exception=True)
        want = PickUpDiscardAction if take else DrawCardAction
        book = next((a for a in info["candidates"]
                     if isinstance(a, want)), None)
        if book is not None:
            if book.action_id != info["chosen"].action_id:
                info.pop("followup_desc", None)
            info["chosen"] = book
            info["reason"] = (
                "turn-2 book: first free draw decision -- take iff it "
                "melds, or a low card (<=3) with two-way adjacency "
                "(non-melding takes otherwise measured -2 each; the "
                "low-card exception +3.6 under the natural conditional)")

    # A pickup is not a complete recommendation without the follow-up
    # discard (user directive 2026-08-29): apply the take on a scratch
    # copy and re-analyze the resulting 11-card node with the same
    # engine. (Not possible for a stock draw -- the card is unknown.)
    from rlcard.games.gin_rummy.utils.action_event import (
        PickUpDiscardAction as _Pickup)
    if isinstance(info.get("chosen"), _Pickup):
        import copy
        from belief_teacher import clone_belief
        from sim import make_scratch, step_fast
        scratch = make_scratch(game)
        step_fast(scratch, info["chosen"])
        bel2 = clone_belief(belief) if isinstance(belief, Belief) \
            else copy.deepcopy(belief)
        info2 = expert.analyze(scratch, bel2)
        desc = describe_action(info2["chosen"], scratch)
        # The follow-up node is often nearly indifferent (measured
        # 2026-08-29: ex5's follow-up landed on three different
        # discards across three seeds) -- say so instead of presenting
        # one arm of a coin-flip as the plan.
        if info2.get("reason") == "deviate" and "se" in info2:
            ck = next(k for k, a in enumerate(info2["candidates"])
                      if a.action_id == info2["chosen"].action_id)
            if info2["se"][ck] > 0 and \
                    info2["diff_means"][ck] / info2["se"][ck] < 2.5:
                desc += " (borderline -- close call)"
        elif info2.get("reason") == "prior" and "payoff_means" in info2:
            bk = int(max(range(len(info2["candidates"])),
                         key=lambda k: info2["payoff_means"][k]))
            if info2["candidates"][bk].action_id != \
                    info2["chosen"].action_id:
                desc += (" (statistical tie with "
                         f"{describe_action(info2['candidates'][bk], scratch)})")
        info["followup_desc"] = desc

    # Optional explanation pipeline (user directive 2026-08-30): full
    # two-sided playouts -> outcome distributions + critical next draws.
    if explain_n and engine != "v4" and "payoff_means" in info:
        from explain import run_explain
        b = boost if isinstance(boost, (int, float)) else \
            min(3.0, 0.5 * (belief.opp_pickups + belief.opp_declines))
        info["explain"] = run_explain(game, belief, info, expert, pos,
                                      n_worlds=explain_n, seed=seed,
                                      boost=b)

    hero_cards = sorted(game.round.players[0].hand, key=cid)
    print("=" * 66)
    print(f"Hero hand : {show_cards(hero_cards)}")
    print(f"            {hand_value_line(hero_cards)}")
    opp_known = game.round.players[1].known_cards
    print(f"Opp known : {show_cards(opp_known) if opp_known else '(none)'}")
    my_known = game.round.players[0].known_cards
    print(f"My known  : {show_cards(my_known) if my_known else '(none)'}"
          "  (pickups the opponent has seen)")
    pile = game.round.dealer.discard_pile
    print(f"Pile top  : {show_card(pile[-1]) if pile else '(empty)'}    "
          f"pile size: {len(pile)}    stock: "
          f"{len(game.round.dealer.stock_pile)}")
    if pos.get("home"):
        flip = pos.get("flip") or (pos["pile"][0] if pos["pile"] else None)
        print(f"Knock max : {pos['knock_max']} (home variant; flip = "
              f"{show_card(flip) if flip else '?'}"
              + ("" if pos.get("flip") is None or not pos["pile"]
                 or cid(pos["pile"][0]) == cid(flip)
                 else ", no longer in the pile") + ")")
    if pos["opp_model"]:
        t, p = pos["opp_model"]
        dflt = (" (default -- matches the promoted bot; 'opp-model: "
                "none' to disable)" if pos.get("opp_model_default") else "")
        print(f"Opp model : knocks at deadwood<={t} with p={p}{dflt}")
    else:
        print("Opp model : NONE (opponent never knocks in rollouts -- "
              "explicit opt-out)")
    from home_variant import CHAMPION
    print(f"Engine    : {CHAMPION} (champion engine + belief -- "
          f"evaluator lockstep)")
    print(f"Belief    : {pos.get('belief_note', 'classic weights')}")
    print(f"            {belief_read(belief, game)}")
    if pos["history_status"] == "complete":
        print("History   : COMPLETE (event replay reproduces the pile)")
    else:
        print("History   : *** PARTIAL/INCONSISTENT -- verdicts cannot "
              "be fully audited ***")
        for p in pos["history_problems"]:
            print(f"            - {p}")
    print("-" * 66)

    if info.get("reason") == "forced":
        print(f"Only one reasonable action: "
              f"{describe_action(info['chosen'], game)}")
        return info, game, belief

    chosen = info["chosen"]
    t_idx = info["t_idx"]
    order = np.argsort(-np.asarray(info["payoff_means"]))
    print(f"Candidates ({info['worlds']} worlds, T={truncate}, z={z}; "
          f"EV = expected final points of the hand, hero view):")
    for k in order:
        a = info["candidates"][k]
        mark = "*" if a.action_id == chosen.action_id else " "
        prior = "  (prior)" if k == t_idx else ""
        ev = info["payoff_means"][k] * 100.0
        d = info["diff_means"][k] * 100.0
        se = info["se"][k] * 100.0
        sig = ""
        if k != t_idx:
            sig = "  significant" if info["gain"][k] > 0 else ""
            sig_str = f"  vs prior {d:+6.2f} +/-{se:5.2f}{sig}"
        else:
            sig_str = ""
        print(f" {mark} {describe_action(a, game):<28} EV {ev:+7.2f}"
              f"{sig_str}{prior}")
    if info["worlds"] < worlds // 2:
        print(f"  NOTE: only {info['worlds']} of {worlds} requested "
              f"worlds survived the opponent-knock veto -- the "
              f"opp-model assumption is heavily constraining here; "
              f"'opp-model: none' shows the unvetoed read")
    print("-" * 66)
    ev = info["payoff_means"][
        next(k for k, a in enumerate(info["candidates"])
             if a.action_id == chosen.action_id)] * 100.0
    why = {"deviate": "search deviates from the prior (paired evidence "
                      f"significant at z={z})",
           "prior": "search keeps the teacher prior (no significant "
                    "improvement found)"}.get(info.get("reason"),
                                              info.get("reason", ""))
    print(f"RECOMMENDATION: {describe_action(chosen, game)}   [{why}]")
    if info.get("followup_desc"):
        print(f"  then: {info['followup_desc']}")
    # Verdict-confidence honesty (caught 2026-08-29, pos_take1_spec):
    # z=1.5 is the BOT's in-game deviation bar -- correct on average
    # over thousands of hands, but a single borderline crossing is
    # seed-sensitive (take1's 'draw +2.70 significant' at one seed was
    # 'draw -2.98' at another; the pooled truth is a ~+1 near-tie). An
    # analysis verdict must say which kind of 'significant' it is.
    if info.get("reason") == "deviate":
        ck = next(k for k, a in enumerate(info["candidates"])
                  if a.action_id == chosen.action_id)
        z_val = (info["diff_means"][ck] / info["se"][ck]
                 if info["se"][ck] > 0 else 99.0)
        if z_val < 2.5:
            print(f"  (borderline deviation, z={z_val:.1f}: the in-game "
                  f"rule fires at z=1.5, but at this margin the verdict "
                  f"is seed-sensitive -- treat as a close call and "
                  f"confirm with --worlds 384 / other --seed values "
                  f"before trusting it over the prior)")
    if info.get("reason") == "prior" and "payoff_means" in info:
        best_k = int(max(range(len(info["candidates"])),
                         key=lambda k: info["payoff_means"][k]))
        if info["candidates"][best_k].action_id != chosen.action_id:
            d = info["diff_means"][best_k] * 100.0
            se = info["se"][best_k] * 100.0
            print(f"  (statistical tie: "
                  f"{describe_action(info['candidates'][best_k], game)} "
                  f"shows a higher raw mean, but only {d:+.2f} +/- "
                  f"{se:.2f} vs the prior -- noise at this resolution; "
                  f"the prior's rule breaks ties)")
    print(f"Hand EV: hero {ev:+.2f} points, opponent {-ev:+.2f} points")
    if info.get("explain"):
        _print_explain(info, game)
    print("=" * 66)
    return info, game, belief


def _print_explain(info, game):
    from explain import card_rows, opp_card_rows, outcome_rows
    ex = info["explain"]
    print("-" * 66)
    print(f"WHY -- each line played out to the end {ex['worlds']} times "
          f"(both sides")
    print(f"played by the engine's continuation model"
          + (f"; {ex['vetoed']} worlds skipped by the opponent-knock "
             f"filter)" if ex["vetoed"] else ")"))
    chosen_id = info["chosen"].action_id
    for k in ex["picks"]:
        a = info["candidates"][k]
        s = ex["stats"][k]
        tag = "  <-- RECOMMENDED" if a.action_id == chosen_id else ""
        avg = s["sum"] / max(s["n"], 1)
        print(f"  {describe_action(a, game)}   (EV {avg:+.1f}){tag}")
        for label, pct, apts, cnt in outcome_rows(s):
            print(f"     {pct:5.1f}%  {label:<30} "
                  f"conditional EV {apts:+7.1f}")
        good, bad = card_rows(s)
        if good or bad:
            print(f"     key next draws after this move:")
            hold = ex.get("opp_hold", {})
            for c, note, apts, cnt, fed in good:
                if c in hold:
                    note = (note + "; " if note else "") + \
                        f"likely in their hand ({hold[c]:.0%})"
                n = f" ({note})" if note else ""
                print(f"        helps: {show_card(DECK[c]):<4}{n:<28} "
                      f"conditional EV {apts:+6.1f} over {cnt}")
            for c, note, apts, cnt, fed in bad:
                extra = note or "no help"
                if fed >= 25:
                    extra += f"; they took our discard {fed:.0f}% of the time"
                print(f"        hurts: {show_card(DECK[c]):<4}({extra}) "
                      f"conditional EV {apts:+6.1f} over {cnt}")
        root_c = cid(a.card) if isinstance(a, DiscardAction) else None
        danger, took = opp_card_rows(s, root_c)
        if took is not None:
            pct = 100.0 * took[2] / max(s["n"], 1)
            print(f"     they take our {show_card(DECK[took[0]])} "
                  f"{pct:.0f}% of the time; conditional EV "
                  f"{took[1]:+.1f}")
        if danger:
            print(f"     cards that hurt us if THEY get them first:")
            for c, apts, cnt in danger:
                print(f"        {show_card(DECK[c]):<4} "
                      f"conditional EV {apts:+6.1f} over {cnt}")


EXAMPLE = """\
# Gin rummy position -- edit and run: python evaluate.py position.txt
# 'me' is whoever is to move: to analyze the other player's decision,
# re-enter the position from their point of view.
me: AS 2S 3S 7H 8H 9H KD KC 5C 2D
opp-known: 5H 6H
pile: 4C 7C QD
event: opp-discard 7C
event: opp-decline 4C
event: opp-pickup 5H
# picked: QD          # uncomment for an 11-card hand you just picked into
# opp-model: 10,1.0   # opponent knocks whenever deadwood <= 10
# knock-max: 5        # home variant: knock threshold = the flipped card's
#                     # value (put the flip at the bottom of pile:)
# flip: 5H            # home variant: name the flip explicitly whenever it
#                     # was picked up (i.e. is no longer the pile bottom)
# boost: 0            # world realism: default "auto" ramps with the
#                     # opponent's turn count; a number fixes it; 0 disables
"""


def main():
    ap = argparse.ArgumentParser("gin rummy position evaluator")
    ap.add_argument("position", nargs="?", help="position file (see --example)")
    ap.add_argument("--worlds", type=int, default=192,
                    help="worlds (default 192 = the champion's final "
                         "search stage)")
    ap.add_argument("--truncate", type=int, default=16)
    ap.add_argument("--z", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", choices=("deep", "v4"), default="deep",
                    help="deep (default) = 2-ply knock-aware rollouts; "
                         "v4 = the historical flat engine")
    ap.add_argument("--belief", choices=("particle", "classic"),
                    default="particle",
                    help="particle (default) = the v6 posterior, needs a "
                         "full transcript (auto-falls back to classic); "
                         "classic = the v3-v5 per-card weights")
    ap.add_argument("--filters", type=int, default=4,
                    help="independent particle-filter replications; "
                         "worlds come from the mixture (verdict-stability "
                         "insurance against filter Monte-Carlo noise)")
    ap.add_argument("--veto", action="store_true",
                    help="restore the opponent-knock world filter on the "
                         "particle path (off by default since 2026-08-30: "
                         "it discards worlds the pickup evidence supports)")
    ap.add_argument("--explain", action="store_true",
                    help="run the explanation pipeline: full two-sided "
                         "playouts -> outcome percentages (gin / clean "
                         "knock / undercut / dead hand, with average "
                         "points) and the critical next draws, for the "
                         "recommended move and its main rivals (slower)")
    ap.add_argument("--playouts", type=int, default=600,
                    help="playouts per candidate line for --explain")
    ap.add_argument("--example", action="store_true",
                    help="print a template position file and exit")
    ap.add_argument("--html", nargs="?", const="", default=None,
                    help="also write a visual report (default: "
                         "<position>.html)")
    ap.add_argument("--open", action="store_true",
                    help="open the HTML report in the browser")
    args = ap.parse_args()
    if args.example:
        print(EXAMPLE, end="")
        return
    if not args.position:
        ap.error("give a position file, or --example for a template")
    with open(args.position, encoding="utf-8-sig") as fh:
        pos = parse_position(fh.read())
    info, game, belief = evaluate_position(pos, args.worlds, args.truncate,
                                           args.z, args.seed, args.engine,
                                           args.belief, args.filters,
                                           args.playouts if args.explain
                                           else 0, args.veto)
    if args.html is not None or args.open:
        import os
        from viz import render_html
        out = args.html or os.path.splitext(args.position)[0] + ".html"
        from home_variant import CHAMPION
        import datetime
        cfg = {"worlds": args.worlds, "truncate": args.truncate,
               "z": args.z,
               "engine_version": f"{CHAMPION} evaluator, "
               f"{datetime.date.today().isoformat()}"}
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(render_html(pos, game, belief, info, cfg))
        print(f"visual report -> {out}")
        if args.open:
            os.startfile(os.path.abspath(out))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
