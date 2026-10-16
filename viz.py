"""
HTML rendering for the position evaluator (evaluate.py --html).

Produces a single self-contained .html file aimed at the PLAYER, not
the engine developer (user directive 2026-08-30): a prominent
game-variant banner (the flipped card and knock threshold are
critically relevant), the opponent's known cards with a plain-language
hand read, OUR hand with the cards the opponent has seen us take
clearly marked (they will not feed those melds), the full turn
history (hero draws show the drawn card; the current turn is annotated
in place, never duplicated), and a verdict table with EVs and
plain-language confidence -- no model meta-information (no worlds/
z-test/belief-source jargon).  No JS, no external assets.
"""

import html

from rlcard.games.gin_rummy.utils import melding as rl_melding
from rlcard.games.gin_rummy.utils import utils as gr_utils
from rlcard.games.gin_rummy.utils.action_event import (
    DeclareDeadHandAction, DiscardAction, DrawCardAction, GinAction,
    KnockAction, PickUpDiscardAction)

from sim import cid, ids_of, min_deadwood_ids

SUIT_SYM = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
RED = ("H", "D")

CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       background: #f4f3ef; color: #1a1a1a; margin: 0; padding: 24px; }
.wrap { max-width: 760px; margin: 0 auto; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.sub { color: #6b6b6b; font-size: 13.5px; margin-bottom: 14px; }
.banner { display: flex; gap: 16px; align-items: center;
          background: #f7ede3; border: 2px solid #c76d33;
          border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; }
.banner .btitle { font-weight: 700; font-size: 16px;
                  letter-spacing: .05em; color: #a4551f; }
.banner .bsub { font-size: 14.5px; margin-top: 2px; }
.panel { background: #fff; border: 1px solid #ddd9d0; border-radius: 12px;
         padding: 18px 20px; margin-bottom: 16px; }
.label { font-size: 12px; color: #6b6b6b; text-transform: uppercase;
         letter-spacing: .04em; margin-bottom: 8px; }
.row { display: flex; gap: 6px; flex-wrap: wrap; align-items: flex-end; }
.card { width: 46px; height: 64px; border-radius: 6px; background: #fff;
        border: 1px solid #c9c4b8; display: inline-flex; flex-direction:
        column; align-items: center; justify-content: center;
        font-size: 16px; font-weight: 600; box-sizing: border-box; }
.card .s { font-size: 15px; font-weight: 400; }
.card.red { color: #b3372f; }
.card.back { background: #8b95a8; border-color: #7c8598; color: #f0f2f5;
             font-weight: 400; font-size: 13px; }
.card.hot { border: 3px solid #c76d33; }
.card.dim { opacity: .8; }
.card.seen { border: 2px dashed #6a5bd0; }
.card.mine { border: 2px solid #3a7d44; }
.card.theirs { border: 2px solid #b3372f; }
.card.pickedme { outline: 2px dashed #3a7d44; outline-offset: -6px; }
.card.pickedopp { outline: 2px dashed #b3372f; outline-offset: -6px; }
.card.big { width: 56px; height: 78px; font-size: 19px; }
.group { display: inline-flex; gap: 4px; padding: 5px; border-radius: 9px;
         margin-right: 8px; margin-bottom: 4px; }
.group.meld { background: #e8f0e4; border: 1px solid #b9d0ae; }
.group.loose { border: 1px dashed #d4cfc4; }
.gcap { font-size: 11px; color: #8a857a; text-align: center; margin-top: 2px; }
.cols { display: flex; gap: 24px; }
.hist { font-size: 13.5px; line-height: 1.9; color: #3a3a3a; }
.hist b { font-weight: 600; }
.stock { width: 46px; height: 64px; border-radius: 6px; background: #8b95a8;
         border: 1px solid #7c8598; color: #f0f2f5; display: inline-flex;
         align-items: center; justify-content: center; font-size: 14px;
         box-sizing: border-box; }
.stock.hot { border: 3px solid #c76d33; }
.ev { width: 100%; border-collapse: collapse; font-size: 14px; }
.ev td { padding: 5px 8px; vertical-align: middle; }
.bar { height: 13px; border-radius: 4px; background: #d8d3c8; }
.bar.best { background: #c76d33; }
.bar.bad { background: #c8574d; }
.rec { background: #f7ede3; border: 1px solid #e3c9ad; border-radius: 9px;
       padding: 12px 16px; font-size: 14.5px; margin-top: 14px; }
.rec b { color: #a4551f; }
.badge { display: inline-block; background: #ece9f5;
         color: #4a3d7a; font-size: 12px; border-radius: 6px;
         padding: 2px 10px; margin-left: 8px; }
.muted { color: #8a857a; font-size: 12.5px; }
.seennote { color: #6a5bd0; font-size: 12.5px; margin-top: 6px; }
"""


def _card_div(card, cls=""):
    color = "red" if card.suit in RED else ""
    rank = html.escape(card.rank.replace("T", "10"))
    return (f'<div class="card {color} {cls}"><span>{rank}</span>'
            f'<span class="s">{SUIT_SYM[card.suit]}</span></div>')


def _back_div(n=1):
    return '<div class="card back">?</div>' * n


def _cname(card):
    return (html.escape(card.rank.replace("T", "10"))
            + SUIT_SYM[card.suit])


def _best_melds(hand):
    """Best meld cluster for a hand of any size (see play_vs_bot.best_melding)."""
    best, best_count = [], sum(gr_utils.get_deadwood_value(c) for c in hand)
    for cluster in rl_melding.get_meld_clusters(hand=hand):
        meld_cards = [c for m in cluster for c in m]
        count = sum(gr_utils.get_deadwood_value(c)
                    for c in hand if c not in meld_cards)
        if count < best_count:
            best, best_count = cluster, count
    return best, best_count


def _classes(card, hot_ids, seen_ids):
    cls = []
    if cid(card) in hot_ids:
        cls.append("hot")
    if cid(card) in seen_ids:
        cls.append("seen")
    return " ".join(cls)


def _hand_html(hand, hot_ids, seen_ids=()):
    melds, _ = _best_melds(hand)
    meld_cards = [c for m in melds for c in m]
    parts = []
    for m in melds:
        cards = "".join(_card_div(c, _classes(c, hot_ids, seen_ids))
                        for c in m)
        kind = "set" if len({c.rank for c in m}) == 1 else "run"
        parts.append(f'<span><span class="group meld">{cards}</span>'
                     f'<div class="gcap">{kind}</div></span>')
    loose = sorted((c for c in hand if c not in meld_cards),
                   key=lambda c: (c.suit, gr_utils.get_rank_id(c)))
    if loose:
        cards = "".join(_card_div(c, _classes(c, hot_ids, seen_ids))
                        for c in loose)
        parts.append(f'<span><span class="group loose">{cards}</span>'
                     f'<div class="gcap">unmelded</div></span>')
    return ('<div style="display:flex; align-items:flex-start; '
            'flex-wrap:wrap;">' + "".join(parts) + "</div>")


def _history_html(pos):
    lines = []
    has_my_events = False
    for kind, card in pos["events"]:
        c = "" if card is None else _cname(card)
        if kind == "opp-discard":
            lines.append(f"opponent <b>discarded {c}</b>")
        elif kind == "opp-decline":
            lines.append(f"opponent <b>declined {c}</b> and drew from stock")
        elif kind == "opp-pickup":
            lines.append(f"opponent <b>took {c}</b> from the pile")
        elif kind == "opp-draw":
            lines.append("opponent <b>drew from stock</b> (pile empty)")
        elif kind == "opp-pass":
            lines.append("opponent <b>passed on the flip</b>")
        elif kind == "my-pass":
            lines.append("we <b>passed on the flip</b>")
            has_my_events = True
        elif kind == "my-discard":
            lines.append(f"we <b>discarded {c}</b>")
            has_my_events = True
        elif kind == "my-pickup":
            lines.append(f"we <b>took {c}</b> from the pile")
            has_my_events = True
        elif kind == "my-draw":
            lines.append(f"we <b>drew {c} from stock</b>" if c
                         else "we <b>drew from stock</b>")
            has_my_events = True
    # The current (still open) turn. Annotate IN PLACE when the final
    # transcript event already is this turn's draw -- appending another
    # line duplicated it ("17. we drew from stock / 18. we drew from
    # stock (this turn)", fixed 2026-08-30).
    if pos["picked"] is not None:
        lines.append(f"we <b>took {_cname(pos['picked'])}</b> from the "
                     "pile (this turn)")
    elif len(pos["me"]) == 11:
        if pos["events"] and pos["events"][-1][0] == "my-draw":
            lines[-1] += " (this turn)"
        else:
            lines.append("we <b>drew from stock</b> (this turn)")
    if not lines:
        lines.append("(no history given)")
    status = pos.get("history_status")
    if status == "complete":
        note = ('<div class="muted" style="margin-top:6px">&#10003; '
                "complete history</div>")
    elif status == "partial":
        probs = "; ".join(html.escape(p)
                          for p in pos.get("history_problems", []))
        note = ('<div style="margin-top:6px;color:#c0392b;font-size:12px">'
                "&#9888; the recorded history is incomplete or "
                f"inconsistent &mdash; take the verdict with care. {probs}"
                "</div>")
    else:
        note = ("" if has_my_events or not pos["events"] else
                '<div class="muted" style="margin-top:6px">only the '
                "opponent's moves were recorded &mdash; add my-draw / "
                "my-discard events for a full history</div>")
    return ('<div class="hist">' + "<br>".join(
        f"{i + 1}. {s}" for i, s in enumerate(lines)) + "</div>" + note)


def _belief_html(belief, game, top_n=8):
    from evaluate import belief_read_parts
    from sim import DECK
    rows, base = belief_read_parts(belief, game, top_n)
    if base >= 1.0 and not rows:
        return ""
    if not rows:
        return ('<div class="muted">no strong read on their hidden '
                "cards yet</div>")
    sure = [r for r in rows if r[1] >= 0.90]
    likely = [r for r in rows if r[1] < 0.90]
    parts = []
    if sure:
        parts.append("<b>almost certainly holding</b>: " + ", ".join(
            f"{_cname(DECK[i])}&nbsp;({p:.0%})" for i, p in sure))
    if likely:
        parts.append("likely: " + ", ".join(
            f"{_cname(DECK[i])}&nbsp;({p:.0%})" for i, p in likely))
    return ('<div class="muted" style="color:#3a3a3a">reading their '
            "play &mdash; " + "; ".join(parts)
            + f' <span class="muted">(chance level {base:.0%})</span>'
            "</div>")


def _action_text(action, game):
    if isinstance(action, DeclareDeadHandAction):
        return "DECLARE DEAD HAND (0&ndash;0)"
    if isinstance(action, DrawCardAction):
        return "draw from stock"
    if isinstance(action, PickUpDiscardAction):
        top = game.round.dealer.discard_pile[-1]
        return "take " + _cname(top) + " from pile"
    if isinstance(action, (DiscardAction, KnockAction)):
        c = _cname(action.card)
        return (f"KNOCK, discarding {c}" if isinstance(action, KnockAction)
                else f"discard {c}")
    if isinstance(action, GinAction):
        return "GIN"
    return html.escape(str(action))


def _plain_followup(desc):
    """Translate evaluate.py's follow-up annotations to player terms."""
    if desc is None:
        return None
    desc = desc.replace(" (borderline -- close call)", " (close call)")
    if " (statistical tie with " in desc:
        head, alt = desc.split(" (statistical tie with ", 1)
        desc = f"{head} &mdash; or {alt.rstrip(')')}; effectively tied"
    return desc


def _verdict_html(info, game, cfg):
    chosen = info["chosen"]
    if "payoff_means" not in info:
        return (f'<div class="rec"><b>{_action_text(chosen, game)}</b> '
                "&mdash; the only reasonable move here.</div>")
    evs = [m * 100.0 for m in info["payoff_means"]]
    lo, hi = min(evs), max(evs)
    span = max(hi - lo, 1e-9)
    chosen_k = next(k for k in range(len(evs))
                    if info["candidates"][k].action_id == chosen.action_id)
    ev_chosen = evs[chosen_k]
    rows = []
    order = sorted(range(len(evs)), key=lambda k: -evs[k])
    for k in order:
        a = info["candidates"][k]
        is_chosen = k == chosen_k
        width = 25 + 75 * (evs[k] - lo) / span
        if is_chosen:
            cls, note = "best", "recommended"
        elif ev_chosen - evs[k] <= 1.5:
            cls, note = "", "nearly equal"
        elif ev_chosen - evs[k] >= 8.0:
            cls, note = "bad", "costly"
        else:
            cls, note = "", ""
        note_html = f' <span class="badge">{note}</span>' if note else ""
        rows.append(
            f"<tr><td style='width:200px; font-weight:{600 if is_chosen else 400}'>"
            f"{_action_text(a, game)}</td>"
            f"<td><div class='bar {cls}' style='width:{width:.0f}%'></div></td>"
            f"<td style='width:70px; text-align:right'>{evs[k]:+.2f}</td>"
            f"<td style='width:110px'>{note_html}</td></tr>")

    # Plain-language confidence: how far ahead of the next-best line?
    others = [evs[k] for k in range(len(evs)) if k != chosen_k]
    margin = ev_chosen - max(others) if others else 99.0
    runner = next((k for k in order if k != chosen_k), None)
    runner_txt = _action_text(info["candidates"][runner], game) \
        if runner is not None else ""
    if margin >= 4.0:
        why = f"clearly the best line ({margin:.1f} points ahead)"
    elif margin >= 1.5:
        why = f"best by {margin:.1f} points"
    elif margin >= 0.0:
        why = (f"a close call: {runner_txt} is only {margin:.1f} "
               "points behind")
    else:
        why = (f"effectively a tie with {runner_txt}: its average is "
               f"{-margin:.1f} higher, within noise")
    table = ("<table class='ev'><tr class='muted'><td>your options</td>"
             "<td>expected final score (this hand)</td>"
             "<td style='text-align:right'>EV</td><td></td></tr>"
             + "".join(rows) + "</table>")
    fol = _plain_followup(info.get("followup_desc"))
    plan = f" &mdash; then <b>{fol}</b>" if fol else ""
    rec = (f'<div class="rec">Recommended: <b>{_action_text(chosen, game)}'
           f"</b>{plan}<br>{why}. With good play on both sides this hand "
           f"is worth about <b>{ev_chosen:+.1f}</b> points to us.</div>")
    return table + rec


def _explain_html(info, game):
    """The optional 'why' panel: outcome distributions + key draws."""
    ex = info.get("explain")
    if not ex:
        return ""
    from explain import card_rows, opp_card_rows, outcome_rows
    from sim import DECK
    chosen_id = info["chosen"].action_id
    blocks = []
    for k in ex["picks"]:
        a = info["candidates"][k]
        s = ex["stats"][k]
        avg = s["sum"] / max(s["n"], 1)
        is_rec = a.action_id == chosen_id
        badge = ' <span class="badge">recommended</span>' if is_rec else ""
        rows = []
        for label, pct, apts, cnt in outcome_rows(s):
            good = apts > 0
            rows.append(
                f"<tr><td style='width:230px'>{label}</td>"
                f"<td><div class='bar {'best' if good else 'bad'}' "
                f"style='width:{max(pct, 2):.0f}%'></div></td>"
                f"<td style='width:60px;text-align:right'>{pct:.0f}%</td>"
                f"<td style='width:150px;text-align:right'>"
                f"conditional EV <b>{apts:+.1f}</b></td></tr>")
        cards_html = ""
        good, bad = card_rows(s)
        hold = ex.get("opp_hold", {})
        if good or bad:
            def chip(c, note, apts, cnt, fed, hurt):
                extra = note or ("no help" if hurt else "")
                if c in hold:
                    extra = (extra + "; " if extra else "") + \
                        f"likely in their hand ({hold[c]:.0%})"
                if hurt and fed >= 25:
                    extra += f"; they took it {fed:.0f}% of the time"
                cap = f"{extra + ': ' if extra else ''}EV {apts:+.1f}"
                return ('<div style="display:inline-flex;flex-direction:'
                        'column;align-items:center;margin:0 8px 6px 0;'
                        'max-width:110px;text-align:center">'
                        + _card_div(DECK[c])
                        + f'<span class="muted" style="margin-top:2px">'
                        f"{cap}</span></div>")
            gh = "".join(chip(*r, hurt=False) for r in good)
            bh = "".join(chip(*r, hurt=True) for r in bad)
            cards_html = '<div style="margin-top:8px">'
            if gh:
                cards_html += ('<div class="muted">draws that help us '
                               'most from here:</div><div class="row">'
                               + gh + "</div>")
            if bh:
                cards_html += ('<div class="muted" style="margin-top:6px">'
                               'draws that hurt us most:</div>'
                               '<div class="row">' + bh + "</div>")
            cards_html += "</div>"
        root_c = cid(a.card) if isinstance(a, DiscardAction) \
            and not isinstance(a, KnockAction) else None
        danger, took = opp_card_rows(s, root_c)
        if took is not None:
            pct = 100.0 * took[2] / max(s["n"], 1)
            cards_html += (f'<div style="margin-top:6px">they take our '
                           f"<b>{_cname(DECK[took[0]])}</b> {pct:.0f}% of "
                           f"the time; conditional EV "
                           f"<b>{took[1]:+.1f}</b></div>")
        if danger:
            dh = "".join(
                '<div style="display:inline-flex;flex-direction:column;'
                'align-items:center;margin:0 8px 6px 0;max-width:110px;'
                'text-align:center">' + _card_div(DECK[c])
                + f'<span class="muted" style="margin-top:2px">EV '
                f"{apts:+.1f}</span></div>"
                for c, apts, cnt in danger)
            cards_html += ('<div style="margin-top:8px">'
                           '<div class="muted">cards that hurt us if the '
                           "opponent gets them first:</div>"
                           '<div class="row">' + dh + "</div></div>")
        blocks.append(
            f'<div style="margin-bottom:16px"><div style="font-weight:600;'
            f'margin-bottom:6px">{_action_text(a, game)}{badge} '
            f'<span style="font-weight:400">(EV {avg:+.1f})</span></div>'
            f"<table class='ev'>{''.join(rows)}</table>{cards_html}</div>")
    note = (f"each line was played to the end of the hand "
            f"{ex['worlds']} times, both sides simulated")
    return ('<div class="panel"><div class="label">How each '
            'line plays out</div>' + "".join(blocks)
            + f'<div class="muted">{note}</div></div>')


def render_html(pos, game, belief, info, cfg):
    rnd = game.round
    hero = rnd.players[0]
    hero_hand = sorted(hero.hand,
                       key=lambda c: (c.suit, gr_utils.get_rank_id(c)))
    opp = rnd.players[1]
    pile = rnd.dealer.discard_pile
    chosen = info["chosen"]

    hot_ids = set()
    hot_pile = hot_stock = False
    if isinstance(chosen, (DiscardAction, KnockAction, GinAction)):
        if hasattr(chosen, "card") and chosen.card is not None:
            hot_ids = {cid(chosen.card)}
    elif isinstance(chosen, PickUpDiscardAction):
        hot_pile = True
    elif isinstance(chosen, DrawCardAction):
        hot_stock = True

    phase = ("to move: take the upcard, or draw from stock?"
             if len(hero_hand) == 10 else
             "to move: choose a discard")
    if pos["opp_model"]:
        t, p = pos["opp_model"]
        opp_line = (f"assumes the opponent knocks whenever their deadwood "
                    f"is {t} or less" if p >= 1.0 else
                    f"assumes the opponent knocks at deadwood &le; {t} "
                    f"about {p:.0%} of the time")
    else:
        opp_line = "assumes the opponent never knocks"

    # --- prominent game-variant banner (user directive 2026-08-30:
    # the flip and the knock threshold are critically relevant, and
    # were previously hidden in small font). ---
    home = pos.get("home")
    flip = pos.get("flip") or (pile[0] if pile else None)
    flip_in_pile = (home and pile and flip is not None
                    and cid(pile[0]) == cid(flip))
    if home and flip is not None:
        where = ("bottom of the pile" if flip_in_pile
                 else "since picked up")
        banner = (f'<div class="banner">{_card_div(flip, "big")}'
                  f'<div><div class="btitle">HOME VARIANT</div>'
                  f'<div class="bsub">first card flipped: '
                  f"<b>{_cname(flip)}</b> ({where}) &nbsp;&rarr;&nbsp; "
                  f"knocking allowed at deadwood &le; "
                  f"<b>{pos['knock_max']}</b></div></div></div>")
    else:
        banner = ('<div class="banner"><div>'
                  '<div class="btitle">STANDARD GAME</div>'
                  '<div class="bsub">knocking allowed at deadwood &le; '
                  f"<b>{pos['knock_max']}</b></div></div></div>")

    ids = ids_of(hero_hand)
    dw = (min(min_deadwood_ids(ids[:k] + ids[k + 1:]) for k in range(11))
          if len(ids) == 11 else min_deadwood_ids(ids))

    opp_cards = "".join(_card_div(c) for c in opp.known_cards)
    opp_cards += _back_div(len(opp.hand) - len(opp.known_cards))
    opp_note = ""
    if opp.known_cards:
        opp_note = ('<div class="muted" style="margin-top:6px">face-up '
                    "cards = pickups we have seen; the rest is hidden"
                    "</div>")

    # our cards the OPPONENT has seen us take (they will not feed
    # these melds -- strategically different from our hidden cards).
    seen_ids = set(ids_of(hero.known_cards))
    seen_note = ""
    if seen_ids:
        seen_list = ", ".join(
            _cname(c) for c in sorted(hero.known_cards, key=cid))
        seen_note = (f'<div class="seennote">dashed cards ({seen_list}): '
                     "picked up in view of the opponent, who will avoid "
                     "discarding into those melds</div>")

    # Discard-pile provenance (user directive 2026-08-30): solid green
    # border = our discard, solid red = theirs; inner dashed ring = the
    # card was picked up at some point and re-discarded, colored by who
    # picked it up last. Read from the transcript; cards without
    # recorded history stay neutral.
    disc_by, picked_by = {}, {}
    for kind, card in pos["events"]:
        if card is None:
            continue
        i = cid(card)
        if kind == "my-discard":
            disc_by[i] = "mine"
        elif kind == "opp-discard":
            disc_by[i] = "theirs"
        elif kind == "my-pickup":
            picked_by[i] = "pickedme"
        elif kind == "opp-pickup":
            picked_by[i] = "pickedopp"

    def _pile_cls(c, base=""):
        i = cid(c)
        cls = [base] if base else []
        if "hot" not in base:
            cls.append(disc_by.get(i, ""))
        cls.append(picked_by.get(i, ""))
        return " ".join(x for x in cls if x)

    if pile:
        divs = [_card_div(c, _pile_cls(c, "dim")) for c in pile[:-1]]
        divs.append(_card_div(pile[-1],
                              _pile_cls(pile[-1], "hot" if hot_pile
                                        else "")))
        if flip_in_pile:
            # badge the founding flip (pile bottom): it is nobody's
            # discard and it set this hand's knock threshold
            divs[0] = ('<div style="display:inline-flex;flex-direction:'
                       'column;align-items:center;vertical-align:top">'
                       + divs[0] +
                       '<span style="font-size:9px;color:#c99a2c;'
                       'letter-spacing:1px;margin-top:2px">FLIP &le; '
                       f'{pos["knock_max"]}</span></div>')
        pile_html = "".join(divs)
    else:
        pile_html = '<span class="muted">(empty)</span>'
    pile_ids = {cid(c) for c in pile}
    pile_legend = ""
    if pile_ids & (set(disc_by) | set(picked_by)):
        pile_legend = ('<div class="muted" style="margin-top:6px">'
                       '<span style="color:#3a7d44">green border</span> = '
                       'our discard, <span style="color:#b3372f">red'
                       '</span> = theirs; a dashed inner ring means the '
                       'card was picked up and later thrown back (ring '
                       'color = who took it)</div>')
    stock_html = (f'<div class="stock{" hot" if hot_stock else ""}">'
                  f"{len(rnd.dealer.stock_pile)}</div>")

    ver = cfg.get("engine_version")
    stamp = (f'<div class="sub">analyzed by <b>{ver}</b></div>'
             if ver else "")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Gin rummy evaluator</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Gin rummy evaluator</h1>
<div class="sub">{phase} &nbsp;&middot;&nbsp; {opp_line}</div>
{banner}
{stamp}

<div class="panel">
<div class="label">Opponent &mdash; {len(opp.known_cards)} known,
{len(opp.hand) - len(opp.known_cards)} hidden</div>
<div class="row">{opp_cards}</div>
{opp_note}
{_belief_html(belief, game)}
</div>

<div class="panel">
<div class="label">Discard pile (bottom &rarr; top) &nbsp;/&nbsp; stock</div>
<div class="row">{pile_html}
<span style="width:14px"></span>{stock_html}</div>
{pile_legend}
</div>

<div class="panel">
<div class="label">Our hand &mdash; {len(hero_hand)} cards,
deadwood {dw}</div>
{_hand_html(hero_hand, hot_ids, seen_ids)}
{seen_note}
</div>

<div class="panel">
<div class="label">Turn history</div>
{_history_html(pos)}
</div>

<div class="panel">
<div class="label">Verdict</div>
{_verdict_html(info, game, cfg)}
</div>
{_explain_html(info, game)}
<div class="muted">Highlighted card / pile / stock = the recommended
move. EV = the expected final score of this hand for us, in
points.</div>
</div></body></html>"""
