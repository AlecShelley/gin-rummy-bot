"""
Optional explanation pipeline for the evaluator (evaluate.py --explain).

Answers "WHY this move?" with evidence a player can use (user directive
2026-08-30), at extra runtime:

  1. OUTCOME DISTRIBUTIONS -- for the recommended move and its main
     rivals (always including the best knock when one exists), the
     remaining hand is played out to COMPLETION many times over fresh
     belief-sampled worlds, both sides played by the engine's own
     continuation model (hero: gin-seeking teacher with the deep
     first-decision branch, exactly as the search evaluates lines;
     opponent: knocks per the opponent model).  Each terminal is
     classified -- we gin / we knock clean / we get undercut / they
     gin / they knock / we undercut their knock / hand dies 0-0 --
     with frequencies and average points.

  2. CRITICAL CARDS -- the same playout batch, grouped by OUR first
     stock draw, shows which draws swing the line's EV, annotated
     structurally (GIN out, enables a knock, improves deadwood by k,
     no help) and with how often a useless draw ended up feeding the
     opponent (we discarded it, they took it).

Worlds are sampled from the same belief (and pass the same
opponent-knock filter) the verdict itself used, so the statistics
decompose the EVs in the main table; full playouts replace the
search's truncated ones, so small differences from the EV column are
expected.
"""

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DeclareDeadHandAction, DiscardAction, DrawCardAction, GinAction,
    KnockAction, PickUpDiscardAction)

import sim
from sim import (DECK, apply_world, cid, fast_legal_actions, ids_of,
                 make_scratch, min_deadwood_ids, restore, snapshot,
                 step_fast)

# outcome keys -> player-facing labels (hero = player 0 throughout)
LABELS = {
    "we_gin": "we GIN",
    "we_knock_win": "we knock cleanly",
    "we_knock_uc": "we knock, get UNDERCUT",
    "they_gin": "they gin",
    "they_knock_win": "they knock cleanly",
    "they_knock_uc": "they knock, we undercut them",
    "dead": "hand dies (0–0)",
}
ORDER = ["we_gin", "we_knock_win", "they_knock_uc", "dead",
         "they_knock_win", "we_knock_uc", "they_gin"]


def _classify(game, payoff0):
    rnd = game.round
    act = rnd.going_out_action
    goer = rnd.going_out_player_id
    if act is None or isinstance(act, DeclareDeadHandAction) \
            or goer is None:
        return "dead"
    if isinstance(act, GinAction):
        return "we_gin" if goer == 0 else "they_gin"
    if goer == 0:
        return "we_knock_win" if payoff0 > 0 else "we_knock_uc"
    return "they_knock_uc" if payoff0 > 0 else "they_knock_win"


class _Track:
    """Per-playout bookkeeping: hero's first stock draw, whether the
    opponent later picked that same card up off the pile, and the
    first card the OPPONENT acquired (stock draw or pickup) -- the
    danger-card statistic for lines where we never draw again."""

    __slots__ = ("first_draw", "fed", "opp_first")

    def __init__(self):
        self.first_draw = None
        self.fed = False
        self.opp_first = None


def _play_to_end(game, policies, track, max_steps=250):
    """sim.playout semantics (untruncated) + tracking; returns payoff0."""
    rnd = game.round
    for _ in range(max_steps):
        if rnd.going_out_action is not None or rnd.is_over:
            return sim._payoff(game)
        legal = fast_legal_actions(game)
        action = legal[0] if len(legal) == 1 \
            else policies[rnd.current_player_id].act(game, legal)
        _note(game, action, track)
        step_fast(game, action)
    return sim._payoff(game)


def _note(game, action, track):
    rnd = game.round
    if isinstance(action, DrawCardAction):
        if rnd.current_player_id == 0 and track.first_draw is None \
                and rnd.dealer.stock_pile:
            track.first_draw = cid(rnd.dealer.stock_pile[-1])
        elif rnd.current_player_id == 1 and track.opp_first is None \
                and rnd.dealer.stock_pile:
            track.opp_first = cid(rnd.dealer.stock_pile[-1])
    elif isinstance(action, PickUpDiscardAction) \
            and rnd.current_player_id == 1 and rnd.dealer.discard_pile:
        top = cid(rnd.dealer.discard_pile[-1])
        if track.opp_first is None:
            track.opp_first = top
        if track.first_draw is not None and top == track.first_draw:
            track.fed = True


def _deep_play(game, policies, pruner, kbranch, max_steps=250):
    """adaptive.playout_deep (untruncated), returning the BEST branch's
    (payoff0, outcome, track) instead of just its payoff.  Snapshot /
    restore across sibling branches only (the parity-proven v5
    pattern)."""
    rnd = game.round
    track = _Track()
    for _ in range(max_steps):
        if rnd.going_out_action is not None or rnd.is_over:
            p = sim._payoff(game)
            return p, _classify(game, p), track
        legal = fast_legal_actions(game)
        if len(legal) == 1:
            _note(game, legal[0], track)
            step_fast(game, legal[0])
            continue
        if rnd.current_player_id != 0:
            a = policies[rnd.current_player_id].act(game, legal)
            _note(game, a, track)
            step_fast(game, a)
            continue
        cands = pruner._candidates(game, legal)[:kbranch]
        if len(cands) == 1:
            _note(game, cands[0], track)
            step_fast(game, cands[0])
            continue
        snap = snapshot(game)
        best = None
        for a in cands:
            restore(game, snap)
            btrack = _Track()
            btrack.first_draw, btrack.fed = track.first_draw, track.fed
            btrack.opp_first = track.opp_first
            _note(game, a, btrack)
            step_fast(game, a)
            p = _play_to_end(game, policies, btrack)
            if best is None or p > best[0]:
                best = (p, _classify(game, p), btrack)
        return best
    p = sim._payoff(game)
    return p, _classify(game, p), track


def _pick_candidates(info):
    """chosen + best knock (if distinct) + best remaining, max 3."""
    evs = info["payoff_means"]
    chosen_k = next(k for k, a in enumerate(info["candidates"])
                    if a.action_id == info["chosen"].action_id)
    picks = [chosen_k]
    knocks = [k for k, a in enumerate(info["candidates"])
              if isinstance(a, KnockAction) and k != chosen_k]
    if knocks:
        picks.append(max(knocks, key=lambda k: evs[k]))
    rest = [k for k in range(len(evs)) if k not in picks]
    if rest and len(picks) < 3:
        picks.append(max(rest, key=lambda k: evs[k]))
    return picks


def run_explain(game, belief, info, expert, pos, n_worlds=600, seed=0,
                boost=0.0):
    """Returns the explanation dict (also stored as info['explain'])."""
    if "payoff_means" not in info:
        return None
    hero_id = game.round.current_player_id
    opp_id = 1 - hero_id
    rollout = expert.rollout if expert._opp_rollout is None else \
        {hero_id: expert.rollout[hero_id], opp_id: expert._opp_rollout}
    pruner = getattr(expert, "_branch_pruner", None)
    kbranch = getattr(expert, "kbranch", 3)

    scratch = make_scratch(game)
    base = snapshot(scratch)
    opp = scratch.round.players[opp_id]
    unseen = ids_of(scratch.round.dealer.stock_pile) + \
        [cid(c) for c in opp.hand if c not in opp.known_cards]
    num_hidden = len(opp.hand) - len(opp.known_cards)
    known_ids = ids_of(opp.known_cards)
    opp_discard_count = int(belief.opp_discarded.sum())
    rng = np.random.default_rng(seed + 977)

    picks = _pick_candidates(info)
    stats = {k: {"n": 0, "sum": 0.0,
                 "outcomes": {o: [0, 0.0] for o in LABELS},
                 "draw": {}, "fed": {}, "odraw": {}} for k in picks}
    done, attempts = 0, 0
    while done < n_worlds and attempts < 5 * n_worlds:
        attempts += 1
        free = belief.sample_free_cards(unseen, num_hidden, rng, boost) \
            if num_hidden > 0 else []
        if expert._world_weight(known_ids + list(free),
                                opp_discard_count) <= 0.0:
            continue
        restore(scratch, base)
        apply_world(scratch, opp_id, [DECK[i] for i in free], rng)
        world = snapshot(scratch)
        for k in picks:
            restore(scratch, world)
            step_fast(scratch, info["candidates"][k])
            if pruner is not None:
                p, out, tr = _deep_play(scratch, rollout, pruner, kbranch)
            else:
                tr = _Track()
                p = _play_to_end(scratch, rollout, tr)
                out = _classify(scratch, p)
            s = stats[k]
            pts = (p if hero_id == 0 else -p) * 100.0
            s["n"] += 1
            s["sum"] += pts
            s["outcomes"][out][0] += 1
            s["outcomes"][out][1] += pts
            if tr.first_draw is not None:
                d = s["draw"].setdefault(tr.first_draw, [0, 0.0])
                d[0] += 1
                d[1] += pts
                if tr.fed:
                    f = s["fed"].setdefault(tr.first_draw, 0)
                    s["fed"][tr.first_draw] = f + 1
            if tr.opp_first is not None:
                d = s["odraw"].setdefault(tr.opp_first, [0, 0.0])
                d[0] += 1
                d[1] += pts
        done += 1

    # structural annotation of possible next draws, per candidate
    hand_ids = ids_of(game.round.players[hero_id].hand)
    thr = pos.get("knock_max", 10)
    for k in picks:
        a = info["candidates"][k]
        if isinstance(a, (KnockAction, GinAction, DeclareDeadHandAction)):
            continue
        if isinstance(a, DiscardAction):
            h10 = [i for i in hand_ids if i != cid(a.card)]
        elif isinstance(a, PickUpDiscardAction):
            top = cid(game.round.dealer.discard_pile[-1])
            h11 = hand_ids + [top]
            h10 = None                    # 11 cards: next draw is theirs
        else:                             # draw from stock
            h10 = list(hand_ids)
        if h10 is None or len(h10) != 10:
            continue
        dw_now = min_deadwood_ids(h10)
        notes = {}
        for c in unseen:
            h11 = h10 + [c]
            dwc = min(min_deadwood_ids([x for x in h11 if x != d])
                      for d in h11)
            if dwc == 0:
                notes[c] = "GIN out"
            elif dwc <= thr < dw_now:
                notes[c] = f"lets us knock (deadwood {dwc})"
            elif dw_now - dwc >= 4:
                notes[c] = f"improves us by {dw_now - dwc}"
            else:
                notes[c] = ""
        stats[k]["notes"] = notes
    if hasattr(belief, "raw_marginals"):
        marg = belief.raw_marginals(unseen, num_hidden)
    else:
        marg = belief.marginals(unseen, num_hidden)
    opp_hold = {c: float(marg[c]) for c in unseen if marg[c] >= 0.6}
    return {"picks": picks, "stats": stats, "worlds": done,
            "vetoed": attempts - done, "opp_hold": opp_hold}


# ---------------------------------------------------------------------------
# Rendering helpers (shared by text and HTML).
# ---------------------------------------------------------------------------

def outcome_rows(stat):
    """[(label, pct, avg_pts, count)] for outcomes that occurred."""
    n = max(stat["n"], 1)
    rows = []
    for o in ORDER:
        cnt, tot = stat["outcomes"][o]
        if cnt:
            rows.append((LABELS[o], 100.0 * cnt / n, tot / cnt, cnt))
    return rows


def card_rows(stat, min_n=6, top=5, bottom=4):
    """Best / worst next draws: [(card_id, note, avg, count, fed_pct)]."""
    if not stat.get("draw"):
        return [], []
    notes = stat.get("notes", {})
    rows = []
    for c, (cnt, tot) in stat["draw"].items():
        if cnt >= min_n:
            fed = 100.0 * stat["fed"].get(c, 0) / cnt
            rows.append((c, notes.get(c, ""), tot / cnt, cnt, fed))
    rows.sort(key=lambda r: -r[2])
    good = [r for r in rows[:top] if r[2] > 0 or r[1]]
    bad = [r for r in rows[-bottom:] if r not in good]
    bad.sort(key=lambda r: r[2])
    return good, bad


def opp_card_rows(stat, root_discard=None, min_n=6, bottom=4,
                  spread_min=3.0):
    """(danger, took): `danger` = cards that hurt us most if the
    OPPONENT gets them first, [(card_id, avg, count)] worst first,
    shown only when which card reaches them actually swings the line
    (spread >= 3 pts); `took` = the stats for them taking OUR root
    discard, (card_id, avg, count) or None -- always worth showing
    for a discard decision."""
    if not stat.get("odraw"):
        return [], None
    rows = [(c, tot / cnt, cnt)
            for c, (cnt, tot) in stat["odraw"].items() if cnt >= min_n]
    took = next(((c, a, n) for c, a, n in rows if c == root_discard),
                None)
    danger = []
    if len(rows) >= 2:
        avgs = [r[1] for r in rows]
        if max(avgs) - min(avgs) >= spread_min:
            line_avg = stat["sum"] / max(stat["n"], 1)
            rows.sort(key=lambda r: r[1])
            danger = [r for r in rows[:bottom]
                      if r[1] < line_avg - 1.0 and r[0] != root_discard]
    return danger, took
