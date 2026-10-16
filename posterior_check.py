"""
POSTERIOR GROUND-TRUTH CHECK (registered results/v9_gates.log,
2026-09-05): the exact Bayesian-consistency test of the champion
particle-filter belief. For a FROZEN hero-observable info set
(hero hand + flip + event history), estimate the TRUE conditional
distribution of the opponent's hidden hand by rejection sampling —
hero side fixed by construction, remaining 41 cards permuted
uniformly, accept iff the v8home deployment line reproduces the
target history exactly (implication-only screens; hero policy seed
fixed; opp seed fresh per sample) — and compare the PF ensemble's
predicted per-card marginals against the empirical frequencies.

    .venv/Scripts/python.exe posterior_check.py --scenario s1
    .venv/Scripts/python.exe posterior_check.py --scenario s2 --workers 12
    .venv/Scripts/python.exe posterior_check.py --scenario s3 --workers 12
    .venv/Scripts/python.exe posterior_check.py --scenario s2 --report
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, DrawCardAction, PickUpDiscardAction)

import sim
from sim import (DEADWOOD_VALUE, cid, fast_legal_actions, flip_book_take,
                 ids_of, min_deadwood_ids, step_fast)

HERO_PS = 900001                         # frozen hero policy seed

SCEN = {
    "s1": dict(kind="decline",
               hand=[0, 8, 21, 25, 29, 31, 34, 38, 46, 48], flip=15,
               seed=108000, out="results/post_s1_rows.jsonl"),
    "s2": dict(kind="take",
               hand=[2, 5, 7, 29, 36, 37, 38, 40, 41, 43], flip=30,
               X=22, seed=110000, out="results/post_s2_rows.jsonl"),
    "s3": dict(kind="draw",
               hand=[0, 8, 21, 25, 29, 31, 34, 38, 46, 48], flip=15,
               D=26, C=25, Y=6, seed=112000,
               out="results/post_s3_rows.jsonl"),
    # Phase-B held-out scenarios (picker seed 114000, 2026-09-06):
    "s4": dict(kind="decline",
               hand=[3, 6, 18, 20, 22, 24, 25, 28, 31, 48], flip=10,
               seed=114500, out="results/post_s4_rows.jsonl"),
    "s5": dict(kind="take",
               hand=[0, 1, 4, 8, 11, 16, 31, 33, 44, 47], flip=13,
               X=12, seed=116000, out="results/post_s5_rows.jsonl"),
    "s6": dict(kind="draw",
               hand=[3, 6, 18, 20, 22, 24, 25, 28, 31, 48], flip=10,
               D=2, C=25, Y=51, seed=118000,
               out="results/post_s6_rows.jsonl"),
    # V11 G3 held-out scenarios (picker seed 130000, 2026-09-08):
    "s7": dict(kind="draw",
               hand=[3, 13, 14, 16, 18, 20, 27, 34, 38, 44], flip=49,
               D=41, C=34, Y=9, seed=132000,
               out="results/post_s7_rows.jsonl"),
    "s8": dict(kind="take",
               hand=[5, 10, 20, 21, 26, 29, 30, 34, 48, 51], flip=44,
               X=31, seed=134000, out="results/post_s8_rows.jsonl"),
}


def _base_game(hand, flip):
    """Fixed hero side, opening alive (home_demo pattern). Returns
    (game, hero_id, DECK)."""
    game = sim.new_game_home(seed=1, flip_index=0)
    rnd = game.round
    hero = int(rnd.current_player_id)
    deck = {}
    for c in rnd.dealer.shuffled_deck:
        deck[cid(c)] = c
    rnd.players[hero].hand[:] = [deck[i] for i in hand]
    rnd.players[hero].known_cards[:] = []
    rnd.players[1 - hero].known_cards[:] = []
    rnd.dealer.discard_pile[:] = [deck[flip]]
    game.settings.going_out_deadwood_count = DEADWOOD_VALUE[flip]
    rnd._home_flip_id = flip
    rnd._home_opening = 1
    game.actions[-1] = DiscardAction(card=deck[flip])
    return game, hero, deck


def _fill(game, hero, deck, rest):
    """Install one permutation: opp = rest[:10], stock = rest[10:]
    (stock pops from the END, so rest[-1] is the first draw)."""
    rnd = game.round
    rnd.players[1 - hero].hand[:] = [deck[i] for i in rest[:10]]
    rnd.dealer.stock_pile[:] = [deck[i] for i in rest[10:]]


def _act(pols, game, legal):
    rnd = game.round
    return legal[0] if len(legal) == 1 else \
        pols[int(rnd.current_player_id)].act(game, legal)


# ---------------------------------------------------------------- samplers

def _sample_s1(sc, rng, n):
    """Pure-function sampling: accept iff opp's book declines the
    flip as P2 (hero's P1 pass is book-automatic for this hand)."""
    hand, flip = sc["hand"], sc["flip"]
    assert not flip_book_take(hand, flip, 1)
    rest0 = [c for c in range(52) if c not in set(hand) and c != flip]
    rows = []
    tried = 0
    while len(rows) < n:
        tried += 1
        rest = list(rest0)
        rng.shuffle(rest)
        opp = rest[:10]
        if flip_book_take(opp, flip, 2):
            continue                     # opp would take: not our line
        rows.append(dict(opp=sorted(opp), tried=tried))
    return rows


def _sample_s2(job):
    sc, n, seed = job
    sim.set_rules("full")
    from home_variant import HomeV8Policy
    hand, flip, X = sc["hand"], sc["flip"], sc["X"]
    rng = random.Random(seed)
    rest0 = [c for c in range(52) if c not in set(hand) and c != flip]
    rows = []
    tried = searched = 0
    while len(rows) < n and tried < n * 30000:
        tried += 1
        rest = list(rest0)
        rng.shuffle(rest)
        opp = rest[:10]
        if X not in opp:
            continue                     # implied: opp discards X from 11
        if not flip_book_take(opp, flip, 2):
            continue                     # implied: opp takes the flip
        searched += 1
        game, hero, deck = _base_game(hand, flip)
        _fill(game, hero, deck, rest)
        pols = {hero: HomeV8Policy(seed=HERO_PS),
                1 - hero: HomeV8Policy(seed=seed * 7 + tried)}
        step_fast(game, _act(pols, game, fast_legal_actions(game)))  # H pass
        rnd = game.round
        a = _act(pols, game, fast_legal_actions(game))               # O option
        if not isinstance(a, PickUpDiscardAction):
            continue                     # book said take; belt+braces
        step_fast(game, a)
        a = _act(pols, game, fast_legal_actions(game))               # O discard
        if not (isinstance(a, DiscardAction) and cid(a.card) == X):
            continue
        rows.append(dict(opp=sorted(opp), tried=tried, searched=searched))
    return rows


def _sample_s3(job):
    sc, n, seed = job
    sim.set_rules("full")
    from home_variant import HomeV8Policy
    hand, flip = sc["hand"], sc["flip"]
    D, C, Y = sc["D"], sc["C"], sc["Y"]
    rng = random.Random(seed)
    rest0 = [c for c in range(52) if c not in set(hand) and c != flip]
    rows = []
    tried = searched = herr = 0
    while len(rows) < n and tried < n * 300000:
        tried += 1
        rest = list(rest0)
        rng.shuffle(rest)
        opp = rest[:10]
        if rest[-1] != D:
            continue                     # implied: hero's forced draw is D
        if flip_book_take(opp, flip, 2):
            continue                     # implied: opp passes the flip
        if Y not in opp and Y != rest[-2]:
            continue                     # implied: opp discards Y from 11
        searched += 1
        game, hero, deck = _base_game(hand, flip)
        _fill(game, hero, deck, rest)
        pols = {hero: HomeV8Policy(seed=HERO_PS),
                1 - hero: HomeV8Policy(seed=seed * 7 + tried)}
        step_fast(game, _act(pols, game, fast_legal_actions(game)))  # H pass
        step_fast(game, _act(pols, game, fast_legal_actions(game)))  # O pass
        step_fast(game, _act(pols, game, fast_legal_actions(game)))  # H draw D
        a = _act(pols, game, fast_legal_actions(game))               # H discard
        if not (isinstance(a, DiscardAction) and cid(a.card) == C):
            herr += 1                    # hero determinism violation
            continue
        step_fast(game, a)
        a = _act(pols, game, fast_legal_actions(game))               # O draw?
        if not isinstance(a, DrawCardAction):
            continue                     # opp took C: not our line
        step_fast(game, a)
        a = _act(pols, game, fast_legal_actions(game))               # O discard
        if not (isinstance(a, DiscardAction) and cid(a.card) == Y):
            continue
        rows.append(dict(opp=sorted(opp), tried=tried,
                         searched=searched, herr=herr))
    return rows


# ---------------------------------------------------------------- PF side

def _snap(name, sc):
    """The measurement info set as a play_and_record-style snapshot."""
    hand, flip = sc["hand"], sc["flip"]
    kind = sc.get("kind", name)
    if kind in ("s1", "decline"):
        return dict(threshold=DEADWOOD_VALUE[flip], flip=flip,
                    events=[("my-pass", None), ("opp-pass", None)],
                    hand=list(hand), opp_known=[], my_known=[],
                    pile=[flip], phase="draw")
    if kind in ("s2", "take"):
        X = sc["X"]
        return dict(threshold=DEADWOOD_VALUE[flip], flip=flip,
                    events=[("my-pass", None), ("opp-pickup", flip),
                            ("opp-discard", X)],
                    hand=list(hand), opp_known=[flip], my_known=[],
                    pile=[X], phase="draw")
    D, C, Y = sc["D"], sc["C"], sc["Y"]
    hand3 = sorted(set(hand) - {C} | {D})
    return dict(threshold=DEADWOOD_VALUE[flip], flip=flip,
                events=[("my-pass", None), ("opp-pass", None),
                        ("my-draw", D), ("my-discard", C),
                        ("opp-decline", C), ("opp-discard", Y)],
                hand=hand3, opp_known=[], my_known=[],
                pile=[flip, C, Y], phase="draw")


def _pf_marginals(name, sc, filters=4, seed=0, pf_cfg=None):
    """(raw, shrunk, unseen, core, ensemble) for the scenario's info
    set, under pf_cfg (default: the champion DEFAULT_PF)."""
    from evaluate import (FilterEnsemble, build_particle_belief,
                          build_state, history_check, parse_position)
    from home_examples import pos_text
    from read_regress import ensemble_marginal
    from particle import DEFAULT_PF
    cfg = dict(DEFAULT_PF, **(pf_cfg or {}))
    pos = parse_position(pos_text(_snap(name, sc), "# posterior check"))
    pos["history_status"], pos["history_problems"] = history_check(pos)
    assert pos["history_status"] == "complete", pos["history_problems"]
    pos["opp_model"] = (pos["knock_max"], 1.0)
    game, _ = build_state(pos)
    raw, core = ensemble_marginal(pos, game, dict(cfg),
                                  filters=filters, seed=seed)
    seen = set(core.hero_hand) | set(core.pile) | set(core.opp_known)
    unseen = [c for c in range(52) if c not in seen]
    reps = [build_particle_belief(pos, game, seed + 7919 * r,
                                  pf_cfg=dict(cfg))[0]
            for r in range(filters)]
    ens = FilterEnsemble([b for b in reps if b is not None])
    shrunk = ens.marginals(unseen, core.h, known_ids=core.opp_known)
    return raw, shrunk, unseen, core, ens


# ---------------------------------------------------------------- report

def _tok(i):
    from home_examples import card_tok
    return card_tok(i)


def report(name, sc, rows, filters=4):
    from particle import _brier_probe
    n = len(rows)
    hands = [r["opp"] for r in rows]
    flip = sc["flip"]
    thresh = DEADWOOD_VALUE[flip]
    raw, shrunk, unseen, core, ens = _pf_marginals(name, sc,
                                                   filters=filters)
    raw8, _, _, _, _ = _pf_marginals(name, sc, filters=8, seed=31337)
    hidden_n = core.h
    freq = np.zeros(52)
    for h in hands:
        for c in h:
            freq[c] += 1
    freq /= n
    print(f"=== posterior check {name.upper()}, {n} accepted samples, "
          f"{filters}-filter ensemble ===")
    base = hidden_n / len(unseen)
    ens_noise = float(np.mean(np.abs(raw8[unseen] - raw[unseen])))
    print(f"unseen {len(unseen)} cards, hidden {hidden_n}, "
          f"chance {base:.3f}; ensemble noise (4f vs 8f) "
          f"mean|d|={ens_noise:.4f}")
    # M2 Brier
    hidden_sets = [set(h) - set(core.opp_known) for h in hands]
    b_raw = b_shr = b_uni = 0.0
    for hs in hidden_sets:
        br, bu = _brier_probe(raw, unseen, list(hs), hidden_n)
        bs, _ = _brier_probe(shrunk, unseen, list(hs), hidden_n)
        b_raw += br
        b_shr += bs
        b_uni += bu
    print(f"M2 Brier: raw {b_raw / n:.4f}  shrunk {b_shr / n:.4f}  "
          f"uniform {b_uni / n:.4f}  (raw vs uniform "
          f"{100 * (1 - b_raw / b_uni):+.1f}%)")
    # M1 per-card reliability
    print("M1 per-card (empirical vs raw PF; |z|>3 flag, |z|>2 watch):")
    flags = []
    for c in sorted(unseen, key=lambda c: -abs(freq[c] - raw[c])):
        p_hat, p = freq[c], float(raw[c])
        se = math.sqrt(max(p_hat * (1 - p_hat), 1e-9) / n)
        z = (p_hat - p) / se
        mark = "FLAG " if abs(z) > 3 else ("watch" if abs(z) > 2 else "")
        if mark or abs(z) > 2:
            flags.append((c, p, p_hat, z, mark))
    if not flags:
        print("  none outside |z|=2 -- calibrated at this resolution")
    for c, p, p_hat, z, mark in flags[:15]:
        print(f"  {mark:6s}{_tok(c):4s} PF {p:.3f} emp {p_hat:.3f} "
              f"z {z:+.1f}")
    # M3 evidence channels
    def channel(tag, cards):
        cards = [c for c in cards if c in unseen]
        if not cards:
            return
        pm = float(sum(raw[c] for c in cards))
        em = float(sum(freq[c] for c in cards))
        se = math.sqrt(sum(freq[c] * (1 - freq[c]) for c in cards) / n)
        print(f"M3 {tag:28s} PF {pm:.3f}  emp {em:.3f} +/- "
              f"{1.96 * se:.3f}  ({', '.join(_tok(c) for c in cards)})")

    def mates(card):
        r, s = card % 13, card // 13
        rank = [x for x in range(52) if x % 13 == r and x != card]
        suit = [s * 13 + q for q in range(max(0, r - 2), min(13, r + 3))
                if q != r]
        return rank, suit
    kind = sc.get("kind", name)
    if kind in ("s1", "decline"):
        ev_cards = [("declined flip", flip)]
    elif kind in ("s2", "take"):
        ev_cards = [("taken flip", flip), ("discarded X", sc["X"])]
    else:
        ev_cards = [("declined flip", flip), ("declined C", sc["C"]),
                    ("discarded Y", sc["Y"])]
    for tag, card in ev_cards:
        rank, suit = mates(card)
        channel(f"rank-mates of {tag} {_tok(card)}", rank)
        channel(f"suit+-2 of {tag} {_tok(card)}", suit)
    # M4 joint structure
    emp_dw = np.array([min_deadwood_ids(list(h)) for h in hands])
    emp_knock = float(np.mean(emp_dw <= thresh))
    cl_dw = []
    cl_w = []
    for b in ens.beliefs:
        w = b.core.weights()
        for wi, p in zip(w, b.core.particles):
            cl_dw.append(min_deadwood_ids(sorted(core.opp_known) +
                                          [int(x) for x in p]))
            cl_w.append(wi / len(ens.beliefs))
    cl_dw = np.array(cl_dw)
    cl_w = np.array(cl_w)
    pf_mean_dw = float(np.sum(cl_dw * cl_w) / cl_w.sum())
    pf_knock = float(np.sum((cl_dw <= thresh) * cl_w) / cl_w.sum())
    print(f"M4 opp deadwood: emp mean {emp_dw.mean():.2f} "
          f"(sd {emp_dw.std():.2f})  PF cloud {pf_mean_dw:.2f}")
    print(f"M4 knockable (dw<=thresh {thresh}): emp {emp_knock:.4f}  "
          f"PF cloud {pf_knock:.4f}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser("posterior ground-truth check")
    ap.add_argument("--scenario", required=True, choices=list(SCEN))
    ap.add_argument("--accepted", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard", type=int, default=10)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--filters", type=int, default=4)
    args = ap.parse_args()
    name = args.scenario
    sc = SCEN[name]
    out = sc["out"]
    if args.report:
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        report(name, sc, rows, filters=args.filters)
        return
    sim.set_rules("full")
    kind = sc.get("kind", name)
    if kind in ("s1", "decline"):
        n = args.accepted or 200000
        rng = random.Random(sc["seed"])
        t0 = time.time()
        rows = _sample_s1(sc, rng, n)
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{name}: {n} accepted / {rows[-1]['tried']} tried "
              f"({time.time() - t0:.0f}s)")
        report(name, sc, rows, filters=args.filters)
        return
    n = args.accepted or 1500
    fn = _sample_s2 if kind in ("s2", "take") else _sample_s3
    jobs = [(sc, args.shard, sc["seed"] + 977 * i)
            for i in range(max(1, n // args.shard))]
    open(out, "w").close()
    t0 = time.time()
    rows = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs), 1):
            with open(out, "a", encoding="utf-8") as fh:
                for row in r:
                    fh.write(json.dumps(row) + "\n")
            rows.extend(r)
            el = time.time() - t0
            print(f"shard {i}/{len(jobs)} ({len(rows)} rows) "
                  f"elapsed {el / 60:.0f}m "
                  f"ETA {el / i * (len(jobs) - i) / 60:.0f}m", flush=True)
    report(name, sc, rows, filters=args.filters)


if __name__ == "__main__":
    main()
