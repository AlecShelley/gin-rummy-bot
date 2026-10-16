"""
Correlated belief sampling, step 1: measure the problem (roadmap #2).

Claim to verify: `Belief.sample_free_cards` (independent per-card marginals)
produces structurally meldless opponent hands, so the search underestimates
opponent progress. Here we play teacher-vs-teacher hands, and at each hero
decision compare the TRUE hidden opponent hand against worlds drawn from
(a) the current marginal sampler and (b) a sequential sampler that multiplies
the remaining weights of meld-neighbors by (1 + boost) after each pick.

Statistics per sampled/true opponent hand (known + hidden cards):
  * deadwood (min_deadwood_ids) -- lower = more melded;
  * adjacency pairs -- count of card pairs that could share a meld.

    .venv/Scripts/python.exe corr_sample.py --hands 300
"""

import argparse
import random

import numpy as np

import sim
from belief import _ADJ, Belief
from sim import FastTeacher, cid, fast_legal_actions, ids_of, \
    min_deadwood_ids, step_fast


def adj_pairs(hand_ids):
    s = set(hand_ids)
    return sum(1 for c in hand_ids for d in _ADJ[c]
               if d in s and d > c)


def sample_sequential(belief, unseen_ids, num_hidden, rng, boost):
    """Pick hidden cards one at a time; each pick multiplies remaining
    meld-neighbor weights by (1 + boost). boost=0 == independent sampling."""
    pool = list(unseen_ids)
    w = belief.w[pool].astype(np.float64).copy()
    picked = []
    for _ in range(num_hidden):
        p = w / w.sum()
        k = rng.choice(len(pool), p=p)
        c = pool.pop(k)
        w = np.delete(w, k)
        picked.append(c)
        if boost > 0.0 and pool:
            adj = set(_ADJ[c])
            for i, d in enumerate(pool):
                if d in adj:
                    w[i] *= (1.0 + boost)
    return picked


def run(hands=300, seed=424, samples=16, boosts=(0.5, 1.0, 2.0)):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    t0, t1 = FastTeacher(), FastTeacher()
    stats = {"true": [], "marginal": []}
    for b in boosts:
        stats[f"seq_b{b}"] = []

    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        rnd = game.round
        belief = Belief(0)
        while rnd.going_out_action is None and not rnd.is_over:
            legal = fast_legal_actions(game)
            if len(legal) > 1 and rnd.current_player_id == 0:
                belief.update(game)
                opp = rnd.players[1]
                known = ids_of(opp.known_cards)
                hidden = [cid(c) for c in opp.hand
                          if c not in opp.known_cards]
                unseen = ids_of(rnd.dealer.stock_pile) + hidden
                m = len(hidden)
                if m >= 3 and len(opp.hand) == 10:
                    stats["true"].append(
                        (min_deadwood_ids(known + hidden),
                         adj_pairs(known + hidden)))
                    for _ in range(samples):
                        free = belief.sample_free_cards(unseen, m, nrng)
                        stats["marginal"].append(
                            (min_deadwood_ids(known + free),
                             adj_pairs(known + free)))
                    for b in boosts:
                        for _ in range(samples // 4):
                            free = sample_sequential(belief, unseen, m,
                                                     nrng, b)
                            stats[f"seq_b{b}"].append(
                                (min_deadwood_ids(known + free),
                                 adj_pairs(known + free)))
            action = legal[0] if len(legal) == 1 else \
                (t0 if rnd.current_player_id == 0 else t1).act(game, legal)
            step_fast(game, action)

    print(f"{len(stats['true'])} probe states, {samples} samples each")
    for name, rows in stats.items():
        a = np.array(rows, dtype=float)
        print(f"  {name:>10}: deadwood {a[:, 0].mean():6.2f}  "
              f"adj-pairs {a[:, 1].mean():5.2f}  (n={len(a)})")


def run_stages(hands=400, seed=424, samples=8,
               boosts=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """True opponent-hand deadwood binned by how many turns the opponent has
    taken, vs each boost's sampled deadwood in the same states. Calibrates
    the stage-dependent boost ramp (a flat boost matches the aggregate mean
    but overrates turn-1 opponents and underrates late ones)."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    t0, t1 = FastTeacher(), FastTeacher()
    bins = {}                            # opp_turns bucket -> {name: [dw...]}

    def bucket(turns):
        return min(turns, 10)

    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        rnd = game.round
        belief = Belief(0)
        while rnd.going_out_action is None and not rnd.is_over:
            legal = fast_legal_actions(game)
            if len(legal) > 1 and rnd.current_player_id == 0:
                belief.update(game)
                opp = rnd.players[1]
                known = ids_of(opp.known_cards)
                hidden = [cid(c) for c in opp.hand
                          if c not in opp.known_cards]
                unseen = ids_of(rnd.dealer.stock_pile) + hidden
                m = len(hidden)
                if m >= 3 and len(opp.hand) == 10:
                    b = bucket(belief.opp_pickups + belief.opp_declines)
                    rows = bins.setdefault(
                        b, {"true": [], **{f"b{x}": [] for x in boosts}})
                    rows["true"].append(min_deadwood_ids(known + hidden))
                    for x in boosts:
                        for _ in range(samples):
                            if x == 0.0:
                                free = belief.sample_free_cards(unseen, m,
                                                                nrng)
                            else:
                                free = sample_sequential(belief, unseen, m,
                                                         nrng, x)
                            rows[f"b{x}"].append(
                                min_deadwood_ids(known + free))
            action = legal[0] if len(legal) == 1 else \
                (t0 if rnd.current_player_id == 0 else t1).act(game, legal)
            step_fast(game, action)

    print("opp_turns |  true | " + " | ".join(f"b={x}" for x in boosts)
          + " | best b")
    for b in sorted(bins):
        rows = bins[b]
        true_m = np.mean(rows["true"])
        means = [np.mean(rows[f"b{x}"]) for x in boosts]
        best = boosts[int(np.argmin([abs(mm - true_m) for mm in means]))]
        print(f"{b:>9} | {true_m:5.1f} | "
              + " | ".join(f"{mm:4.1f}" for mm in means)
              + f" | {best}  (n={len(rows['true'])})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("correlated sampling diagnostics")
    ap.add_argument("--hands", type=int, default=300)
    ap.add_argument("--seed", type=int, default=424)
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()
    if args.stages:
        run_stages(args.hands, args.seed)
    else:
        run(args.hands, args.seed)
