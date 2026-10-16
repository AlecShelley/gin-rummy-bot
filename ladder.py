"""
Track 3 (v6): 3-ply close-call rollouts -- the escalation ladder's next
accuracy-per-world rung.

v5's winning mechanism was playout_deep: at the final stage, inside a
world, the hero's FIRST multi-action rollout decision branches top-k and
takes the exact (deterministic-within-world) max. This extends it one
decision deeper: the SECOND hero decision in each branch also branches.
Rationale: the 16/48/384 scaling probe measured that MORE WORLDS buy
nothing on the particle posterior (-0.19 +/- 1.63 at 2x cost) --
accuracy-per-world is the live vein, exactly v5's lesson.

Fusion caveat (results/v6_prior_art.md section 1): per-world hero maxing
is strategy fusion whose deferral bias grows with depth; the first
branch is mostly legitimate draw disambiguation, the second is where the
tax may start. The gate decides.

Snapshot discipline: each nesting level restores only its OWN ancestor
snapshot within its own line, so plain sim.snapshot/restore is sound
(the endgame _BSnap hazard applies only to cross-branch SIBLING
restores).

    .venv/Scripts/python.exe ladder.py --parity
    .venv/Scripts/python.exe ladder.py --vs v6 --hands 1500 --seed 67100
"""

import argparse
import math
import multiprocessing as mp
import random
import time

import numpy as np

import sim
from adaptive import AdaptiveExpert
from pf_expert import (ParticleBelief, PFExpert, V5_CFG, _chunk as
                       _pf_chunk)
from sim import (fast_legal_actions, leaf_value, playout, restore,
                 snapshot, step_fast)


def playout_deep_n(game, policies, budget, root_hero_id, pruner, kbranch,
                   branches):
    """playout_deep with `branches` hero branch points instead of one.
    branches=1 reproduces adaptive.playout_deep exactly."""
    rnd = game.round
    b = 250 if budget is None else budget
    trunc = None if budget is None else True
    while True:
        if rnd.going_out_action is not None or rnd.is_over:
            return sim._payoff(game)
        if b == 0:
            return leaf_value(game) if trunc else sim._payoff(game)
        legal = fast_legal_actions(game)
        if len(legal) == 1:
            step_fast(game, legal[0])
            b -= 1
            continue
        if rnd.current_player_id != root_hero_id:
            step_fast(game, policies[rnd.current_player_id].act(game,
                                                                legal))
            b -= 1
            continue
        cands = pruner._candidates(game, legal)[:kbranch]
        if len(cands) == 1:
            step_fast(game, cands[0])
            b -= 1
            continue
        snap = snapshot(game)
        sign = 1.0 if root_hero_id == 0 else -1.0
        best = -1e9
        for a in cands:
            restore(game, snap)
            step_fast(game, a)
            nb = None if budget is None else b - 1
            if branches > 1:
                v = playout_deep_n(game, policies, nb, root_hero_id,
                                   pruner, kbranch, branches - 1)
            else:
                v = playout(game, policies, truncate=nb)
            best = max(best, sign * v)
        return sign * best


class LadderExpert(PFExpert):
    """PFExpert whose final-stage deep rollouts use `deep_plies` hero
    branch points (1 == v6 exactly)."""

    def __init__(self, deep_plies=2, **kw):
        super().__init__(**kw)
        self.deep_plies = deep_plies

    def analyze(self, game, belief):
        # Identical to AdaptiveExpert.analyze except the deep dispatch;
        # kept in sync by the --parity gate (deep_plies=1 bit-exact).
        from rlcard.games.gin_rummy.utils.action_event import (
            DiscardAction, KnockAction)
        from sim import DECK, apply_world, cid, ids_of, make_scratch

        legal = fast_legal_actions(game)
        candidates = self._candidates(game, legal)
        if len(candidates) == 1:
            return {"chosen": candidates[0], "reason": "forced",
                    "candidates": candidates}
        has_discard = any(isinstance(e, DiscardAction) for e in legal)
        has_knock = any(isinstance(e, KnockAction) for e in legal)
        node = "discard" if has_discard or has_knock else "draw"
        if node not in self.classes:
            return {"chosen": self.teacher.act(game, legal),
                    "reason": "class-scoped", "candidates": candidates}

        hero_id = game.round.current_player_id
        opp_id = 1 - hero_id
        belief.update(game)
        boost = self.sample_boost
        if boost == "auto":
            boost = min(3.0, 0.5 * (belief.opp_pickups +
                                    belief.opp_declines))
        teacher_action = self.teacher.act(game, legal)
        t_idx = next((k for k, a in enumerate(candidates)
                      if a.action_id == teacher_action.action_id), None)
        if t_idx is None:
            candidates.insert(0, teacher_action)
            t_idx = 0

        rollout = self.rollout if self._opp_rollout is None else \
            {hero_id: self.rollout[hero_id], opp_id: self._opp_rollout}
        scratch = make_scratch(game)
        base = snapshot(scratch)
        opp = scratch.round.players[opp_id]
        unseen = ids_of(scratch.round.dealer.stock_pile) + \
            [cid(c) for c in opp.hand if c not in opp.known_cards]
        num_hidden = len(opp.hand) - len(opp.known_cards)
        known_ids = ids_of(opp.known_cards)
        opp_discard_count = int(belief.opp_discarded.sum())

        k_max = self.stages[-1]
        C = len(candidates)
        payoffs = np.zeros((k_max, C))
        weights = np.zeros(k_max)
        sign = 1.0 if hero_id == 0 else -1.0
        truncate = self.truncate
        attempts = 0
        worlds_done = 0
        stats = None
        for s in range(len(self.stages)):
            target = self.stages[s]
            deep_here = self.deep and s >= self.deep_from
            max_attempts = 3 * target
            while worlds_done < target and attempts < max_attempts:
                attempts += 1
                if num_hidden > 0:
                    free_ids = belief.sample_free_cards(
                        unseen, num_hidden, self.rng, boost)
                else:
                    free_ids = []
                weight = self._world_weight(known_ids + list(free_ids),
                                            opp_discard_count)
                if weight <= 0.0:
                    continue
                restore(scratch, base)
                apply_world(scratch, opp_id,
                            [DECK[i] for i in free_ids], self.rng)
                world = snapshot(scratch)
                for k, action in enumerate(candidates):
                    restore(scratch, world)
                    step_fast(scratch, action)
                    if deep_here:
                        v = playout_deep_n(scratch, rollout, truncate,
                                           hero_id, self._branch_pruner,
                                           self.kbranch,
                                           self.deep_plies)
                    else:
                        v = playout(scratch, rollout, truncate=truncate)
                    payoffs[worlds_done, k] = sign * v
                weights[worlds_done] = weight
                worlds_done += 1
            if worlds_done == 0:
                break
            stats = self._stats(payoffs, weights, worlds_done, t_idx)
            if s < len(self.stages) - 1:
                means, se = stats[0].copy(), stats[1]
                means[t_idx] = -1e9
                thresh = 0.0 if s == 0 else self.mid_alpha * self.z * se
                if not np.any(means > thresh):
                    break
        if worlds_done == 0 or weights[:worlds_done].sum() <= 0.0:
            return {"chosen": teacher_action, "reason": "no-worlds",
                    "candidates": candidates}
        means, se, n_eff, payoff_means = stats
        gain = means - self.z * se
        best = int(np.argmax(gain))
        deviate = gain[best] > 0.0
        return {"chosen": candidates[best] if deviate else teacher_action,
                "reason": "deviate" if deviate else "prior",
                "candidates": candidates, "t_idx": t_idx,
                "payoff_means": payoff_means, "diff_means": means,
                "se": se, "gain": gain, "worlds": worlds_done,
                "n_eff": float(n_eff)}


class LadderPolicy:
    def __init__(self, seed=0, deep_plies=2, veto=True):
        self.expert = LadderExpert(deep_plies=deep_plies, veto=veto,
                                   opp_model=(10, 1.0), sample_boost=0.0,
                                   seed=seed, **V5_CFG)
        self.seed = seed
        self._round_key = None
        self._belief = None
        self._hand_no = 0
        self.move_time, self.move_count = 0.0, 0

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._hand_no += 1
            self._belief = ParticleBelief(
                rnd.current_player_id,
                seed=self.seed * 100003 + self._hand_no)
        t0 = time.perf_counter()
        action = self.expert.decide(game, self._belief)
        self.move_time += time.perf_counter() - t0
        self.move_count += 1
        return action


def _chunk(job):
    opp_name, hands, seat, seed, rules, deep_plies, veto = job
    sim.set_rules(rules)
    import meta
    rng = random.Random(seed)
    pol_a = LadderPolicy(seed=seed, deep_plies=deep_plies, veto=veto)
    pol_b = meta.population()[opp_name]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    return points, pol_a.move_time, pol_a.move_count


def lmatch(opp_name, hands, workers, seed, rules="full", deep_plies=2,
           veto=True):
    per = max(1, hands // (2 * workers))
    jobs = [(opp_name, per, seat, seed + 31 * w + seat, rules,
             deep_plies, veto) for w in range(workers)
            for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_chunk, jobs)
    points = np.concatenate([r[0] for r in results])
    move_t = sum(r[1] for r in results)
    move_n = sum(r[2] for r in results)
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    print(f"ladder[p{deep_plies},veto={veto}] vs {opp_name} [{rules}]: "
          f"{mean:+.2f} +/- {ci:.2f} over {len(points)} hands "
          f"({move_t / max(move_n, 1):.3f} s/searched-move, "
          f"{time.time() - t0:.0f}s wall)", flush=True)
    return mean, ci, len(points)


def parity():
    """deep_plies=1 must be bit-identical to PFExpert == v6. Uses the
    classic Belief (deterministic stream) for the comparison since the
    particle belief's own rng would desynchronize any two experts."""
    sim.set_rules("full")
    from q_expert import _decision_stream
    cfgs = ((2.5, (10, 1.0)),)
    base = _decision_stream(lambda b, m: AdaptiveExpert(
        stages=(16, 48, 192), deep=True, deep_from=2, kbranch=3,
        truncate=16, z=1.5, opp_model=m, seed=11, sample_boost=b), cfgs)
    mine = _decision_stream(lambda b, m: LadderExpert(
        deep_plies=1, veto=True, stages=(16, 48, 192), deep=True,
        deep_from=2, kbranch=3, truncate=16, z=1.5, opp_model=m,
        seed=11, sample_boost=b), cfgs)
    same = base == mine
    print(f"ladder parity: {len(base)} decisions, "
          f"{'IDENTICAL -- PASS' if same else 'DIFFER -- FAIL'}")
    return same


def main():
    ap = argparse.ArgumentParser("v6 3-ply ladder gates")
    ap.add_argument("--vs", default="v6")
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=67100)
    ap.add_argument("--plies", type=int, default=2)
    ap.add_argument("--no-veto", dest="veto", action="store_false")
    ap.add_argument("--parity", action="store_true")
    args = ap.parse_args()
    if args.parity:
        parity()
        return
    lmatch(args.vs, args.hands, args.workers, args.seed, args.rules,
           args.plies, args.veto)


if __name__ == "__main__":
    main()
