"""
AdaptiveExpert: three single-hand search upgrades on the corrected serial
loop (user-directed, 2026-08-26). No opponent-model learning across hands.

1. ADAPTIVE WORLD BUDGET (`stages`, default 16/48/192): every decision
   starts with a cheap screen; if after a stage every alternative's
   OPTIMISTIC bound (diff + z*se) is already below the prior, stop -- the
   prior was keeping anyway. Close calls escalate, up to stages[-1] worlds.
   Deviating is only allowed at the highest stage actually reached, so the
   sequential looks can only ADD evidence before a deviation -- the z-test
   discipline (lesson #3) stays conservative. Same average cost as v4,
   2-4x the worlds exactly where the 1-3 pt gaps live.

2. TWO-PLY ROLLOUTS (`deep`, from stage index `deep_from`): inside a
   world, at the hero's FIRST multi-action rollout decision, branch over
   the top-`kbranch` pruned actions and take the best continuation.
   Playouts are deterministic within a determinized world, so this max is
   exact 2-ply search, not a noisy argmax. Depth is uniform across
   candidates within a world, so paired diffs stay apples-to-apples.

3. ENDGAME MODE (`endgame_stock`): when the stock is at most this size,
   drop truncation (exact terminal payoffs -- no leaf approximation in
   the undercut-sensitive phase), widen candidates to ALL discards + best
   knock, and start at stage 2.

With stages=(48,) and both features off, decisions are bit-identical to
the corrected-v4 reference (machinery parity, --parity).

    .venv/Scripts/python.exe adaptive.py --parity
    .venv/Scripts/python.exe adaptive.py --time
    .venv/Scripts/python.exe adaptive.py --vs v4 --hands 1500 --seed 98500
"""

import argparse
import math
import multiprocessing as mp
import random
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, KnockAction)

import sim
from belief import Belief
from expert import RolloutExpert
from sim import (DECK, FastTeacher, apply_world, cid, fast_legal_actions,
                 ids_of, leaf_value, make_scratch, min_deadwood_ids, playout,
                 restore, snapshot, step_fast)


def playout_deep(game, policies, budget, root_hero_id, pruner, kbranch):
    """sim.playout semantics, except the ROOT-hero seat's first
    multi-action decision branches over `kbranch` pruned actions and takes
    the best (hero-view) continuation. Returns player-0-view payoff.
    `budget` None = untruncated (max_steps)."""
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
            step_fast(game, policies[rnd.current_player_id].act(game, legal))
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
            v = playout(game, policies, truncate=None if budget is None
                        else b - 1)
            best = max(best, sign * v)
        return sign * best


class AdaptiveExpert(RolloutExpert):
    """Corrected serial loop + staged budget + 2-ply rollouts + endgame."""

    def __init__(self, stages=(16, 48, 192), deep=False, deep_from=2,
                 kbranch=3, endgame_stock=0, mid_alpha=0.5, **kw):
        kw.setdefault("worlds", stages[-1])
        super().__init__(**kw)
        self.stages = tuple(stages)
        self.mid_alpha = mid_alpha
        self.deep = deep
        self.deep_from = deep_from
        self.kbranch = kbranch
        self.endgame_stock = endgame_stock
        self._branch_pruner = RolloutExpert(top_discards=max(1, kbranch - 1))

    def analyze(self, game, belief):
        legal = fast_legal_actions(game)
        endgame = self.endgame_stock > 0 and \
            len(game.round.dealer.stock_pile) <= self.endgame_stock
        if endgame:
            candidates = self._endgame_candidates(game, legal)
        else:
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
            boost = min(3.0, 0.5 * (belief.opp_pickups + belief.opp_declines))
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
        known_ids = ids_of(opp.known_cards)      # pristine (quirk-corrected)
        opp_discard_count = int(belief.opp_discarded.sum())

        k_max = self.stages[-1]
        C = len(candidates)
        payoffs = np.zeros((k_max, C))
        weights = np.zeros(k_max)
        sign = 1.0 if hero_id == 0 else -1.0
        truncate = None if endgame else self.truncate
        first_stage = 1 if (endgame and len(self.stages) > 1) else 0
        attempts = 0
        worlds_done = 0
        stats = None
        stage_reached = first_stage
        for s in range(first_stage, len(self.stages)):
            target = self.stages[s]
            stage_reached = s
            deep_here = self.deep and s >= self.deep_from
            max_attempts = 3 * target
            while worlds_done < target and attempts < max_attempts:
                attempts += 1
                if num_hidden > 0:
                    free_ids = belief.sample_free_cards(unseen, num_hidden,
                                                        self.rng, boost)
                else:
                    free_ids = []
                weight = self._world_weight(known_ids + list(free_ids),
                                            opp_discard_count)
                if weight <= 0.0:
                    continue
                restore(scratch, base)
                apply_world(scratch, opp_id, [DECK[i] for i in free_ids],
                            self.rng)
                world = snapshot(scratch)
                for k, action in enumerate(candidates):
                    restore(scratch, world)
                    step_fast(scratch, action)
                    if deep_here:
                        v = playout_deep(scratch, rollout, truncate,
                                         hero_id, self._branch_pruner,
                                         self.kbranch)
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
                # Stage screens (stop early => keep the prior, which the
                # final z-test would do anyway for the screened cases):
                #  - first stage: escalate iff some alternative's point
                #    estimate beats the prior at all;
                #  - later stages: escalate iff a deviation is within
                #    REACH -- mean > mid_alpha * z * se -- so borderline
                #    cases (incl. every would-be deviation, mean > z*se)
                #    get re-verified with more worlds, while small positive
                #    blips that could never clear the final test stop here.
                thresh = 0.0 if s == 0 else \
                    self.mid_alpha * self.z * se
                if not np.any(means > thresh):
                    break                # prior confirmed -- stop early

        if worlds_done == 0 or weights[:worlds_done].sum() <= 0.0:
            return {"chosen": teacher_action, "reason": "no-worlds",
                    "candidates": candidates}
        means, se, n_eff, payoff_means = stats
        gain = means - self.z * se
        best = int(np.argmax(gain))
        # Deviate only with the evidence of the highest stage we actually
        # ran; an early "prior confirmed" exit keeps the prior by design.
        deviate = gain[best] > 0.0 and \
            worlds_done >= min(self.stages[stage_reached], worlds_done)
        return {"chosen": candidates[best] if deviate else teacher_action,
                "reason": "deviate" if deviate else "prior",
                "candidates": candidates, "t_idx": t_idx,
                "payoff_means": payoff_means, "diff_means": means,
                "se": se, "gain": gain, "worlds": worlds_done,
                "n_eff": float(n_eff)}

    def _stats(self, payoffs, weights, worlds_done, t_idx):
        w = weights[:worlds_done]
        diffs = payoffs[:worlds_done] - payoffs[:worlds_done, [t_idx]]
        w_sum = w.sum()
        means = (w[:, None] * diffs).sum(axis=0) / w_sum
        centered = diffs - means
        var = (w[:, None] * centered ** 2).sum(axis=0) / w_sum
        n_eff = w_sum ** 2 / (w ** 2).sum()
        se = np.sqrt(var / max(n_eff - 1.0, 1.0))
        payoff_means = (w[:, None] * payoffs[:worlds_done]).sum(axis=0) \
            / w_sum
        return means, se, n_eff, payoff_means

    def _endgame_candidates(self, game, legal):
        """All discards + deadwood-best knock (no top-4 pruning)."""
        if len(legal) == 1:
            return legal
        discards = [e for e in legal if isinstance(e, DiscardAction)]
        knocks = [e for e in legal if isinstance(e, KnockAction)]
        if not discards and not knocks:
            return legal
        rnd = game.round
        hand_ids = ids_of(rnd.players[rnd.current_player_id].hand)
        chosen = list(discards)
        if knocks:
            best, best_dw = None, 999
            for event in knocks:
                drop = cid(event.card)
                dw = min_deadwood_ids([i for i in hand_ids if i != drop])
                if dw < best_dw:
                    best, best_dw = event, dw
            chosen.append(best)
        return chosen


# ---------------------------------------------------------------------------
# Harness (btmatch pattern; config as a plain dict through the job tuple).
# ---------------------------------------------------------------------------

def _build(cfg, seed):
    return AdaptiveExpert(
        stages=tuple(cfg["stages"]), deep=cfg["deep"],
        deep_from=cfg["deep_from"], kbranch=cfg["kbranch"],
        endgame_stock=cfg["endgame_stock"],
        mid_alpha=cfg.get("mid_alpha", 0.5), truncate=cfg["truncate"],
        z=cfg["z"], opp_model=(10, 1.0), sample_boost=2.5, seed=seed)


class _AdaptivePolicy:
    def __init__(self, cfg, seed=0):
        self.expert = _build(cfg, seed)
        self._round_key = None
        self._belief = None
        self.move_time, self.move_count = 0.0, 0

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._belief = Belief(rnd.current_player_id)
        t0 = time.perf_counter()
        action = self.expert.decide(game, self._belief)
        self.move_time += time.perf_counter() - t0
        self.move_count += 1
        return action


def _amatch_chunk(job):
    cfg, opp_name, hands, seat, seed, rules = job
    sim.set_rules(rules)
    import meta
    rng = random.Random(seed)
    pol_a = _AdaptivePolicy(cfg, seed=seed)
    if opp_name.startswith("v4k"):       # plain fixed-K v4 (compute control)
        k = int(opp_name[3:])
        pol_b = _AdaptivePolicy({"stages": [k], "deep": False,
                                 "deep_from": 2, "kbranch": 3,
                                 "endgame_stock": 0, "truncate": 16,
                                 "z": 1.5}, seed=seed + 7)
    else:
        pol_b = meta.population()[opp_name]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    return points, pol_a.move_time, pol_a.move_count


def amatch(cfg, opp_name, hands, workers, seed, rules="full"):
    per = max(1, hands // (2 * workers))
    jobs = [(cfg, opp_name, per, seat, seed + 31 * w + seat, rules)
            for w in range(workers) for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_amatch_chunk, jobs)
    points = np.concatenate([r[0] for r in results])
    move_t = sum(r[1] for r in results)
    move_n = sum(r[2] for r in results)
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    tag = (f"stages={','.join(map(str, cfg['stages']))}"
           f"{',deep' if cfg['deep'] else ''}"
           f"{',eg' + str(cfg['endgame_stock']) if cfg['endgame_stock'] else ''}")
    print(f"adaptive[{tag}] vs {opp_name} [{rules}]: {mean:+.2f} +/- "
          f"{ci:.2f} over {len(points)} hands  "
          f"({move_t / max(move_n, 1):.3f} s/searched-move, "
          f"{time.time() - t0:.0f}s wall)", flush=True)
    return mean, ci, len(points)


def parity():
    """stages=(48,), features off == corrected-v4 reference, bit-exact."""
    sim.set_rules("full")
    from q_expert import _RefExpert, _decision_stream
    cfgs = ((0.0, (10, 1.0)), (2.5, (10, 1.0)))
    base = _decision_stream(lambda b, m: _RefExpert(
        worlds=48, truncate=16, z=1.5, opp_model=m, seed=11,
        sample_boost=b), cfgs)
    mine = _decision_stream(lambda b, m: AdaptiveExpert(
        stages=(48,), truncate=16, z=1.5, opp_model=m, seed=11,
        sample_boost=b), cfgs)
    same = base == mine
    print(f"adaptive parity: {len(base)} decisions, "
          f"{'IDENTICAL -- PASS' if same else 'DIFFER -- FAIL'}")
    return same


def timing(cfg, hands=12, seed=90500):
    sim.set_rules("full")
    from q_expert import _time_expert
    v4 = RolloutExpert(worlds=48, truncate=16, z=1.5, opp_model=(10, 1.0),
                       sample_boost=2.5, seed=0)
    t, n = _time_expert(v4, hands, seed)
    print(f"v4: {t:.3f} s/searched-move ({n})", flush=True)
    ex = _build(cfg, 0)
    t2, n2 = _time_expert(ex, hands, seed)
    print(f"adaptive: {t2:.3f} s/searched-move ({n2})  ratio {t2 / t:.2f}x",
          flush=True)


def main():
    ap = argparse.ArgumentParser("adaptive search gates")
    ap.add_argument("--vs", default="v4")
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stages", default="16,48,192")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--deep_from", type=int, default=2)
    ap.add_argument("--kbranch", type=int, default=3)
    ap.add_argument("--endgame_stock", type=int, default=0)
    ap.add_argument("--mid_alpha", type=float, default=0.5)
    ap.add_argument("--truncate", type=int, default=16)
    ap.add_argument("--z", type=float, default=1.5)
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=98500)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--time", dest="timing", action="store_true")
    args = ap.parse_args()
    cfg = {"stages": [int(s) for s in args.stages.split(",")],
           "deep": args.deep, "deep_from": args.deep_from,
           "kbranch": args.kbranch, "endgame_stock": args.endgame_stock,
           "mid_alpha": args.mid_alpha, "truncate": args.truncate,
           "z": args.z}
    if args.parity:
        parity()
        return
    if args.timing:
        timing(cfg)
        return
    amatch(cfg, args.vs, args.hands, args.workers, args.seed, args.rules)


if __name__ == "__main__":
    main()
