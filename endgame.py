"""
Track 2 (v6): cross-world exact endgame search.

When the stock is short (endgame_stock), the final escalation stage
replaces truncated/2-ply rollouts with OBSERVATION-GROUPED backward
induction over all collected worlds:

  * advance each world with the rollout policies until the hero's next
    real decision (or terminal), recording the hero-VISIBLE observation
    trace (opponent discard/pickup/decline cards, the hero's own drawn
    card);
  * partition worlds by that observation; within a partition the hero
    must choose ONE move (teacher-pruned top-k) -- no strategy fusion;
    different partitions may choose differently -- legitimate
    disambiguation, the hero really observes the difference;
  * recurse to `depth` hero decisions, then finish each world with a
    teacher playout; terminal payoffs are exact full-rules _payoff
    (no leaf approximation in the undercut-sensitive phase);
  * per-world values flow back up (each world ends in exactly one leaf
    under the chosen grouped strategy), so the v5 paired z-test versus
    the teacher prior applies UNCHANGED at the root.

Survey grounding (results/v6_prior_art.md section 1): per-world max is
strategy fusion whose deferral bias is candidate-asymmetric -- the
z-test would certify it (that is how endgame-v1 lost, lesson #13). The
sound constructions are cross-world (GIB achievable sets, alpha-mu same-
move-in-all-worlds); this is that, specialized to a modeled (not
adversarial) opponent. Root candidates stay top-4 (lesson #13's other
half: precision upgrades must not widen the comparison surface).

    .venv/Scripts/python.exe endgame.py --parity
    .venv/Scripts/python.exe endgame.py --vs v5 --hands 1500 --seed 63100
"""

import argparse
import math
import multiprocessing as mp
import random
import time

import numpy as np

from rlcard.games.gin_rummy.utils.action_event import (
    DiscardAction, DrawCardAction, KnockAction, PickUpDiscardAction)

import sim
from adaptive import AdaptiveExpert
from belief import Belief
from expert import RolloutExpert
from sim import (DECK, apply_world, cid, fast_legal_actions, ids_of,
                 make_scratch, playout, restore, snapshot, step_fast)

V5_CFG = dict(stages=(16, 48, 192), deep=True, deep_from=2, kbranch=3,
              endgame_stock=0, mid_alpha=0.5, truncate=16, z=1.5)


class _BSnap:
    """Branch-safe snapshot. sim.restore only TRUNCATES move_sheet /
    game.actions (their contents are assumed to be a shared prefix --
    true for v5's same-point sibling worlds, FALSE for states on
    diverged search branches). The engine reads the PHASE from the last
    entries, so restoring a sibling with plain restore() leaves another
    branch's tail in place and corrupts fast_legal_actions (observed:
    scoring actions surfacing mid-hand). This wrapper additionally
    captures the tails beyond a common base and rebuilds them exactly."""

    __slots__ = ("snap", "tail_m", "tail_a", "base_m", "base_a")

    def __init__(self, game, base_m, base_a):
        self.snap = snapshot(game)
        self.base_m = base_m
        self.base_a = base_a
        self.tail_m = list(game.round.move_sheet[base_m:])
        self.tail_a = list(game.actions[base_a:])


def _brestore(game, bs):
    sheet = game.round.move_sheet
    del sheet[bs.base_m:]
    sheet.extend(bs.tail_m)
    del game.actions[bs.base_a:]
    game.actions.extend(bs.tail_a)
    restore(game, bs.snap)


def _advance(game, policies, hero_id, max_steps=250):
    """Step `game` with `policies` until the hero faces a real (>1
    option) decision or the hand ends. Returns (obs_key, terminal,
    value_if_terminal). The obs key contains only hero-visible facts."""
    rnd = game.round
    obs = []
    for _ in range(max_steps):
        if rnd.going_out_action is not None or rnd.is_over:
            return tuple(obs), True, sim._payoff(game)
        legal = fast_legal_actions(game)
        if rnd.current_player_id == hero_id and len(legal) > 1:
            return tuple(obs), False, 0.0
        if len(legal) == 1:
            action = legal[0]
        else:
            action = policies[rnd.current_player_id].act(game, legal)
        mover = rnd.current_player_id
        if isinstance(action, DrawCardAction):
            if mover == hero_id:
                # the hero sees their own draw: stock pops from the end
                obs.append(("hd", cid(rnd.dealer.stock_pile[-1])))
            else:
                obs.append(("od",))
        elif isinstance(action, PickUpDiscardAction):
            obs.append(("p", mover == hero_id,
                        cid(rnd.dealer.discard_pile[-1])))
        elif isinstance(action, DiscardAction):
            obs.append(("dc", mover == hero_id, cid(action.card)))
        else:
            obs.append(("go", mover == hero_id, action.action_id))
        step_fast(game, action)
    return tuple(obs), True, sim._payoff(game)


class ExactEndgameExpert(AdaptiveExpert):
    """v5 staged search; in endgame positions the FINAL stage evaluates
    candidates by cross-world grouped backward induction instead of
    per-world (deep) playouts. With exact_stock=0, decisions are
    bit-identical to AdaptiveExpert (machinery parity)."""

    def __init__(self, exact_stock=6, exact_depth=2, exact_k=3, **kw):
        super().__init__(**kw)
        self.exact_stock = exact_stock
        self.exact_depth = exact_depth
        self.exact_k = exact_k
        self._exact_pruner = RolloutExpert(top_discards=max(1,
                                                            exact_k - 1))
        self.eg_decisions = 0            # endgame-mode decisions seen
        self.eg_exact = 0                # ... that escalated to exact

    # -- the cross-world evaluator ------------------------------------------

    @staticmethod
    def _res(scratch, w):
        """Restore either snapshot flavor (root worlds are plain Snaps
        at a shared point; recursion states are branch-safe _BSnaps)."""
        if isinstance(w, _BSnap):
            _brestore(scratch, w)
        else:
            restore(scratch, w)

    def _group_value(self, scratch, worlds, weights, rollout, hero_id,
                     depth):
        """Value of a set of worlds that share the hero's observation
        history, all standing AT a hero decision (same hero hand).
        Returns per-world values (hero view, aligned with `worlds`)."""
        sign = 1.0 if hero_id == 0 else -1.0
        self._res(scratch, worlds[0])
        legal = fast_legal_actions(scratch)
        cands = self._exact_pruner._candidates(scratch, legal)
        cands = cands[:self.exact_k]
        if len(cands) == 1 or depth == 0:
            # single option (apply it) or budget out (the rollout policy
            # plays on): finish each world with an exact playout.
            vals = np.empty(len(worlds))
            for i, w in enumerate(worlds):
                self._res(scratch, w)
                if len(cands) == 1:
                    step_fast(scratch, cands[0])
                vals[i] = sign * playout(scratch, rollout,
                                         truncate=None)
            return vals
        best_vals, best_mean = None, -1e18
        for action in cands:
            vals = self._candidate_values(scratch, worlds, weights,
                                          rollout, hero_id, action,
                                          depth)
            mean = float(np.average(vals, weights=weights))
            if mean > best_mean:
                best_mean, best_vals = mean, vals
        return best_vals

    def _candidate_values(self, scratch, worlds, weights, rollout,
                          hero_id, action, depth):
        """Per-world values of playing `action` in every world, then
        continuing with the grouped-optimal strategy."""
        sign = 1.0 if hero_id == 0 else -1.0
        n = len(worlds)
        vals = np.empty(n)
        groups = {}
        act_is_draw = isinstance(action, DrawCardAction)
        for i, w in enumerate(worlds):
            self._res(scratch, w)
            # The applied action's own observable outcome must join the
            # group key: a hero DRAW yields a world-specific card BEFORE
            # _advance starts recording -- without this, worlds with
            # different drawn cards share a group and the partition
            # invariant (identical hero hands) breaks.
            drawn = cid(scratch.round.dealer.stock_pile[-1]) \
                if act_is_draw else -1
            step_fast(scratch, action)
            obs, terminal, tval = _advance(scratch, rollout, hero_id)
            obs = (drawn,) + obs
            if terminal:
                vals[i] = sign * tval
            else:
                groups.setdefault(obs, ([], [], []))
                g = groups[obs]
                g[0].append(i)
                g[1].append(_BSnap(scratch, self._base_m, self._base_a))
                g[2].append(weights[i])
        for obs, (idxs, snaps, ws) in groups.items():
            sub = self._group_value(scratch, snaps, np.array(ws),
                                    rollout, hero_id, depth - 1)
            for j, i in enumerate(idxs):
                vals[i] = sub[j]
        return vals

    # -- the decision (v5 staged loop with an exact final stage) ------------

    def analyze(self, game, belief):
        legal = fast_legal_actions(game)
        exact_here = self.exact_stock > 0 and \
            len(game.round.dealer.stock_pile) <= self.exact_stock
        if not exact_here:
            return super().analyze(game, belief)
        candidates = self._candidates(game, legal)   # top-4: lesson #13
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
        # branch-safe snapshot base: recursion states rebuild their
        # move_sheet/actions tails from here (see _BSnap)
        self._base_m = len(scratch.round.move_sheet)
        self._base_a = len(scratch.actions)
        opp = scratch.round.players[opp_id]
        unseen = ids_of(scratch.round.dealer.stock_pile) + \
            [cid(c) for c in opp.hand if c not in opp.known_cards]
        num_hidden = len(opp.hand) - len(opp.known_cards)
        known_ids = ids_of(opp.known_cards)
        opp_discard_count = int(belief.opp_discarded.sum())

        # Stage 1: cheap screen with plain untruncated playouts (exact
        # payoffs, no leaf) on stages[0] worlds; stage 2: the surviving
        # close calls get the cross-world grouped search on all
        # stages[1] worlds. Conservative: early stop only confirms the
        # prior (thresh 0 < z*se).
        k_screen = self.stages[0]
        k_full = self.stages[1] if len(self.stages) > 1 else k_screen
        worlds, weights = [], []
        attempts, max_attempts = 0, 3 * k_full
        while len(worlds) < k_full and attempts < max_attempts:
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
            worlds.append(snapshot(scratch))
            weights.append(weight)
        if not worlds:
            return {"chosen": teacher_action, "reason": "no-worlds",
                    "candidates": candidates}
        weights = np.array(weights)
        C = len(candidates)
        sign = 1.0 if hero_id == 0 else -1.0

        n_scr = min(k_screen, len(worlds))
        payoffs = np.zeros((len(worlds), C))
        for k, action in enumerate(candidates):
            for i in range(n_scr):
                restore(scratch, worlds[i])
                step_fast(scratch, action)
                payoffs[i, k] = sign * playout(scratch, rollout,
                                               truncate=None)
        self.eg_decisions += 1
        stats = self._stats(payoffs, weights, n_scr, t_idx)
        means = stats[0].copy()
        means[t_idx] = -1e9
        if np.any(means > 0.0) and len(worlds) > 0:
            # close call: full cross-world exact evaluation
            self.eg_exact += 1
            for k, action in enumerate(candidates):
                payoffs[:, k] = self._candidate_values(
                    scratch, worlds, weights, rollout, hero_id, action,
                    self.exact_depth)
            stats = self._stats(payoffs, weights, len(worlds), t_idx)
        means, se, n_eff, payoff_means = stats
        gain = means - self.z * se
        best = int(np.argmax(gain))
        deviate = gain[best] > 0.0
        return {"chosen": candidates[best] if deviate else teacher_action,
                "reason": "deviate" if deviate else "prior",
                "candidates": candidates, "t_idx": t_idx,
                "payoff_means": payoff_means, "diff_means": means,
                "se": se, "gain": gain, "worlds": len(worlds),
                "n_eff": float(n_eff)}


# ---------------------------------------------------------------------------
# Harness (oracle_gate/pf_expert pattern).
# ---------------------------------------------------------------------------

class EGPolicy:
    def __init__(self, seed=0, exact_stock=6, exact_depth=2, exact_k=3,
                 pf=False):
        # sample_boost is unused when pf=True (ParticleBelief ignores it)
        self.expert = ExactEndgameExpert(
            exact_stock=exact_stock, exact_depth=exact_depth,
            exact_k=exact_k, opp_model=(10, 1.0),
            sample_boost=0.0 if pf else 2.5, seed=seed, **V5_CFG)
        self.pf = pf
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
            if self.pf:
                from pf_expert import ParticleBelief
                self._belief = ParticleBelief(
                    rnd.current_player_id,
                    seed=self.seed * 100003 + self._hand_no)
            else:
                self._belief = Belief(rnd.current_player_id)
        t0 = time.perf_counter()
        action = self.expert.decide(game, self._belief)
        self.move_time += time.perf_counter() - t0
        self.move_count += 1
        return action


def _chunk(job):
    opp_name, hands, seat, seed, rules, es, ed, ek, pf = job
    sim.set_rules(rules)
    import meta
    rng = random.Random(seed)
    pol_a = EGPolicy(seed=seed, exact_stock=es, exact_depth=ed,
                     exact_k=ek, pf=pf)
    pol_b = meta.population()[opp_name]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    ex = pol_a.expert
    return (points, pol_a.move_time, pol_a.move_count,
            ex.eg_decisions, ex.eg_exact)


def egmatch(opp_name, hands, workers, seed, rules="full", exact_stock=6,
            exact_depth=2, exact_k=3, pf=False):
    per = max(1, hands // (2 * workers))
    jobs = [(opp_name, per, seat, seed + 31 * w + seat, rules,
             exact_stock, exact_depth, exact_k, pf)
            for w in range(workers) for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_chunk, jobs)
    points = np.concatenate([r[0] for r in results])
    move_t = sum(r[1] for r in results)
    move_n = sum(r[2] for r in results)
    eg_d = sum(r[3] for r in results)
    eg_x = sum(r[4] for r in results)
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    print(f"exact-eg[s{exact_stock} d{exact_depth} k{exact_k}"
          f"{',pf' if pf else ''}] vs {opp_name} [{rules}]: "
          f"{mean:+.2f} +/- {ci:.2f} over {len(points)} hands "
          f"({move_t / max(move_n, 1):.3f} s/searched-move, "
          f"eg {eg_d / max(len(points), 1):.2f}/hand, "
          f"exact {eg_x / max(eg_d, 1):.0%} of eg, "
          f"{time.time() - t0:.0f}s wall)", flush=True)
    return mean, ci, len(points)


def parity():
    """exact_stock=0 must be bit-identical to AdaptiveExpert."""
    sim.set_rules("full")
    from q_expert import _decision_stream
    cfgs = ((2.5, (10, 1.0)),)
    base = _decision_stream(lambda b, m: AdaptiveExpert(
        stages=(16, 48, 192), deep=True, deep_from=2, kbranch=3,
        truncate=16, z=1.5, opp_model=m, seed=11, sample_boost=b), cfgs)
    mine = _decision_stream(lambda b, m: ExactEndgameExpert(
        exact_stock=0, stages=(16, 48, 192), deep=True, deep_from=2,
        kbranch=3, truncate=16, z=1.5, opp_model=m, seed=11,
        sample_boost=b), cfgs)
    same = base == mine
    print(f"endgame parity: {len(base)} decisions, "
          f"{'IDENTICAL -- PASS' if same else 'DIFFER -- FAIL'}")
    return same


def main():
    ap = argparse.ArgumentParser("v6 exact endgame gates")
    ap.add_argument("--vs", default="v5")
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=63100)
    ap.add_argument("--exact_stock", type=int, default=6)
    ap.add_argument("--exact_depth", type=int, default=2)
    ap.add_argument("--exact_k", type=int, default=3)
    ap.add_argument("--pf", action="store_true",
                    help="worlds from the particle belief (v6 base)")
    ap.add_argument("--parity", action="store_true")
    args = ap.parse_args()
    if args.parity:
        parity()
        return
    egmatch(args.vs, args.hands, args.workers, args.seed, args.rules,
            args.exact_stock, args.exact_depth, args.exact_k, args.pf)


if __name__ == "__main__":
    main()
