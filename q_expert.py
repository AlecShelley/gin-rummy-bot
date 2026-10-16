"""
Gates 2+3: the v5 Q-net inside the v4 searcher.

Gate 2 (`QRolloutExpert`): the net replaces FastTeacher on the HERO seat of
every rollout, batched -- sim.playout calls policies one state at a time, so
this class re-implements the world/candidate loop as a LOCKSTEP advance:
all K determinized lanes step together, and each lockstep round runs ONE net
forward over the lanes that reached a hero multi-action decision. The
opponent seat is unchanged from v4 (MixedKnock around FastTeacher --
continuation realism beat strength, the 2026-08-23 lesson). Hero rollout
decisions argmax Q over the teacher-pruned candidate set (label support),
with a per-lane cloned Belief feeding build_obs.

Parity discipline: lane construction consumes self.rng in EXACTLY v4's order
(sample -> weight -> apply_world per accepted world), and MixedKnock at
p=1.0 never consumes its RNG, so `--mock-parity` (hero rollout decisions
routed to FastTeacher through the same lockstep machinery) must reproduce
reference decisions BIT-IDENTICALLY (refactor_check.py pattern). The
statistics phase is copied verbatim from expert.analyze -- paired diffs,
z-SE gate, never argmax raw means (lesson #3).

ONE deliberate semantic difference from expert.RolloutExpert (found while
chasing parity, 2026-08-24): v4 computes each world's weight from
`opp.known_cards` READ OFF THE SCRATCH, which for world k+1 is still
polluted by world k's rollout (rollout pickups extend known_cards; restore
happens after the weight check) -- so v4 sporadically over-rejects
knockable-opponent worlds based on stale rollout state. This class uses the
pristine decision-time known_cards (the information-set-correct choice).
Consequence: with an opp_model active, decisions are NOT bit-equal to v4;
`--mock-parity` therefore checks (a) exact identity vs true RolloutExpert
on configs without an opp_model, and (b) exact identity vs a
quirk-corrected v4 reference on the opp_model config.

Gate 3 (`QPruneExpert`): v4's search unchanged, but discard candidates are
net-ranked instead of deadwood-ranked (one forward per real decision).

    .venv/Scripts/python.exe q_expert.py --mock-parity
    .venv/Scripts/python.exe q_expert.py --time --sweepK 16,24,32,48,64
    .venv/Scripts/python.exe q_expert.py --vs v4 --worlds 32 --hands 1500 --seed 93000
    .venv/Scripts/python.exe q_expert.py --mode prune --vs v4 --hands 1500 --seed 95000
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
from belief import Belief, build_obs
from expert import RolloutExpert
from legacy.qnet import QNet
from sim import (DECK, FastTeacher, apply_world, cid, fast_legal_actions,
                 ids_of, leaf_value, make_scratch, min_deadwood_ids, playout,
                 restore, snapshot, step_fast)

DEFAULT_QNET_PATH = "results/v5_q/qnet.pt"


def clone_belief(b):
    """Cheap copy for rollout lanes: the lane's move_sheet is a deepcopy of
    the real one, so _processed/_pile stay consistent and update() consumes
    only the lane's new rollout moves."""
    c = Belief.__new__(Belief)
    c.hero_id = b.hero_id
    c.alpha, c.beta, c.gamma = b.alpha, b.beta, b.gamma
    c.w = b.w.copy()
    c.opp_declined = b.opp_declined.copy()
    c.opp_discarded = b.opp_discarded.copy()
    c.hero_discarded = b.hero_discarded.copy()
    c.opp_pickups = b.opp_pickups
    c.opp_declines = b.opp_declines
    c._processed = b._processed
    c._pile = list(b._pile)
    return c


class QRolloutExpert(RolloutExpert):
    """RolloutExpert whose hero rollout seat is the batched Q-net."""

    def __init__(self, net_path=DEFAULT_QNET_PATH, device="cpu", mock=False,
                 net_margin=0.0, hero_mode=None, **kw):
        super().__init__(**kw)
        # hero_mode: who plays the rollout HERO seat. "net" (batched Q),
        # "teacher" (FastTeacher -- the mock/ablation, == v4 semantics),
        # "bt" (BeliefTeacher: hand-reading discard safety, per-lane belief).
        if hero_mode is None:
            hero_mode = "teacher" if mock else "net"
        self.hero_mode = hero_mode
        self.mock = hero_mode == "teacher"
        self.net = QNet(net_path, device=device) if hero_mode == "net" \
            else None
        if hero_mode == "bt":
            from belief_teacher import BeliefTeacher
            self._bt = BeliefTeacher()
        else:
            self._bt = None
        # net_margin: rollout hero plays the teacher's action unless the net
        # clears it by this many points (gate-1 lesson: greedy argmax over a
        # noisy Q selects noise; margin keeps continuations teacher-parity
        # while retaining the net's confident deviations). 0 = greedy.
        self.net_margin = net_margin

    # -- the modified decision (structure mirrors expert.analyze) -----------

    def analyze(self, game, belief):
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
            boost = min(3.0, 0.5 * (belief.opp_pickups + belief.opp_declines))

        teacher_action = self.teacher.act(game, legal)
        t_idx = next((k for k, a in enumerate(candidates)
                      if a.action_id == teacher_action.action_id), None)
        if t_idx is None:
            candidates.insert(0, teacher_action)
            t_idx = 0

        opp_rollout = self._opp_rollout if self._opp_rollout is not None \
            else self.rollout[opp_id]
        opp = game.round.players[opp_id]
        unseen = ids_of(game.round.dealer.stock_pile) + \
            [cid(c) for c in opp.hand if c not in opp.known_cards]
        num_hidden = len(opp.hand) - len(opp.known_cards)
        opp_discard_count = int(belief.opp_discarded.sum())

        # Lane build -- self.rng consumed in v4's exact order (parity).
        lanes, snaps = [], []
        weights = np.zeros(self.worlds)
        attempts = 0
        max_attempts = 3 * self.worlds
        while len(lanes) < self.worlds and attempts < max_attempts:
            attempts += 1
            if num_hidden > 0:
                if self.uniform_worlds:
                    pick = self.rng.choice(len(unseen), size=num_hidden,
                                           replace=False)
                    free_ids = [unseen[i] for i in pick]
                else:
                    free_ids = belief.sample_free_cards(unseen, num_hidden,
                                                        self.rng, boost)
            else:
                free_ids = []
            opp_hand_ids = ids_of(opp.known_cards) + list(free_ids)
            weight = self._world_weight(opp_hand_ids, opp_discard_count)
            if weight <= 0.0:
                continue
            lane = make_scratch(game)
            apply_world(lane, opp_id, [DECK[i] for i in free_ids], self.rng)
            weights[len(lanes)] = weight
            lanes.append(lane)
            snaps.append(snapshot(lane))
        worlds_done = len(lanes)
        if worlds_done == 0 or weights[:worlds_done].sum() <= 0.0:
            return {"chosen": teacher_action, "reason": "no-worlds",
                    "candidates": candidates}

        sign = 1.0 if hero_id == 0 else -1.0
        payoffs = np.zeros((worlds_done, len(candidates)))
        for k, action in enumerate(candidates):
            for i in range(worlds_done):
                restore(lanes[i], snaps[i])
                step_fast(lanes[i], action)
            payoffs[:, k] = sign * self._lockstep(lanes, hero_id, opp_id,
                                                  opp_rollout, belief)

        # Statistics: verbatim expert.analyze (lesson #3 -- z-SE gate).
        w = weights[:worlds_done]
        diffs = payoffs - payoffs[:, [t_idx]]
        w_sum = w.sum()
        means = (w[:, None] * diffs).sum(axis=0) / w_sum
        centered = diffs - means
        var = (w[:, None] * centered ** 2).sum(axis=0) / w_sum
        n_eff = w_sum ** 2 / (w ** 2).sum()
        se = np.sqrt(var / max(n_eff - 1.0, 1.0))
        gain = means - self.z * se
        best = int(np.argmax(gain))
        deviate = gain[best] > 0.0
        payoff_means = (w[:, None] * payoffs).sum(axis=0) / w_sum
        return {"chosen": candidates[best] if deviate else teacher_action,
                "reason": "deviate" if deviate else "prior",
                "candidates": candidates, "t_idx": t_idx,
                "payoff_means": payoff_means, "diff_means": means,
                "se": se, "gain": gain, "worlds": worlds_done,
                "n_eff": float(n_eff)}

    # -- lockstep world advance ---------------------------------------------

    def _lockstep(self, lanes, hero_id, opp_id, opp_rollout, decision_belief):
        """Advance all lanes to termination/truncation; hero multi-action
        decisions are batched through the net (or FastTeacher when mock).
        Semantics match sim.playout(truncate=T): every step consumes budget,
        the frontier scores leaf_value, terminals score _payoff."""
        n = len(lanes)
        out = np.zeros(n)
        budget = [self.truncate] * n
        # Beliefs are cloned LAZILY, on a lane's first use -- most rollout
        # lanes never reach a decision that consults the belief (measured
        # 1.68x -> ~parity wall-clock for bt mode).
        bels = None if self.mock else [None] * n
        bt = self._bt
        hero_teacher = self.rollout[hero_id]
        active = list(range(n))
        while active:
            queued, still = [], []
            for i in active:
                lane = lanes[i]
                rnd = lane.round
                while True:
                    if rnd.going_out_action is not None or rnd.is_over:
                        out[i] = sim._payoff(lane)
                        break
                    if budget[i] == 0:
                        out[i] = leaf_value(lane)
                        break
                    legal = fast_legal_actions(lane)
                    if len(legal) == 1:
                        step_fast(lane, legal[0])
                        budget[i] -= 1
                        continue
                    if rnd.current_player_id == opp_id:
                        step_fast(lane, opp_rollout.act(lane, legal))
                        budget[i] -= 1
                        continue
                    if self.mock:
                        step_fast(lane, hero_teacher.act(lane, legal))
                        budget[i] -= 1
                        continue
                    if bt is not None:   # belief-aware teacher, per lane
                        if bels[i] is None:
                            bels[i] = clone_belief(decision_belief)
                        bt.belief = bels[i]
                        step_fast(lane, bt.act(lane, legal))
                        budget[i] -= 1
                        continue
                    queued.append((i, legal))
                    still.append(i)
                    break
            if queued:
                obs_list, cand_lists = [], []
                for i, legal in queued:
                    lane = lanes[i]
                    cands = self._candidates(lane, legal)
                    t_act = self.teacher.act(lane, legal)
                    if all(a.action_id != t_act.action_id for a in cands):
                        cands.insert(0, t_act)
                    cand_lists.append((cands, t_act))
                    if bels[i] is None:
                        bels[i] = clone_belief(decision_belief)
                    obs_list.append(build_obs(lane, hero_id, bels[i],
                                              [e.action_id for e in legal]))
                q = self.net.q_values(np.stack(obs_list))
                for (i, legal), (cands, t_act), qrow in zip(
                        queued, cand_lists, q):
                    ids = [a.action_id for a in cands]
                    action = cands[int(np.argmax(qrow[ids]))]
                    if self.net_margin > 0.0 and \
                            qrow[action.action_id] - qrow[t_act.action_id] \
                            <= self.net_margin:
                        action = t_act
                    step_fast(lanes[i], action)
                    budget[i] -= 1
            active = still
        return out


class QPruneExpert(RolloutExpert):
    """Gate 3: v4's search with net-ranked discard candidates."""

    def __init__(self, net_path=DEFAULT_QNET_PATH, device="cpu", **kw):
        super().__init__(**kw)
        self.net = QNet(net_path, device=device)
        self._cur_belief = None

    def analyze(self, game, belief):
        self._cur_belief = belief
        return super().analyze(game, belief)

    def _candidates(self, game, legal):
        if len(legal) == 1:
            return legal
        discards = [e for e in legal if isinstance(e, DiscardAction)]
        knocks = [e for e in legal if isinstance(e, KnockAction)]
        if not discards and not knocks:
            return legal                 # draw decision: unchanged
        rnd = game.round
        hand_ids = ids_of(rnd.players[rnd.current_player_id].hand)
        chosen = []
        if discards:
            obs = build_obs(game, rnd.current_player_id, self._cur_belief,
                            [e.action_id for e in legal])
            q = self.net.q_values(obs)
            ranked = sorted(discards, key=lambda e: -q[e.action_id])
            chosen.extend(ranked[:self.top_discards])
        if knocks:                       # deadwood-best knock, as in v4
            best, best_dw = None, 999
            for event in knocks:
                drop = cid(event.card)
                dw = min_deadwood_ids([i for i in hand_ids if i != drop])
                if dw < best_dw:
                    best, best_dw = event, dw
            chosen.append(best)
        return chosen


# ---------------------------------------------------------------------------
# Mock parity: QRolloutExpert(mock) must be decision-identical to the
# reference -- true v4 where that is exact (no opp_model), a quirk-corrected
# v4 where v4's stale-knowns weight quirk forks the stream (see docstring).
# ---------------------------------------------------------------------------

class _RefExpert(RolloutExpert):
    """expert.analyze verbatim except the world weight uses the pristine
    decision-time known_cards instead of the scratch's post-rollout ones."""

    def analyze(self, game, belief):
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
        known_ids = ids_of(opp.known_cards)      # pristine -- the one fix
        opp_discard_count = int(belief.opp_discarded.sum())
        payoffs = np.zeros((self.worlds, len(candidates)))
        weights = np.zeros(self.worlds)
        sign = 1.0 if hero_id == 0 else -1.0
        attempts = 0
        worlds_done = 0
        max_attempts = 3 * self.worlds
        while worlds_done < self.worlds and attempts < max_attempts:
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
                payoffs[worlds_done, k] = sign * playout(
                    scratch, rollout, truncate=self.truncate)
            weights[worlds_done] = weight
            worlds_done += 1
        if worlds_done == 0 or weights[:worlds_done].sum() <= 0.0:
            return {"chosen": teacher_action, "reason": "no-worlds",
                    "candidates": candidates}
        w = weights[:worlds_done]
        diffs = payoffs[:worlds_done] - payoffs[:worlds_done, [t_idx]]
        w_sum = w.sum()
        means = (w[:, None] * diffs).sum(axis=0) / w_sum
        centered = diffs - means
        var = (w[:, None] * centered ** 2).sum(axis=0) / w_sum
        n_eff = w_sum ** 2 / (w ** 2).sum()
        se = np.sqrt(var / max(n_eff - 1.0, 1.0))
        gain = means - self.z * se
        best = int(np.argmax(gain))
        deviate = gain[best] > 0.0
        payoff_means = (w[:, None] * payoffs[:worlds_done]).sum(axis=0) / w_sum
        return {"chosen": candidates[best] if deviate else teacher_action,
                "reason": "deviate" if deviate else "prior",
                "candidates": candidates, "t_idx": t_idx,
                "payoff_means": payoff_means, "diff_means": means,
                "se": se, "gain": gain, "worlds": worlds_done,
                "n_eff": float(n_eff)}


def _decision_stream(make_expert, configs):
    out = []
    for boost, opp_model in configs:
        ex = make_expert(boost, opp_model)
        teach = FastTeacher()
        rng = random.Random(2024)
        for _ in range(6):
            game = sim.new_game(rng.randrange(1 << 30))
            rnd = game.round
            belief = Belief(0)
            while rnd.going_out_action is None and not rnd.is_over:
                legal = fast_legal_actions(game)
                if len(legal) == 1:
                    step_fast(game, legal[0])
                    continue
                if rnd.current_player_id == 0:
                    a = ex.decide(game, belief)
                    out.append(a.action_id)
                else:
                    a = teach.act(game, legal)
                step_fast(game, a)
    return out


def mock_parity():
    sim.set_rules("full")
    ok = True
    for label, ref_cls, configs in (
            ("vs true v4 (no opp_model)", RolloutExpert,
             ((0.0, None), (2.5, None))),
            ("vs corrected-v4 ref (opp_model)", _RefExpert,
             ((0.0, (10, 1.0)), (2.5, (10, 1.0))))):
        base = _decision_stream(lambda b, m: ref_cls(
            worlds=24, truncate=16, z=1.5, opp_model=m, seed=11,
            sample_boost=b), configs)
        mock = _decision_stream(lambda b, m: QRolloutExpert(
            mock=True, worlds=24, truncate=16, z=1.5, opp_model=m, seed=11,
            sample_boost=b), configs)
        same = base == mock
        ok = ok and same
        print(f"mock parity {label}: {len(base)} decisions, "
              f"{'IDENTICAL -- PASS' if same else 'DIFFER -- FAIL'}")
        if not same:
            diffs = [(k, a, b) for k, (a, b) in enumerate(zip(base, mock))
                     if a != b]
            print(f"first diffs: {diffs[:5]}  (len {len(base)} vs "
                  f"{len(mock)})")
    return ok


# ---------------------------------------------------------------------------
# Wall-clock calibration: seconds per searched decision, v4 vs net-K sweep.
# ---------------------------------------------------------------------------

def _time_expert(expert, hands, seed):
    teach = FastTeacher()
    rng = random.Random(seed)
    times = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        rnd = game.round
        belief = Belief(0)
        while rnd.going_out_action is None and not rnd.is_over:
            legal = fast_legal_actions(game)
            if len(legal) == 1:
                step_fast(game, legal[0])
                continue
            if rnd.current_player_id == 0:
                t0 = time.perf_counter()
                a = expert.decide(game, belief)
                times.append(time.perf_counter() - t0)
            else:
                a = teach.act(game, legal)
            step_fast(game, a)
    return float(np.mean(times)), len(times)


def calibrate(net_path, device, hands, sweep, truncate, seed=90500):
    sim.set_rules("full")
    v4 = RolloutExpert(worlds=48, truncate=16, z=1.5, opp_model=(10, 1.0),
                       sample_boost=2.5, seed=0)
    t_v4, n = _time_expert(v4, hands, seed)
    print(f"v4 (K=48,T=16): {t_v4:.3f} s/searched-move ({n} moves)",
          flush=True)
    budget = 1.05 * t_v4
    best_k = None
    for k in sweep:
        ex = QRolloutExpert(net_path=net_path, device=device, worlds=k,
                            truncate=truncate, z=1.5, opp_model=(10, 1.0),
                            sample_boost=2.5, seed=0)
        t_k, n = _time_expert(ex, hands, seed)
        ok = t_k <= budget
        print(f"net K={k},T={truncate}: {t_k:.3f} s/searched-move ({n})"
              f"{'  <= budget' if ok else '  OVER budget'}", flush=True)
        if ok:
            best_k = k
    print(f"parity config: K={best_k} at T={truncate} "
          f"(budget {budget:.3f} s/move)", flush=True)
    return best_k


# ---------------------------------------------------------------------------
# Gate harness (value_expert.vmatch pattern).
# ---------------------------------------------------------------------------

class _QExpertPolicy:
    """Engine-facing wrapper (meta._ExpertPolicy pattern)."""

    def __init__(self, cfg, seed=0):
        kw = dict(worlds=cfg["worlds"], truncate=cfg["truncate"], z=cfg["z"],
                  opp_model=(10, 1.0), sample_boost=2.5, seed=seed)
        if cfg["mode"] == "prune":
            self.expert = QPruneExpert(cfg["net_path"], cfg["device"], **kw)
        elif cfg["mode"] == "mock":      # teacher rollouts, K ablation
            self.expert = QRolloutExpert(mock=True, **kw)
        elif cfg["mode"] == "bt":        # belief-aware teacher rollouts
            from belief_teacher import BTSerialExpert
            self.expert = BTSerialExpert(**kw)
        else:
            self.expert = QRolloutExpert(
                cfg["net_path"], cfg["device"],
                net_margin=cfg.get("net_margin", 0.0), **kw)
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


def _qematch_chunk(job):
    cfg, opp_name, hands, seat, seed, rules = job
    sim.set_rules(rules)
    import meta
    rng = random.Random(seed)
    pol_a = _QExpertPolicy(cfg, seed=seed)
    pol_b = meta.population()[opp_name]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    return points, pol_a.move_time, pol_a.move_count


def qematch(cfg, opp_name, hands, workers, seed, rules="full"):
    per = max(1, hands // (2 * workers))
    jobs = [(cfg, opp_name, per, seat, seed + 31 * w + seat, rules)
            for w in range(workers) for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_qematch_chunk, jobs)
    points = np.concatenate([r[0] for r in results])
    move_t = sum(r[1] for r in results)
    move_n = sum(r[2] for r in results)
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    print(f"v5[{cfg['mode']}](K={cfg['worlds']},T={cfg['truncate']},"
          f"z={cfg['z']}) vs {opp_name} [{rules}]: {mean:+.2f} +/- {ci:.2f} "
          f"over {len(points)} hands  "
          f"(measured {move_t / max(move_n, 1):.3f} s/searched-move, "
          f"{time.time() - t0:.0f}s wall)", flush=True)
    return mean, ci, len(points)


def main():
    ap = argparse.ArgumentParser("v5 in-search gates")
    ap.add_argument("--vs", default="v4")
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--mode", default="net",
                    choices=("net", "prune", "mock", "bt"))
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--truncate", type=int, default=16)
    ap.add_argument("--z", type=float, default=1.5)
    ap.add_argument("--net_path", default=DEFAULT_QNET_PATH)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--net_margin", type=float, default=0.0)
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=93000)
    ap.add_argument("--mock-parity", dest="mock_parity", action="store_true")
    ap.add_argument("--time", dest="timing", action="store_true")
    ap.add_argument("--sweepK", default="16,24,32,48,64")
    args = ap.parse_args()
    if args.mock_parity:
        mock_parity()
        return
    if args.timing:
        sweep = [int(k) for k in args.sweepK.split(",")]
        calibrate(args.net_path, args.device, args.hands if args.hands <= 60
                  else 30, sweep, args.truncate)
        return
    cfg = {"mode": args.mode, "worlds": args.worlds,
           "truncate": args.truncate, "z": args.z,
           "net_path": args.net_path, "device": args.device,
           "net_margin": args.net_margin}
    qematch(cfg, args.vs, args.hands, args.workers, args.seed, args.rules)


if __name__ == "__main__":
    main()
