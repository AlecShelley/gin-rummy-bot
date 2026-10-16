"""
The empirical meta-game: population, payoff matrix, Nash, double oracle.

This is the poker-blueprint layer of the v3 pipeline. Individual policies
(heuristics, mixed-knock variants, network clones) are the *pure strategies*;
their pairwise seat-swapped match results form an empirical zero-sum matrix
game; the LP solves for its Nash mixture; and the double-oracle loop hunts for
a best response to that mixture, adds it, and re-solves -- stopping when no
found response beats the mixture by more than a threshold. The 0/50/100%
knock-mixing the user asked for lives in the MixedKnock members and in the
Nash weights themselves.

All matches run engine-driven (sim kernel) in parallel workers; network
policies play on CPU with one thread each.

    python meta.py --matrix --hands 1000 --workers 12
    python meta.py --nash
    python meta.py --exploit <member> --hands 800
"""

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import OrderedDict

import numpy as np

import sim
from belief import base_obs
from sim import FastTeacher, MixedKnock, fast_legal_actions, playout

MATRIX_DIR = "results/v3_meta"


# ---------------------------------------------------------------------------
# Population registry: name -> builder of a fast (engine-facing) policy.
# Builders run inside workers, so heavyweight loads stay per-process.
# ---------------------------------------------------------------------------

class _RlcardAdapter:
    """Wrap an rlcard env-agent (reads 5x52 obs dicts) as a fast policy."""

    def __init__(self, agent):
        self.agent = agent

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        ids = [e.action_id for e in legal]
        state = {"obs": base_obs(game),
                 "legal_actions": OrderedDict((i, None) for i in ids),
                 "raw_legal_actions": ids}
        action_id = int(self.agent.step(state))
        return legal[ids.index(action_id)]


class _V3Adapter:
    """A v3 policy as a fast policy, with its own Belief per hand."""

    def __init__(self, path):
        from exit_gen import _V3Clone
        from belief import Belief, build_obs
        self.clone = _V3Clone(path)
        self.Belief = Belief
        self.build_obs = build_obs
        self._round_key = None
        self._belief = None

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._belief = self.Belief(rnd.current_player_id)
        obs = self.build_obs(game, rnd.current_player_id, self._belief,
                             [e.action_id for e in legal])
        return self.clone.act_with_obs(obs, legal)


def _rule_agent():
    from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
    return _RlcardAdapter(GinRummyNoviceRuleAgent())


def _smart_agent():
    from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
    from smart_draw import SmartDraw
    return _RlcardAdapter(SmartDraw(GinRummyNoviceRuleAgent()))


def _v_bot(path):
    def build():
        import torch
        torch.set_num_threads(1)
        from gin_rummy_bot import GinRummyBot
        return _RlcardAdapter(GinRummyBot(path, device=torch.device("cpu"),
                                          greedy=True))
    return build


def _mixed(threshold, p, knock_at=0):
    def build():
        return MixedKnock(FastTeacher(knock_at=knock_at), threshold, p,
                          rng=random.Random(0))
    return build


class _RandomPolicy:
    """Uniform over legal actions -- the dead-hand tripwire."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, game, legal):
        return legal[self.rng.randrange(len(legal))]


class _ExpertPolicy:
    """The rollout expert as an engine-facing policy: the hand-reading
    adversary for exploitability probes. `opp_model` is what it assumes about
    its TARGET's knocking -- the range-reading knob."""

    def __init__(self, opp_model, seed=0, sample_boost=0.0, top_discards=4):
        from expert import RolloutExpert
        from belief import Belief
        self.expert = RolloutExpert(worlds=48, truncate=16, z=1.5,
                                    opp_model=opp_model, seed=seed,
                                    sample_boost=sample_boost,
                                    top_discards=top_discards)
        self.Belief = Belief
        self._round_key = None
        self._belief = None

    def act(self, game, legal):
        if len(legal) == 1:
            return legal[0]
        rnd = game.round
        if not hasattr(rnd, "_v3_token"):
            rnd._v3_token = object()
        if rnd._v3_token is not self._round_key:
            self._round_key = rnd._v3_token
            self._belief = self.Belief(rnd.current_player_id)
        return self.expert.decide(game, self._belief)


# The knock-family grid: the same 0/50/100 discretization everywhere.
BR_GRID = [(t, p) for t in (0, 2, 4, 6, 8, 10) for p in (0.0, 0.5, 1.0)]


def _br_name(threshold, p):
    return f"br_t{threshold}_p{int(p * 100)}"


# Members the payoff matrix is built over by default. BR-grid and adversary
# members exist in the registry but join the matrix only when named.
CORE_NAMES = ["rule", "smart", "teacher", "knock6_p100", "knock6_p50",
              "knock10_p100", "knock10_p50", "v1", "v2"]


def population():
    """name -> builder for every known member. Static and complete, because
    Windows spawn workers re-import this module: runtime registry mutations
    in the parent would never reach them."""
    pop = {
        "rule": _rule_agent,
        "smart": _smart_agent,
        "teacher": FastTeacher,                       # patient, gin-only
        "knock6_p100": _mixed(6, 1.0),
        "knock6_p50": _mixed(6, 0.5),
        "knock10_p100": _mixed(10, 1.0),
        "knock10_p50": _mixed(10, 0.5),
        "v1": _v_bot("results/final/gin_rummy_bot.policy.pt"),
        "v2": _v_bot("results/v2_final/gin_rummy_bot.policy.pt"),
    }
    for threshold, p in BR_GRID:
        pop[_br_name(threshold, p)] = _mixed(threshold, p)
    for threshold in (0, 6, 10):
        for p in (0.0, 0.5, 1.0):
            pop[f"adv_t{threshold}_p{int(p * 100)}"] = (
                lambda t=threshold, q=p: _ExpertPolicy(opp_model=(t, q)))
    v3_best = "results/v3/best.policy.pt"
    if os.path.exists(v3_best):
        pop["v3"] = lambda: _V3Adapter(v3_best)
    # The search bot itself -- the v3 deliverable. No opponent model by
    # default: no assumptions about who it faces.
    pop["v3search"] = lambda: _ExpertPolicy(opp_model=None)
    # Correlated-world variant (roadmap #2): boost=2.5 matches the true mean
    # deadwood of hidden opponent hands (corr_sample.py, 2026-08-23).
    pop["v3boost"] = lambda: _ExpertPolicy(opp_model=None, sample_boost=2.5)
    # Wider candidate set (roadmap #3): safe-but-slightly-worse discards can
    # only be found if they enter the search at all.
    pop["v3wide"] = lambda: _ExpertPolicy(opp_model=None, top_discards=8)
    # Correct knock model + realistic worlds together. PROMOTED to v4 on
    # 2026-08-23: beats v3search head-to-head (+1.59 +/- 0.87, 3984 fresh
    # hands; second independent run +2.49 +/- 1.41), extracts +6.59 +/- 0.76
    # from smart (teacher-level), costs nothing vs non-knockers.
    pop["adv10_boost"] = lambda: _ExpertPolicy(opp_model=(10, 1.0),
                                               sample_boost=2.5)
    pop["v4"] = pop["adv10_boost"]

    # v5 -- PROMOTED 2026-08-27 (all gates green; unexploited by the
    # 21-adversary full-rules probe, results/probe_v5_full_20260826.log;
    # teacher 4000-hand decider -0.12 +/- 0.96 tie).
    # Architecture: adaptive staged world budget (16/48/192,
    # escalate near-significant, deviate only at the highest stage reached)
    # + 2-ply close-call rollouts, on the quirk-corrected serial loop
    # (adaptive.py). Beats v4 +1.20 +/- 0.93 (3984 fresh hands) and the
    # compute-matched plain-K v4-K80 +1.61 +/- 1.55, at ~1.7x v4
    # wall-clock (~0.16 s/searched-move). results/v5_q/v5_gates.log.
    def _v5():
        from adaptive import _AdaptivePolicy
        return _AdaptivePolicy({"stages": [16, 48, 192], "deep": True,
                                "deep_from": 2, "kbranch": 3,
                                "endgame_stock": 0, "mid_alpha": 0.5,
                                "truncate": 16, "z": 1.5})
    pop["v5"] = _v5

    # v6 -- particle-filter belief worlds inside the unchanged v5 search
    # (pf_expert.PFPolicy; filter in particle.py: sequential PF with
    # softened-teacher likelihoods + resample-move rejuvenation,
    # calibrated results/v6_particle_calibration.txt). Gates
    # (results/v6_gates.log): offline Brier +13.7%/+13.8% held-out
    # beats belief; worlds-gate dominates (late knock-Brier .191 vs
    # .459); vs v5 selection +2.31 +/- 1.59 (seed 62100) and fresh
    # confirm +1.92 +/- 0.97 (seed 64300, 3984 hands); compute-matched
    # control (v5 stages 16/48/256 deep) only +0.50 +/- 0.94 at the
    # same cost (0.378 vs 0.441 s/searched-move) -- the architecture
    # wins. Oracle bound context: +15.91 +/- 1.44 (perfect hand info).
    # v6 is FROZEN at its gated config (2026-08-31 promotion of v7):
    # the pickup likelihood it was gated with is the gain logistic and
    # rejuvenation used the plain 6-event window. DEFAULT_PF has since
    # moved on -- pin the old knobs explicitly.
    def _v6():
        from pf_expert import PFPolicy
        return PFPolicy(seed=0, pf_cfg=dict(
            pickup_mode="logistic", tau_p=1.0, mh_all_takes=False,
            eps=0.05, tau_po=None))      # + pre-v10 eps/tau_po pins
    pop["v6"] = _v6

    # v7 -- THE CHAMPION (promoted 2026-08-31, results/v8_gates.log):
    # v6's search + filter with the redesigned opponent-action
    # likelihood: binary-rule pickup model (take iff gain >= 1, meld
    # tier on top, tau_p=0.1 = rule-mismatch rate -- fixes the
    # hidden-5S inversion) and mh_all_takes rejuvenation (all
    # pickup/decline evidence always re-scored). Match: TIE vs v6
    # (selection +0.80 +/- 1.62, confirm -0.30 +/- 0.99 @ 3984);
    # promoted on the offline posterior gains (held-out Brier,
    # worlds overlap .560->.580 / knockB .239->.224, read
    # regressions) + clean probe (all four adversaries lose).
    # v7 is FROZEN at its gated config since the v10 promotion
    # (2026-09-07, results/v10_gates.log): DEFAULT_PF moved to
    # eps=0.02 + tau_po=0.0 -- pin the pre-v10 knobs explicitly.
    def _v7():
        from pf_expert import PFPolicy
        return PFPolicy(seed=0, pf_cfg=dict(eps=0.05, tau_po=None))
    pop["v7"] = _v7

    # Belief-poisoning adversaries (v7 probe, 2026-08-29, poison.py):
    # teacher fundamentals, discard tie-break reversed into a false
    # signal ("salting"). ad_*_k10 = the same with a knock-at-10 layer.
    def _adv(mode, knock=None):
        def build():
            from poison import Advertiser
            adv = Advertiser(mode=mode)
            if knock is not None:
                return MixedKnock(adv, knock, 1.0, rng=random.Random(0))
            return adv
        return build
    pop["ad_self"] = _adv("self")
    pop["ad_jam"] = _adv("jam")
    pop["ad_self_k10"] = _adv("self", knock=10)
    pop["random"] = _RandomPolicy
    return pop


# ---------------------------------------------------------------------------
# Engine-driven seat-swapped matches.
# ---------------------------------------------------------------------------

def _match_chunk(job):
    """Worker: `hands` games of a vs b, a in `seat`; returns a's points."""
    name_a, name_b, hands, seat, seed, rules = job
    sim.set_rules(rules)
    rng = random.Random(seed)
    builders = population()
    pol_a, pol_b = builders[name_a](), builders[name_b]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    return points


def match(name_a, name_b, hands, workers, pool, seed=1000,
          rules="classic"):
    per = max(1, hands // (2 * workers))
    jobs = [(name_a, name_b, per, seat, seed + 31 * w + seat, rules)
            for w in range(workers) for seat in (0, 1)]
    points = np.concatenate(pool.map(_match_chunk, jobs))
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    return mean, ci, len(points)


def build_matrix(hands=1000, workers=12, seed=1000, names=None,
                 rules="classic"):
    os.makedirs(MATRIX_DIR, exist_ok=True)
    names = names or [n for n in CORE_NAMES + ["v3", "v3search"]
                      if n in population()]
    n = len(names)
    means = np.zeros((n, n))
    cis = np.zeros((n, n))
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for i, j in itertools.combinations(range(n), 2):
            m, ci, _ = match(names[i], names[j], hands, workers, pool,
                             seed + 7919 * (i * n + j), rules)
            means[i, j], means[j, i] = m, -m
            cis[i, j] = cis[j, i] = ci
            print(f"  {names[i]:>14} vs {names[j]:<14} {m:+7.2f} +/- {ci:.2f}",
                  flush=True)
    payload = {"names": names, "means": means.tolist(), "cis": cis.tolist(),
               "hands": hands, "seed": seed, "rules": rules}
    suffix = "" if rules == "classic" else f"_{rules}"
    with open(os.path.join(MATRIX_DIR, f"matrix{suffix}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"matrix[{rules}] over {n} members in {time.time() - t0:.0f}s "
          f"-> {MATRIX_DIR}/matrix{suffix}.json", flush=True)
    return payload


# ---------------------------------------------------------------------------
# Nash of the zero-sum empirical game.
# ---------------------------------------------------------------------------

def solve_nash(means, names, iters=400_000):
    """Nash of a zero-sum matrix by regret matching+, cross-checked with
    fictitious play. No LP dependency: scipy's HiGHS DLL is blocked by this
    machine's Application Control policy, and for matrices this small an
    iterative solve to ~1e-3 exploitability is just as good.

    Returns the symmetric mixture (row player of the antisymmetric matrix)
    plus the exploitability of that mixture within the matrix -- how much the
    best pure counter-strategy still gains against it.
    """
    a = np.array(means, dtype=float)
    n = len(a)

    # Regret matching+ self-play, linear averaging.
    regrets = np.zeros(n)
    strategy_sum = np.zeros(n)
    x = np.ones(n) / n
    for t in range(iters):
        payoff = a @ x                   # payoff of each pure row vs mixture
        value = x @ payoff
        regrets = np.maximum(regrets + payoff - value, 0.0)
        total = regrets.sum()
        x = regrets / total if total > 1e-12 else np.ones(n) / n
        strategy_sum += x
    x = strategy_sum / strategy_sum.sum()

    # Fictitious play cross-check.
    counts = np.ones(n)
    for _ in range(iters // 2):
        counts[int(np.argmax(a @ (counts / counts.sum())))] += 1.0
    y = counts / counts.sum()

    # Exploitability inside the matrix: best pure response gain vs mixture.
    exploit_x = float((a @ x).max())     # value of the game is 0 by symmetry
    exploit_y = float((a @ y).max())
    if exploit_y < exploit_x:            # keep whichever solve is tighter
        x, exploit_x = y, exploit_y
    drift = float(np.abs(x - y).max())

    order = np.argsort(-x)
    support = [(names[i], float(x[i])) for i in order if x[i] > 1e-3]
    return {"weights": {names[i]: float(x[i]) for i in range(n)},
            "support": support, "value": 0.0,
            "matrix_exploitability": exploit_x,
            "solver_drift": drift}


def nash_from_file(rules="classic"):
    suffix = "" if rules == "classic" else f"_{rules}"
    with open(os.path.join(MATRIX_DIR, f"matrix{suffix}.json"),
              encoding="utf-8") as fh:
        payload = json.load(fh)
    result = solve_nash(payload["means"], payload["names"])
    print("Nash of the empirical game:")
    for name, weight in result["support"]:
        print(f"  {name:>14}: {weight:.3f}")
    print(f"  in-matrix exploitability {result['matrix_exploitability']:.3f} "
          f"pts/hand, solver drift {result['solver_drift']:.4f}")
    with open(os.path.join(MATRIX_DIR, f"nash{suffix}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result


# ---------------------------------------------------------------------------
# Best response search (the oracle in double oracle) and the hypothesis test.
# ---------------------------------------------------------------------------

def br_search(mixture, hands=600, workers=12, seed=4400,
              rules="classic"):
    """Best response to a Nash mixture, over the heuristic knock family
    (BR_GRID: threshold x p in {0, .5, 1} -- the 0/50/100 discretization).

    Score of a candidate = sum_i w_i * match(candidate, member_i): the exact
    expected payoff against the mixture, no per-hand mixture sampling needed.
    t=0 IS the patient teacher, so the search can never return something
    worse than the mixture's own strongest heuristic.
    """
    support = [(name, w) for name, w in mixture.items() if w > 1e-3]
    results = []
    with mp.Pool(workers) as pool:
        for threshold, p in BR_GRID:
            key = _br_name(threshold, p)
            total, worst = 0.0, None
            for name, w in support:
                m, _, _ = match(key, name, hands, workers, pool,
                                seed + 13 * threshold + int(p * 7), rules)
                total += w * m
                worst = m if worst is None else min(worst, m)
            results.append((total, worst, threshold, p))
            print(f"  BR cand t={threshold} p={p:.1f}: vs-mixture "
                  f"{total:+6.2f} (worst member {worst:+6.2f})", flush=True)
    results.sort(reverse=True)
    total, worst, threshold, p = results[0]
    print(f"BEST RESPONSE: t={threshold} p={p} gains {total:+.2f} pts/hand "
          f"vs the mixture", flush=True)
    return {"gain": total, "threshold": threshold, "p": p,
            "name": _br_name(threshold, p)}


def probe_exploitability(target, hands=400, workers=12, seed=6600,
                         advs=("adv_t0_p100", "adv_t6_p100", "adv_t10_p100"),
                         rules="classic"):
    """How much can a dedicated adversary gain against `target`?

    Two adversary families: the heuristic knock grid (BR_GRID), and the
    hand-reading rollout adversaries. Reports the best gain found -- an upper
    bound is unknowable, so this is a LOWER bound on exploitability, useful
    for comparing bots to each other under the same probe budget."""
    rows = []
    with mp.Pool(workers) as pool:
        for threshold, p in BR_GRID:
            name = _br_name(threshold, p)
            m, ci, _ = match(name, target, hands, workers, pool,
                             seed + 17 * threshold + int(p * 3), rules)
            rows.append((m, ci, name))
            print(f"  {name:>14} vs {target}: {m:+6.2f} +/- {ci:.2f}",
                  flush=True)
        for adv in advs:
            m, ci, _ = match(adv, target, hands, workers, pool,
                             seed + hash(adv) % 997, rules)
            rows.append((m, ci, adv))
            print(f"  {adv:>14} vs {target}: {m:+6.2f} +/- {ci:.2f}",
                  flush=True)
    rows.sort(reverse=True)
    best = rows[0]
    print(f"EXPLOITABILITY({target}) >= {best[0]:+.2f} +/- {best[1]:.2f} "
          f"(by {best[2]})", flush=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)
    tag = target if rules == "classic" else f"{target}_{rules}"
    with open(os.path.join(MATRIX_DIR, f"exploit_{tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump([{"adv": n, "gain": m, "ci": c} for m, c, n in rows],
                  fh, indent=1)
    return rows


def hypothesis_test(hands=900, workers=12, seed=5500, threshold=6,
                    rules="classic"):
    """The user's range-balancing hypothesis, measured directly.

    Adversary: the rollout expert whose determinizer conditions on the
    TARGET's knock policy (adv_t{threshold}_p*). Targets: MixedKnock at the
    same threshold with p in {0, .5, 1}. If mixing protects the non-knock
    range, the adversary's edge should be smallest against p=0.5.
    """
    rows = {}
    with mp.Pool(workers) as pool:
        for p in (0.0, 0.5, 1.0):
            adv = f"adv_t{threshold}_p{int(p * 100)}"
            target = _br_name(threshold, p)
            mean, ci, n = match(adv, target, hands, workers, pool,
                                seed + int(p * 10), rules)
            rows[p] = (mean, ci)
            print(f"  adversary(model p={p:.1f}) vs MixedKnock(t={threshold},"
                  f"p={p:.1f}): {mean:+.2f} +/- {ci:.2f} ({n} hands)",
                  flush=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)
    hyp = "hypothesis" if rules == "classic" else f"hypothesis_{rules}"
    with open(os.path.join(MATRIX_DIR, hyp + ".json"), "w",
              encoding="utf-8") as fh:
        json.dump({str(p): rows[p] for p in rows}, fh, indent=1)
    return rows




def main():
    ap = argparse.ArgumentParser("empirical meta-game")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--nash", action="store_true")
    ap.add_argument("--br", action="store_true",
                    help="best-response search vs the saved Nash mixture")
    ap.add_argument("--exploit", default=None,
                    help="probe exploitability of this population member")
    ap.add_argument("--hypothesis", action="store_true",
                    help="range-balancing test: hand-reading adversary vs "
                         "MixedKnock targets p in {0, .5, 1}")
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    if args.matrix:
        build_matrix(args.hands, args.workers, args.seed)
    if args.nash:
        nash_from_file()
    if args.br:
        with open(os.path.join(MATRIX_DIR, "nash.json"),
                  encoding="utf-8") as fh:
            nash = json.load(fh)
        br_search(nash["weights"], hands=args.hands // 2 or 400,
                  workers=args.workers, seed=args.seed + 4400)
    if args.hypothesis:
        hypothesis_test(args.hands, args.workers, args.seed + 5500)
    if args.exploit:
        probe_exploitability(args.exploit, max(300, args.hands // 2),
                             args.workers, args.seed + 6600)


if __name__ == "__main__":
    main()
