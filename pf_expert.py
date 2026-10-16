"""
Track 1 (v6): the particle belief inside the v5 searcher.

ParticleBelief duck-types belief.Belief for AdaptiveExpert -- update()
pulls move_sheet increments (now including the hero's own draws, which
are hero-legal) and feeds them to particle.ParticleCore; sample_free_
cards() returns a weight-proportional particle. The searcher itself is
UNTOUCHED v5 (adaptive.AdaptiveExpert, promoted config); the only
manipulated variable is where worlds come from.

Two world-weight modes (the (10,1.0) veto vs the PF's own no-knock
evidence overlap -- keeping both double-counts):
  veto=True   keep v5's _world_weight knock rejection (minimal change)
  veto=False  _world_weight = 1.0; the particle posterior already
              carries the no-knock evidence via p_knock

    .venv/Scripts/python.exe pf_expert.py --vs v5 --hands 1500 --seed 62100
    .venv/Scripts/python.exe pf_expert.py --vs v5 --no-veto ...
"""

import argparse
import math
import multiprocessing as mp
import random
import time

import numpy as np

from rlcard.games.gin_rummy.utils.move import (
    DealHandMove, DiscardMove, DrawCardMove, PickupDiscardMove)

import sim
from adaptive import AdaptiveExpert
from particle import DEFAULT_PF, ParticleCore, make_model
from sim import HomePassMove, cid, ids_of, playout

V5_CFG = dict(stages=(16, 48, 192), deep=True, deep_from=2, kbranch=3,
              endgame_stock=0, mid_alpha=0.5, truncate=16, z=1.5)


class ParticleBelief:
    """Belief-compatible wrapper around ParticleCore for one hand."""

    def __init__(self, hero_id, cfg=None, seed=0):
        self.hero_id = hero_id
        self.cfg = dict(DEFAULT_PF, **(cfg or {}))
        self.rng = np.random.default_rng(seed)
        self.core = None
        self._processed = 0
        self._pile = []                  # resolution replay pile
        self.desyncs = 0

    # -- Belief-compat attribute surface (search loop reads these) ----------

    @property
    def opp_discarded(self):
        return self.core.bel.opp_discarded if self.core is not None \
            else np.zeros(52, dtype=np.float32)

    @property
    def opp_declined(self):
        return self.core.bel.opp_declined if self.core is not None \
            else np.zeros(52, dtype=np.float32)

    @property
    def hero_discarded(self):
        return self.core.bel.hero_discarded if self.core is not None \
            else np.zeros(52, dtype=np.float32)

    @property
    def opp_pickups(self):
        return self.core.bel.opp_pickups if self.core is not None else 0

    @property
    def opp_declines(self):
        return self.core.bel.opp_declines if self.core is not None else 0

    # -- event resolution ----------------------------------------------------

    def _resolve(self, game, start):
        """Resolve move_sheet[start:] into ParticleCore events. Reads
        only hero-legal facts: move types, movers, public cards, and the
        hero's own drawn cards."""
        out = []
        sheet = game.round.move_sheet
        for i in range(start, len(sheet)):
            move = sheet[i]
            if isinstance(move, DealHandMove):
                continue
            is_hero = move.player.player_id == self.hero_id
            if isinstance(move, HomePassMove):
                # opening pass: decline evidence only, no card moved
                out.append(("hero_pass" if is_hero else "opp_pass",
                            None))
            elif isinstance(move, DrawCardMove):
                if is_hero:
                    out.append(("hero_draw", cid(move.card)))
                else:
                    out.append(("opp_draw", None))
            elif isinstance(move, PickupDiscardMove):
                taken = self._pile.pop()
                out.append(("hero_pickup" if is_hero else "opp_pickup",
                            taken))
            elif isinstance(move, DiscardMove):
                card_id = cid(move.action.card)
                self._pile.append(card_id)
                out.append(("hero_discard" if is_hero else "opp_discard",
                            card_id))
        return out, len(sheet)

    def update(self, game):
        if self.core is None:
            # home-variant flip seeds the pile (nobody's discard), sets
            # the 10+10 deal arithmetic, and fixes the knock threshold
            # the no-knock evidence uses (the flip's deadwood value,
            # public via game.settings).
            flip_id = getattr(game.round, "_home_flip_id", None)
            if flip_id is not None:
                self._pile.append(flip_id)
            events, self._processed = self._resolve(game, 0)
            # reconstruct the hero's dealt hand by unwinding hero moves
            hero0 = set(ids_of(game.round.players[self.hero_id].hand))
            for kind, card in reversed(events):
                if kind in ("hero_draw", "hero_pickup"):
                    hero0.discard(card)
                elif kind == "hero_discard":
                    hero0.add(card)
            opp_size = (20 if flip_id is not None else 21) - len(hero0)
            model = make_model(
                self.cfg,
                knock_max=game.settings.going_out_deadwood_count)
            self.core = ParticleCore(
                hero0, opp_size, self.rng, model=model,
                n_particles=self.cfg["n_particles"],
                kdraw=self.cfg["kdraw"],
                pile0=[flip_id] if flip_id is not None else None,
                mh_all_takes=self.cfg.get("mh_all_takes", False))
            for ev in events:
                self.core.apply_event(*ev)
        else:
            events, self._processed = self._resolve(game,
                                                    self._processed)
            for ev in events:
                self.core.apply_event(*ev)

    # -- sampling (what the searcher consumes) -------------------------------

    def sample_free_cards(self, unseen_ids, num_hidden, rng, boost=0.0):
        core = self.core
        if core is None or core.h != num_hidden:
            self.desyncs += 1            # fallback: uniform (never seen
            pick = rng.choice(len(unseen_ids), size=num_hidden,
                              replace=False)     # in testing; safety net)
            return [unseen_ids[i] for i in pick]
        return core.sample(rng)

    def marginals(self, unseen_ids, num_hidden, known_ids=()):
        if self.core is None:
            p = np.zeros(52, dtype=np.float32)
            if unseen_ids:
                p[list(unseen_ids)] = num_hidden / len(unseen_ids)
        else:
            p = self.core.shrunk_marginal(list(unseen_ids), num_hidden,
                                          self.cfg["kappa"])
        p = p.astype(np.float32)
        for i in known_ids:
            p[i] = 1.0
        return p


class PFExpert(AdaptiveExpert):
    """v5 search; optionally drop the (10,1.0) world veto (the particle
    posterior's no-knock evidence subsumes it when p_knock > 0)."""

    def __init__(self, veto=True, **kw):
        super().__init__(**kw)
        self._veto = veto

    def _world_weight(self, opp_hand_ids, opp_discard_count):
        if not self._veto:
            return 1.0
        return super()._world_weight(opp_hand_ids, opp_discard_count)


class PFPolicy:
    """Engine-facing wrapper, token pattern. `search_cfg` overrides
    V5_CFG keys (stages/kbranch/... -- the Track 3 ladder knobs)."""

    def __init__(self, seed=0, pf_cfg=None, veto=True, search_cfg=None):
        cfg = dict(V5_CFG, **(search_cfg or {}))
        self.expert = PFExpert(veto=veto, opp_model=(10, 1.0),
                               sample_boost=0.0, seed=seed, **cfg)
        self.pf_cfg = pf_cfg
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
                rnd.current_player_id, cfg=self.pf_cfg,
                seed=self.seed * 100003 + self._hand_no)
        t0 = time.perf_counter()
        action = self.expert.decide(game, self._belief)
        self.move_time += time.perf_counter() - t0
        self.move_count += 1
        return action


def _chunk(job):
    opp_name, hands, seat, seed, rules, pf_cfg, veto, search_cfg = job
    sim.set_rules(rules)
    import meta
    rng = random.Random(seed)
    pol_a = PFPolicy(seed=seed, pf_cfg=pf_cfg, veto=veto,
                     search_cfg=search_cfg)
    pol_b = meta.population()[opp_name]()
    points = []
    for _ in range(hands):
        game = sim.new_game(rng.randrange(1 << 30))
        policies = {seat: pol_a, 1 - seat: pol_b}
        payoff0 = playout(game, policies)
        points.append((payoff0 if seat == 0 else -payoff0) * 100.0)
    return points, pol_a.move_time, pol_a.move_count


def pfmatch(opp_name, hands, workers, seed, rules="full", pf_cfg=None,
            veto=True, search_cfg=None, tag=""):
    per = max(1, hands // (2 * workers))
    jobs = [(opp_name, per, seat, seed + 31 * w + seat, rules, pf_cfg,
             veto, search_cfg) for w in range(workers)
            for seat in (0, 1)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_chunk, jobs)
    points = np.concatenate([r[0] for r in results])
    move_t = sum(r[1] for r in results)
    move_n = sum(r[2] for r in results)
    mean = float(points.mean())
    ci = 1.96 * float(points.std(ddof=1)) / math.sqrt(len(points))
    print(f"pf-v6[veto={veto}{',' + tag if tag else ''}] vs {opp_name} "
          f"[{rules}]: {mean:+.2f} +/- {ci:.2f} over {len(points)} "
          f"hands ({move_t / max(move_n, 1):.3f} s/searched-move, "
          f"{time.time() - t0:.0f}s wall)", flush=True)
    return mean, ci, len(points)


def main():
    ap = argparse.ArgumentParser("v6 particle-belief search gates")
    ap.add_argument("--vs", default="v5")
    ap.add_argument("--hands", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rules", default="full")
    ap.add_argument("--seed", type=int, default=62100)
    ap.add_argument("--no-veto", dest="veto", action="store_false")
    ap.add_argument("--n", type=int, default=0,
                    help="override n_particles")
    ap.add_argument("--stages", default="",
                    help="override search stages, e.g. 16,48,384")
    ap.add_argument("--kbranch", type=int, default=0)
    ap.add_argument("--calibrated", action="store_true",
                    help="use the Brier-grid winner PF config")
    ap.add_argument("--pickup-rule", action="store_true",
                    help="v7 candidate: binary-rule pickup likelihood "
                         "+ mh_all_takes evidence-forgetting fix")
    ap.add_argument("--tau_p", type=float, default=0.1,
                    help="rule-mismatch rate for --pickup-rule")
    args = ap.parse_args()
    pf_cfg = {}
    if args.n:
        pf_cfg["n_particles"] = args.n
    if args.calibrated:                  # results/v6_particle_calibration.txt
        pf_cfg.update(tau_d=1.0, mu=0.0, p_knock=0.0)
    if args.pickup_rule:
        pf_cfg.update(pickup_mode="rule", tau_p=args.tau_p,
                      mh_all_takes=True)
    search_cfg = {}
    tag = []
    if args.stages:
        search_cfg["stages"] = tuple(
            int(s) for s in args.stages.split(","))
        tag.append("st" + args.stages)
    if args.kbranch:
        search_cfg["kbranch"] = args.kbranch
        tag.append(f"kb{args.kbranch}")
    if args.calibrated:
        tag.append("cal")
    if args.pickup_rule:
        tag.append(f"rule/tp{args.tau_p}")
    pfmatch(args.vs, args.hands, args.workers, args.seed, args.rules,
            pf_cfg or None, args.veto, search_cfg or None,
            ",".join(tag))


if __name__ == "__main__":
    main()
