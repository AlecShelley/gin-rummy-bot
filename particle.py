"""
Track 1 (v6): particle-filter opponent-hand belief.

Replaces belief.py's per-card multiplicative weights with trajectory-level
Bayesian inference: N particles, each a concrete hypothesis of the
opponent's hidden cards, weighted per public event by
P(observed action | particle hand, softened opponent model). A discard
then carries its full content ("that was their least useful card, so
everything they KEPT looks better than it"), compounding over the
history -- the poker range update. Survey-backed design
(results/v6_prior_art.md section 2):

  * Likelihood = softmax(-energy / tau) over the opponent's options,
    energy = deadwood-after + mu * live-count (the belief_teacher.py
    scalarization of FastTeacher's lexicographic rule), with an
    epsilon-uniform floor so no observed action zeroes a particle
    (Kermit's brittleness lesson).
  * Opponent stock-draw + discard-c turns are processed ATOMICALLY with
    proper importance weights: a particle holding c samples the kept
    draw uniformly from its pool (proposal == true draw distribution, so
    the importance factor is the discard likelihood alone); a particle
    lacking c is forced into the "drew c and dropped it" branch, whose
    true draw probability 1/|pool| enters the weight explicitly. Bayes
    prices drew-and-dropped correctly with no repair heuristic.
  * No-knock evidence: every opponent discard turn where they COULD
    knock but didn't multiplies by (1 - p_knock) -- the softened,
    per-turn-compounding generalization of the (10, 1.0) world veto,
    and the symmetry-breaking statistic the search actually consumes
    (lesson #12; v4's +6.59 came from exactly this channel).
  * Hero stock draws refute particles holding the drawn card: repaired
    by swapping in a fresh pool card (all consistent completions are
    equally likely given the event, so no weight change), preserving
    diversity where kill-and-resample would collapse it.
  * ESS < N/2 triggers systematic resampling.

Offline gate (run this before ANY searcher integration):

    .venv/Scripts/python.exe particle.py --calibrate --hands 400

logs full two-seat public transcripts with ground truth (teacher, clone,
random opponents), grid-fits (tau_d, tau_p, p_knock, mu) on a train
split, and reports held-out Brier for uniform / current Belief /
particle filter side by side. The particle filter must CLEARLY beat the
current belief (held-out ~0.179, +12.5% over uniform) on teacher AND
clone streams and stay sane on random before touching the searcher.
"""

import argparse
import math
import multiprocessing as mp
import random as pyrandom
import time

import numpy as np

from belief import _ADJ, Belief
from sim import DEADWOOD_VALUE, cid, ids_of, min_deadwood_ids

# Defaults; --calibrate overwrites (results/v6_particle_calibration.txt).
# lik_floor 0.02 + resample-move measured clearly better than the
# heavier 0.10 clip once MH rejuvenation exists (dev, 2026-08-27).
# pickup_mode/mh_all_takes: the v7 likelihood campaign knobs (HANDOFF_V7
# items 1-3; PROMOTED 2026-08-31, results/v8_gates.log). "rule" models
# the teacher's binary take rule with a meld tier and tau_p
# reinterpreted as the rule-MISMATCH rate (fixes the hidden-5S
# inverted-likelihood defect); mh_all_takes=True makes rejuvenation
# re-score ALL pickup/decline evidence (rare, information-dense)
# instead of letting MH swaps walk particles away from old constraints,
# windowing only the discards. The v6-as-gated behavior =
# pickup_mode="logistic", tau_p=1.0, mh_all_takes=False (the frozen
# registry v6 pins these).
DEFAULT_PF = dict(tau_d=2.0, mu=0.3, tau_p=0.1, p_knock=0.5, eps=0.02,
                  lik_floor=0.02, n_particles=256, kdraw=4, kappa=80.0,
                  pickup_mode="rule", mh_all_takes=True, draw_gain=1,
                  tau_po=0.0)
# tau_po=0.0 + eps=0.02: the v10 likelihood-sharpening promotion
# (2026-09-07, results/v10_gates.log). Opening pass/take events are
# book-DETERMINISTIC in deployment, so their mismatch rate is zero
# (the posterior ground-truth check measured 10x excess weight on
# book-refuted hands under tau_p=0.1); the mid-game rate stays 0.1.
# Gated: truth reads hold-or-sharpen; stream Brier >= champion
# (random-mismatch -0.28%); three fresh ground-truth scenarios all
# improve; match TIE (pooled -0.72 +/- 1.10 @ 4000); probes clean
# (book-inverter +0.90 +/- 1.81, ad_self/ad_jam margins hold).
# Frozen pre-v10 identities pin eps=0.05, tau_po=None explicitly
# (meta.py, play_vs_bot.py, read_regress.py, home_variant.py).

# V11 fitted discard weights (discard_fit.py, 186k v10home self-play
# decisions, held-out logloss: pickup 1.124 vs energy-baseline 1.220,
# draw 0.940 vs 1.044 -- results/v11_gates.log). Feature order:
# (dw_after - best_dw, live-in-their-unseen, DEADWOOD_VALUE,
#  in-hand adjacency, is-the-drawn-card). Selected by
# discard_mode="fit"; the "energy" default is untouched.
# v11b 6-feature weights (feature order: dw_after - best_dw,
# live-in-their-unseen, DEADWOOD_VALUE, in-hand adjacency,
# is-the-drawn-card, MELD PROTECTION = VALUE + dw_after - dw_before).
# The 6th feature separates "dump big junk" from "never break melds"
# -- in the 5-feature fit the value weight absorbed both (the home10
# four-king failure); with meldprot split out, value goes ~0 and
# protection is strongly negative. Held-out logloss vs energy
# baseline: open 1.581/1.634, pickup 1.058/1.166, draw 0.940/1.044.
V11_W_PICKUP = np.array([-0.5002, -0.3089, 0.0995, -0.6517, 0.0,
                         -0.4007])
V11_W_DRAW = np.array([-0.4882, -0.4722, 0.0557, -0.7255, 0.1375,
                       -0.4325])
# "open" = the hand's first discard after TAKING the flip (pile empty
# at decision time) -- the v11 G2 retry split: these discards are
# less structured and carry distinct weights; lumping them into
# "pickup" regressed the S2 posterior benchmark by 4.2 points.
V11_W_OPEN = np.array([-0.4916, -0.1405, -0.0938, -0.4998, 0.0,
                       -0.5854])


def make_model(cfg, knock_max=10):
    """SoftOpponent from a DEFAULT_PF-style dict -- the ONE construction
    path (pf_expert / evaluate / the offline gates), so new likelihood
    knobs propagate everywhere by being added here."""
    return SoftOpponent(
        tau_d=cfg["tau_d"], mu=cfg["mu"], tau_p=cfg["tau_p"],
        p_knock=cfg["p_knock"], eps=cfg["eps"],
        lik_floor=cfg["lik_floor"], knock_max=knock_max,
        pickup_mode=cfg.get("pickup_mode", "logistic"),
        draw_gain=cfg.get("draw_gain", 1), nu=cfg.get("nu", 0.0),
        sigma=cfg.get("sigma", 0.0),
        discard_mode=cfg.get("discard_mode", "energy"),
        eps_out=cfg.get("eps_out", 0.05),
        tau_po=cfg.get("tau_po", None))

_ADJ_MASK = np.zeros((52, 52), dtype=bool)
for _c in range(52):
    _ADJ_MASK[_c][_ADJ[_c]] = True


def _live(card_id, pool_mask):
    return int(np.count_nonzero(_ADJ_MASK[card_id] & pool_mask))


class SoftOpponent:
    """Softened FastTeacher likelihoods, id-space, mask-based."""

    def __init__(self, tau_d=2.0, mu=0.3, tau_p=0.1, p_knock=0.5,
                 eps=0.05, lik_floor=0.10, knock_max=10,
                 pickup_mode="rule", draw_gain=1, nu=0.0, sigma=0.0,
                 discard_mode="energy", eps_out=0.05, tau_po=None):
        self.tau_d = tau_d               # discard softmax temperature
        self.mu = mu                     # live-count weight in the energy
        # nu (v10, 2026-09-06): discarded-card VALUE term in the energy.
        # The posterior ground-truth check (results/v9_gates.log tail)
        # showed deadwood-TIED discards are tie-broken near-
        # deterministically toward the higher-value dead card by the
        # deployment opponent; the nu-less energy scored them ~50/50
        # (the missing discard-choice counterfactual). nu=0.0
        # reproduces the frozen v6/v7 energy exactly.
        self.nu = nu
        # sigma (v10, 2026-09-06): IN-HAND synergy term -- discarding a
        # card whose meld partners sit in the same hand is penalized.
        # The energy was deadwood-myopic: dropping a drawn 7S while
        # holding 7D SHEDS deadwood, so the model approved, while the
        # real opponent keeps the pair (S3 ground truth: P(7D) 0.087
        # empirical vs 0.232 modeled). mu*live only counts UNSEEN
        # neighbors and cannot express this. sigma=0.0 = frozen v6/v7.
        self.sigma = sigma
        # discard_mode "tier" (v10, 2026-09-06): the ground-truth rows
        # showed the observed discard sits in the DEADWOOD-ARGMIN set
        # for 100% of 1500 true hands, while the teacher tie-break
        # within that set matches only 23% (~uniform-within). The
        # likelihood is therefore TWO-SCALE: (1-eps_out) mass on the
        # argmin set (distributed by the mild energy softmax within
        # it), eps_out on the rest (energy softmax) -- sharp between
        # tiers, soft within. A single softmax temperature couples the
        # two scales, which is why sharpening tau_d backfired
        # (results/v10_gates.log). "energy" = frozen v6/v7 behavior.
        self.discard_mode = discard_mode
        self.eps_out = eps_out
        self.tau_po = tau_po             # opening mismatch (see pickup_prob)
        # tau_p: pickup logistic temperature ("logistic" mode) or the
        # rule-MISMATCH rate ("rule" mode) -- see pickup_prob.
        self.tau_p = tau_p
        self.pickup_mode = pickup_mode
        self.draw_gain = draw_gain       # the teacher's take threshold
        self.p_knock = p_knock           # knock prob when knock is legal
        self.knock_max = knock_max       # knock threshold (home: the flip)
        self.eps = eps                   # uniform floor mix
        # Robust likelihood clip: no single event term may down-weight a
        # particle below this probability. Measured (dev, 2026-08-27):
        # floor 0.1 beats none on held-out Brier -- strong NEGATIVE
        # evidence per event is mostly ensemble-killing noise -- and it
        # is simultaneously the model-mismatch insurance the prior-art
        # survey calls for (Kermit's XSkat lesson).
        self.lik_floor = lik_floor

    def discard_logprob(self, hand11, discard_id, their_unseen_mask,
                        ctx="pickup", drawn=None):
        """(log P(discard `discard_id` | 11-card hand), best post-discard
        deadwood) -- the latter feeds the no-knock evidence for free.
        ctx/drawn (v11): the turn context -- "pickup" (hand complete
        from a take) vs "draw" (post-stock-draw, `drawn` = the drawn
        card when known). Only discard_mode="fit" consumes them."""
        energies = np.empty(len(hand11))
        dws = np.empty(len(hand11), dtype=int)
        best_dw = 999
        target_i = -1
        hand_mask = np.zeros(52, dtype=bool)
        hand_mask[list(hand11)] = True
        for j, drop in enumerate(hand11):
            dw = min_deadwood_ids([i for i in hand11 if i != drop])
            dws[j] = dw
            best_dw = min(best_dw, dw)
            energies[j] = dw + self.mu * _live(drop, their_unseen_mask) \
                - self.nu * DEADWOOD_VALUE[drop] \
                + self.sigma * _live(drop, hand_mask)
            if drop == discard_id:
                target_i = j
        if target_i < 0:                 # not in hand: impossible event
            return math.log(1e-9), best_dw
        if self.discard_mode == "fit" or (
                self.discard_mode == "fitdraw" and ctx == "draw"):
            # "fitdraw" (the v11 reduced candidate): fitted weights
            # only where the defect was measured (post-draw discards);
            # pickup/open keep the frozen energy model (the full fit
            # regressed the S2 posterior benchmark).
            w = {"draw": V11_W_DRAW, "open": V11_W_OPEN}.get(
                ctx, V11_W_PICKUP)
            dw11 = min_deadwood_ids(list(hand11))
            z = np.empty(len(hand11))
            for j, drop in enumerate(hand11):
                z[j] = (w[0] * (dws[j] - best_dw)
                        + w[1] * _live(drop, their_unseen_mask)
                        + w[2] * DEADWOOD_VALUE[drop]
                        + w[3] * _live(drop, hand_mask)
                        + w[4] * (1.0 if drop == drawn else 0.0)
                        + w[5] * (DEADWOOD_VALUE[drop]
                                  + dws[j] - dw11))
            z -= z.max()
            pz = np.exp(z)
            p = float(pz[target_i] / pz.sum())
            p = (1.0 - self.eps) * p + self.eps / len(hand11)
            return math.log(max(p, self.lik_floor)), best_dw
        arr = energies / -self.tau_d
        arr -= arr.max()
        probs = np.exp(arr)
        if self.discard_mode == "tier":
            in_a = dws == best_dw
            share = probs * in_a
            out = probs * ~in_a
            if in_a[target_i]:
                p = (1.0 - self.eps_out) * float(
                    probs[target_i] / max(share.sum(), 1e-12))
            else:
                p = self.eps_out * float(
                    probs[target_i] / max(out.sum(), 1e-12))
        else:
            p = float(probs[target_i] / probs.sum())
        p = (1.0 - self.eps) * p + self.eps / len(hand11)
        return math.log(max(p, self.lik_floor)), best_dw

    def pickup_prob(self, hand10, top_id, opening=0):
        """P(take the upcard | 10-card hand).

        "logistic" (v6): sigmoid((gain - 0.5) / tau_p) over the deadwood
        gain. MEASURED DEFECT (the hidden-5S incident, 2026-08-31): the
        gain magnitude conflates "how big the deadwood swing" with
        "whether they take" -- a junk hand dumping a king swings 6+
        points (p ~ 0.97) while a developed hand extending a run swings
        1-2 (p ~ 0.6) even though the teacher takes BOTH
        deterministically. The likelihood is inverted exactly on the
        takes that prove melds.

        "rule" (v7): the teacher's actual binary rule -- take iff
        gain >= draw_gain -- with tau_p reinterpreted as the rule-
        MISMATCH rate, plus a meld tier: a card that slots into the
        hand's melds (contributes zero marginal deadwood) is taken
        near-certainly by ANY rational player, so meld takes sit at the
        top of the scale and can never score below junk-dump takes.
          meld/extension take : 1 - 0.3 * tau_p
          rule take (no meld) : 1 - tau_p
          rule says pass      : 0.5 * tau_p   (speculative-take floor)
        """
        hand10 = list(hand10)
        before = min_deadwood_ids(hand10)
        candidate = hand10 + [top_id]
        if self.pickup_mode == "rule":
            # tau_po (v10, 2026-09-06): opening-specific mismatch rate.
            # At flip-option nodes the deployment opponent plays
            # sim.flip_book_take DETERMINISTICALLY, so the ground-truth
            # posterior check showed tau_p=0.1 leaves ~10x excess weight
            # on hands the book would have taken with; mid-game takes
            # are SEARCH decisions that really do deviate from the gain
            # rule, so tau_p stays soft there. None = frozen v6/v7.
            t = self.tau_po if (opening and self.tau_po is not None) \
                else self.tau_p
            # meld tier first: adding the card leaves deadwood <= before
            # iff it joins/completes a meld; every such take also
            # satisfies the gain rule (dropping any deadwood card then
            # strictly improves), and needs no 10-drop scan.
            if min_deadwood_ids(candidate) <= before:
                p = 1.0 - 0.3 * t
            elif opening:
                # OPENING option (home rule): the book -- first player
                # takes iff it melds; the SECOND player also takes
                # with adjacency >= 2 (the extra-turn clause,
                # p2adj_gate.py +3.05 +/- 1.50).
                if opening == 2:
                    fr, fs = top_id % 13, top_id // 13
                    adj = sum(1 for c in hand10
                              if c % 13 == fr or (c // 13 == fs
                                                  and abs(c % 13 - fr)
                                                  <= 2))
                    p = (1.0 - t) if adj >= 2 else 0.5 * t
                else:
                    p = 0.5 * t
            else:
                after = min(min_deadwood_ids(
                    [i for i in candidate if i != drop])
                    for drop in hand10)
                gain = before - after
                p = (1.0 - self.tau_p) if gain >= self.draw_gain \
                    else 0.5 * self.tau_p
        else:
            after = min(min_deadwood_ids(
                [i for i in candidate if i != drop]) for drop in hand10)
            gain = before - after
            p = 1.0 / (1.0 + math.exp(-(gain - 0.5) / self.tau_p))
        p = (1.0 - self.eps) * p + self.eps * 0.5
        return min(max(p, self.lik_floor), 1.0 - self.lik_floor)

    def no_knock_logprob(self, best_dw):
        """log P(they did NOT knock this turn | best post-discard dw)."""
        if best_dw > self.knock_max:
            return 0.0                   # knock illegal, no evidence
        return math.log(max(1.0 - self.p_knock, 1e-3))


class ParticleCore:
    """The filter, driven by resolved public events (no game object) so
    the offline harness and the live wrapper share one code path.

    Events (card is an int id or None):
        ('opp_draw', None)      opp drew from stock (decline evidence is
                                implicit via the tracked pile top)
        ('opp_pickup', card)    opp took the pile top
        ('opp_discard', card)
        ('hero_draw', card)     hero's own stock draw (hero-legal info)
        ('hero_pickup', card)
        ('hero_discard', card)
    """

    def __init__(self, hero_hand0, opp_hand_size, rng, model=None,
                 n_particles=256, kdraw=4, mh_window=6, pile0=None,
                 mh_all_takes=True):
        self.model = model or SoftOpponent()
        self.rng = rng
        self.n = n_particles
        self.kdraw = kdraw               # kept-draw subsample (APF step)
        self.hero_hand = set(hero_hand0)
        # pile0: cards already face-up at hand start (the home variant's
        # flip) -- excluded from the initial particle pool.
        self.pile = list(pile0) if pile0 else []
        self.opp_known = set()
        self.hero_known_to_opp = set()
        pool = [c for c in range(52) if c not in self.hero_hand
                and c not in self.pile]
        self.h = opp_hand_size           # opp hidden-card count
        self.particles = []
        for _ in range(self.n):
            pick = self.rng.choice(len(pool), size=self.h, replace=False)
            self.particles.append({pool[i] for i in pick})
        self.logw = np.zeros(self.n)
        self._pending = None             # opp draw-type event this turn
        # A classic calibrated Belief rides along: its per-card weights
        # guide repair-swap proposals (rejuvenation toward the event-
        # weight posterior instead of uniform dilution).
        self.bel = Belief(hero_id=0)
        # Resample-move (Gilks & Berzuini): after each resample, one MH
        # card-swap proposal per particle, accepted on the likelihood of
        # the last `mh_window` opponent events re-evaluated under the
        # swapped hand (frozen-composition approximation). Restores the
        # diversity that sharp evidence + resampling destroy, so the
        # likelihood can stay selective instead of being floor-clipped
        # into blandness.
        self.mh_window = mh_window
        # mh_all_takes (v7 fix, HANDOFF_V7 item 3): the windowed
        # re-score let MH swaps walk particles away from OLD
        # pickup/decline constraints (evidence forgetting). True =
        # always re-score every pickup/decline (rare, information-
        # dense); the window then applies to discards only.
        self.mh_all_takes = mh_all_takes
        self._opp_evlog = []             # (kind, card) opp events seen
        # Home OPENING tracking: flip-option takes/declines carry
        # book evidence (option 1 = meld-only, option 2 adds the
        # adjacency clause); the opening ends at the first draw or
        # discard. Harmless for standard games.
        self._opening = True
        self._hero_passed = False        # hero passed -> opp holds opt 2

    # -- masks --------------------------------------------------------------

    def _pool_mask(self):
        """Cards unseen to hero (stock + opp hidden), as a 52-bool mask."""
        m = np.ones(52, dtype=bool)
        for c in self.hero_hand:
            m[c] = False
        for c in self.pile:
            m[c] = False
        for c in self.opp_known:
            m[c] = False
        return m

    def _their_unseen_mask(self, hand11):
        """What the OPPONENT cannot see given their hypothesized hand."""
        m = np.ones(52, dtype=bool)
        for c in hand11:
            m[c] = False
        for c in self.pile:
            m[c] = False
        for c in self.hero_known_to_opp:
            m[c] = False
        return m

    # -- events -------------------------------------------------------------

    def _swap_in(self, p, pool_ids):
        """Belief-weighted replacement card for a repaired particle."""
        avail = [c for c in pool_ids if c not in p]
        if not avail:
            return None
        w = np.maximum(self.bel.w[avail], 1e-9)
        return avail[int(self.rng.choice(len(avail), p=w / w.sum()))]

    def apply_event(self, kind, card):
        if kind in ("hero_draw", "opp_draw", "hero_discard",
                    "opp_discard"):
            self._opening = False
        if kind == "hero_draw":
            self.hero_hand.add(card)
            pool_ids = None
            for p in self.particles:
                if card in p:            # refuted: repair by swap
                    p.discard(card)
                    if pool_ids is None:
                        m = self._pool_mask()
                        pool_ids = [c for c in range(52) if m[c]]
                    d = self._swap_in(p, pool_ids)
                    if d is not None:
                        p.add(d)
        elif kind == "hero_pickup":
            taken = self.pile.pop()
            self.hero_hand.add(taken)
            self.hero_known_to_opp.add(taken)
        elif kind == "hero_discard":
            self.hero_hand.discard(card)
            self.hero_known_to_opp.discard(card)
            self.pile.append(card)
        elif kind == "hero_pass":
            self._hero_passed = True     # opp's opening act = option 2
        elif kind == "opp_pass":
            # opening pass (home rule): pure decline evidence on the
            # flip -- like opp_draw's decline term, but NO draw happens
            # and no discard follows, so _pending stays untouched.
            # Meld-only (opening-book) likelihood: a pass says little
            # (the book passes everything that doesn't meld).
            if self.pile:
                top = self.pile[-1]
                opt = 2 if self._hero_passed else 1
                self.bel._evt_decline(top)
                self._opp_evlog.append((f"decline_open{opt}", top))
                for i, p in enumerate(self.particles):
                    hand10 = list(p) + list(self.opp_known)
                    pk = self.model.pickup_prob(hand10, top,
                                                opening=opt)
                    self.logw[i] += math.log(max(1.0 - pk, 1e-6))
        elif kind == "opp_draw":
            if self.pile:                # decline evidence on the top card
                top = self.pile[-1]
                self.bel._evt_decline(top)
                self._opp_evlog.append(("decline", top))
                for i, p in enumerate(self.particles):
                    hand10 = list(p) + list(self.opp_known)
                    pk = self.model.pickup_prob(hand10, top)
                    self.logw[i] += math.log(max(1.0 - pk, 1e-6))
            self._pending = "stock"
        elif kind == "opp_pickup":
            taken = self.pile.pop()
            self.bel._evt_opp_pickup(taken)
            opt = 0 if not self._opening else \
                (2 if self._hero_passed else 1)
            self._opp_evlog.append(
                (f"pickup_open{opt}" if opt else "pickup", taken))
            for i, p in enumerate(self.particles):
                hand10 = list(p) + list(self.opp_known)
                pk = self.model.pickup_prob(hand10, taken, opening=opt)
                self.logw[i] += math.log(max(pk, 1e-6))
            self.opp_known.add(taken)
            self._pending = "pickup"
        elif kind == "opp_discard":
            self.bel._evt_opp_discard(card)
            self.bel._clamp()
            # v11: tag the turn context so MH rescoring can replay the
            # discard under the right fitted-model context (_d = post-
            # stock-draw, _o = post-take with empty pile, _p = pickup).
            if self._pending == "stock":
                tag = "discard_d"
            elif not self.pile:
                tag = "discard_o"
            else:
                tag = "discard_p"
            self._opp_evlog.append((tag, card))
            self._opp_discard(card)
            self._pending = None
            self._maybe_resample()

    def _opp_discard(self, c):
        """One opponent discard, closing their (draw, discard) turn."""
        stock_turn = self._pending == "stock"
        from_known = c in self.opp_known
        pool = self._pool_mask()
        pool_ids = [x for x in range(52) if pool[x]]
        for i, p in enumerate(self.particles):
            if stock_turn:
                avail = [x for x in pool_ids if x not in p]
                n_avail = max(len(avail), 1)
                if from_known or c in p:
                    # They kept the draw d. Fully-adapted APF step:
                    # subsample kdraw candidate draws, weight by the MEAN
                    # discard likelihood (unbiased estimate of E_d[lik]
                    # under the true uniform draw), and keep the d
                    # sampled proportional to its likelihood -- the
                    # optimal proposal, so kept cards are ones the
                    # opponent would actually have kept.
                    if avail:
                        k = min(self.kdraw, n_avail)
                        sub = self.rng.choice(n_avail, size=k,
                                              replace=False)
                        base = list(p) + list(self.opp_known)
                        liks = np.empty(k)
                        dws = np.empty(k, dtype=int)
                        for j in range(k):
                            d = avail[int(sub[j])]
                            hand11 = base + [d]
                            lp, bdw = self.model.discard_logprob(
                                hand11, c,
                                self._their_unseen_mask(hand11),
                                ctx="draw", drawn=d)
                            liks[j] = math.exp(lp)
                            dws[j] = bdw
                        self.logw[i] += math.log(max(liks.mean(), 1e-12))
                        pick = int(self.rng.choice(
                            k, p=np.maximum(liks, 1e-12)
                            / np.maximum(liks, 1e-12).sum()))
                        best_dw = int(dws[pick])
                        p.add(avail[int(sub[pick])])
                    else:
                        hand11 = list(p) + list(self.opp_known)
                        lp, best_dw = self.model.discard_logprob(
                            hand11, c, self._their_unseen_mask(hand11),
                            ctx="draw")
                        self.logw[i] += lp
                    p.discard(c)         # no-op when from_known
                else:
                    # forced branch: they drew c itself and dropped it
                    hand11 = list(p) + list(self.opp_known) + [c]
                    lp, best_dw = self.model.discard_logprob(
                        hand11, c, self._their_unseen_mask(hand11),
                        ctx="draw", drawn=c)
                    self.logw[i] += lp + math.log(1.0 / n_avail)
                    # particle unchanged: c passed through their hand
            else:
                # pickup turn or an opening 11-card discard: the hand is
                # already complete and must contain c.
                if not from_known and c not in p:
                    # contradiction with no draw to explain it: repair by
                    # swapping a uniform held card out for c, with a
                    # fixed wrong-particle penalty.
                    if p:
                        drop = list(p)[int(self.rng.integers(len(p)))]
                        p.discard(drop)
                    p.add(c)
                    self.logw[i] += math.log(0.05)
                hand11 = list(p) + list(self.opp_known)
                lp, best_dw = self.model.discard_logprob(
                    hand11, c, self._their_unseen_mask(hand11),
                    ctx="open" if not self.pile else "pickup")
                self.logw[i] += lp
                p.discard(c)             # no-op when from_known
            self.logw[i] += self.model.no_knock_logprob(best_dw)
        if from_known:
            self.opp_known.discard(c)
        # hidden-count bookkeeping: a stock turn keeps h (draw +1,
        # discard -1) unless the discard came from known (then h+1); a
        # pickup turn with a hidden discard shrinks h by 1.
        if stock_turn and from_known:
            self.h += 1
        elif (not stock_turn) and (not from_known):
            self.h -= 1
        self.pile.append(c)

    def _maybe_resample(self):
        lw = self.logw - self.logw.max()
        w = np.exp(lw)
        w_sum = w.sum()
        ess = w_sum ** 2 / (w ** 2).sum()
        if ess < self.n / 2.0:
            w /= w_sum
            positions = (self.rng.random() + np.arange(self.n)) / self.n
            cum = np.cumsum(w)
            idx = np.clip(np.searchsorted(cum, positions), 0, self.n - 1)
            self.particles = [set(self.particles[j]) for j in idx]
            self.logw = np.zeros(self.n)
            if self.mh_window > 0 and self._opp_evlog:
                self._rejuvenate()

    def _window_loglik(self, p):
        """Log-likelihood of recent opponent events under hand
        hypothesis `p`, frozen-composition approximation: discarded
        cards are temporarily re-added, pickups evaluated pre-pickup.
        With mh_all_takes, EVERY pickup/decline is re-scored (so MH
        swaps cannot walk particles away from old take evidence) and
        only discards are windowed."""
        total = 0.0
        n_ev = len(self._opp_evlog)
        win_start = n_ev - self.mh_window
        for idx, (kind, card) in enumerate(self._opp_evlog):
            if idx < win_start and not (
                    self.mh_all_takes
                    and not kind.startswith("discard")):
                continue
            if kind.startswith("decline"):
                opt = int(kind[-1]) if kind[-1].isdigit() else 0
                hand = list(p) + [k for k in self.opp_known if k != card]
                total += math.log(max(
                    1.0 - self.model.pickup_prob(hand, card,
                                                 opening=opt), 1e-6))
            elif kind.startswith("pickup"):
                opt = int(kind[-1]) if kind[-1].isdigit() else 0
                hand = list(p) + [k for k in self.opp_known if k != card]
                total += math.log(max(
                    self.model.pickup_prob(hand, card, opening=opt),
                    1e-6))
            else:                        # discard: re-add the gone card
                ctx = {"discard_d": "draw",
                       "discard_o": "open"}.get(kind, "pickup")
                hand11 = list(p) + list(self.opp_known) + [card]
                mask = self._their_unseen_mask(hand11)
                if ctx == "draw" and self.model.discard_mode in (
                        "fit", "fitdraw"):
                    # v11 G4a fix: mirror the PRIMARY two-branch draw
                    # mixture instead of a single sharp evaluation.
                    # (Legacy modes keep the single-call path for
                    # bit-exact frozen behavior.)
                    # In the frozen reconstruction both branches share
                    # this hand11 -- the forced drew-and-dropped branch
                    # has drawn=card (weight 1/n_avail), the kept-draw
                    # branch's true drawn card is one of the held cards
                    # (its isdrawn boost lands on a non-target
                    # candidate: second order), approximated drawn=None.
                    # Without the mixture the fitted weights compound
                    # at every resample and crush partner-holding
                    # hypotheses (measured -0.07/-0.08 on the
                    # truth-keyed reads).
                    pool = self._pool_mask()
                    n_avail = max(
                        1 + sum(1 for x in range(52)
                                if pool[x] and x not in p), 1)
                    lpf, best_dw = self.model.discard_logprob(
                        hand11, card, mask, ctx="draw", drawn=card)
                    lpk, _ = self.model.discard_logprob(
                        hand11, card, mask, ctx="draw", drawn=None)
                    lik = (math.exp(lpf) / n_avail
                           + (1.0 - 1.0 / n_avail) * math.exp(lpk))
                    total += math.log(max(lik, 1e-12)) \
                        + self.model.no_knock_logprob(best_dw)
                else:
                    lp, best_dw = self.model.discard_logprob(
                        hand11, card, mask, ctx=ctx)
                    total += lp + self.model.no_knock_logprob(best_dw)
        return total

    def _rejuvenate(self):
        """One MH card-swap proposal per particle against the windowed
        likelihood (resample-move). Proposal: drop a uniform held card,
        add a belief-weighted pool card; Hastings-corrected by the
        belief-weight ratio."""
        m = self._pool_mask()
        pool_ids = [c for c in range(52) if m[c]]
        if not pool_ids:
            return
        for i, p in enumerate(self.particles):
            if not p:
                continue
            cur_ll = self._window_loglik(p)
            held = list(p)
            a = held[int(self.rng.integers(len(held)))]
            avail = [c for c in pool_ids if c not in p]
            if not avail:
                continue
            wprop = np.maximum(self.bel.w[avail], 1e-9)
            j = int(self.rng.choice(len(avail), p=wprop / wprop.sum()))
            b = avail[j]
            q = set(p)
            q.discard(a)
            q.add(b)
            new_ll = self._window_loglik(q)
            hastings = math.log(max(self.bel.w[a], 1e-9)) - \
                math.log(max(self.bel.w[b], 1e-9))
            if math.log(max(self.rng.random(), 1e-300)) < \
                    new_ll - cur_ll + hastings:
                self.particles[i] = q

    # -- outputs ------------------------------------------------------------

    def weights(self):
        lw = self.logw - self.logw.max()
        w = np.exp(lw)
        return w / w.sum()

    def marginal52(self):
        """P(opp holds c), 52-vector over the hidden portion."""
        w = self.weights()
        m = np.zeros(52)
        for wi, p in zip(w, self.particles):
            for c in p:
                m[c] += wi
        return m

    def sample(self, rng):
        """One particle's hidden ids, drawn proportional to weight."""
        w = self.weights()
        i = int(rng.choice(self.n, p=w))
        return list(self.particles[i])

    def n_eff_distinct(self):
        """Effective number of DISTINCT weighted hands (duplicates from
        resampling collapse), the honest sample size of the marginal."""
        w = self.weights()
        distinct = {}
        for wi, p in zip(w, self.particles):
            key = tuple(sorted(p))
            distinct[key] = distinct.get(key, 0.0) + wi
        dv = np.array(list(distinct.values()))
        return 1.0 / float((dv ** 2).sum())

    def shrunk_marginal(self, unseen_ids, num_hidden, kappa=15.0):
        """marginal52 shrunk toward the ride-along Belief's analytic
        marginal by the finite-particle noise (lam = n_eff/(n_eff+kappa)).
        Measured (2026-08-27): the particle posterior's DIRECTION beats
        the Belief's at every turn (mean mass on truly-held cards), but
        its per-card noise at n_eff ~ 20-40 distinct hands loses on
        Brier; the Belief marginal is the natural low-variance backbone
        to shrink toward. This is what the Brier gate and build_obs
        consume; world SAMPLING uses the particles directly."""
        marg = self.marginal52()
        n_eff = self.n_eff_distinct()
        lam = n_eff / (n_eff + kappa)
        base = self.bel.marginals(list(unseen_ids), num_hidden)
        return lam * marg + (1.0 - lam) * base


# ---------------------------------------------------------------------------
# Offline calibration harness: full two-seat public transcripts with
# ground truth, scored side by side against the current Belief.
# ---------------------------------------------------------------------------

def _log_streams_v6(num_hands, seed, opponent="teacher"):
    """Play hands; per hand return (events, probes, opp_hand_size,
    hero_hand0) where events is the resolved public stream
    [(kind, card_id_or_None), ...] in ParticleCore's vocabulary, and
    probes are (event_index, unseen_ids, hidden_truth_ids) at hero
    decision nodes -- same probe definition as belief._log_streams."""
    import sim
    from rlcard.games.gin_rummy.utils.action_event import (
        DiscardAction, DrawCardAction, PickUpDiscardAction)

    policy0 = sim.FastTeacher()
    if callable(opponent):               # poison.py: adversary builders
        policy1 = opponent()
    elif opponent == "teacher":
        policy1 = sim.FastTeacher()
    elif opponent == "random":
        class RandomFast:
            def __init__(self, rng):
                self.rng = rng

            def act(self, game, legal):
                return legal[self.rng.randrange(len(legal))]
        policy1 = RandomFast(pyrandom.Random(seed + 999))
    else:                                # v2 clone, env-free
        from gin_rummy_bot import GinRummyBot
        from belief import base_obs
        bot = GinRummyBot("results/v2_final/gin_rummy_bot.policy.pt")

        class CloneFast:
            def act(self, game, legal):
                if len(legal) == 1:
                    return legal[0]
                ids = [e.action_id for e in legal]
                probs = bot._probs({"obs": base_obs(game),
                                    "legal_actions": dict.fromkeys(ids)})
                return legal[ids.index(int(np.argmax(probs)))]
        policy1 = CloneFast()

    rng = pyrandom.Random(seed)
    streams = []
    for _ in range(num_hands):
        game = sim.new_game(rng.randrange(1 << 30))
        rnd = game.round
        hero_hand0 = ids_of(rnd.players[0].hand)
        opp_size = len(rnd.players[1].hand)
        events, probes = [], []
        while rnd.going_out_action is None and not rnd.is_over:
            legal = sim.fast_legal_actions(game)
            if rnd.current_player_id == 0 and len(legal) > 1:
                opp = rnd.players[1]
                unseen = ids_of(rnd.dealer.stock_pile) + \
                    [cid(c) for c in opp.hand if c not in opp.known_cards]
                hidden = [cid(c) for c in opp.hand
                          if c not in opp.known_cards]
                probes.append((len(events), unseen, hidden))
            action = legal[0] if len(legal) == 1 else \
                (policy0 if rnd.current_player_id == 0 else policy1).act(
                    game, legal)
            mover = rnd.current_player_id
            side = "hero" if mover == 0 else "opp"
            if isinstance(action, DrawCardAction):
                drawn = cid(rnd.dealer.stock_pile[-1])   # pops from END
                events.append((side + "_draw",
                               drawn if mover == 0 else None))
            elif isinstance(action, PickUpDiscardAction):
                events.append((side + "_pickup",
                               cid(rnd.dealer.discard_pile[-1])))
            elif isinstance(action, DiscardAction):
                events.append((side + "_discard", cid(action.card)))
            sim.step_fast(game, action)
        streams.append((events, probes, opp_size, hero_hand0))
    return streams


def _brier_probe(marg52, unseen, hidden, num_hidden):
    """Mean squared error over the unseen pool, plus the uniform
    baseline's -- the belief.py metric exactly."""
    p = np.array([marg52[c] for c in unseen])
    truth = np.isin(np.array(unseen), np.array(hidden)).astype(float)
    bm = float(np.mean((p - truth) ** 2))
    bu = float(np.mean((num_hidden / len(unseen) - truth) ** 2))
    return bm, bu


def _score_pf(streams, cfg, seed=0):
    """(mean particle Brier, mean uniform Brier) over all probes."""
    cfg = dict(cfg)
    cfg.setdefault("lik_floor", 0.10)
    model = make_model(cfg)
    n = cfg["n_particles"]
    kappa = cfg.get("kappa", 15.0)
    bm_sum, bu_sum, cnt = 0.0, 0.0, 0
    rng = np.random.default_rng(seed)
    for events, probes, opp_size, hero_hand0 in streams:
        core = ParticleCore(hero_hand0, opp_size, rng, model=model,
                            n_particles=n, kdraw=cfg.get("kdraw", 4),
                            mh_all_takes=cfg.get("mh_all_takes", False))
        cursor = 0
        for upto, unseen, hidden in probes:
            while cursor < upto:
                core.apply_event(*events[cursor])
                cursor += 1
            m = len(hidden)
            if not unseen or m == 0:
                continue
            marg = core.shrunk_marginal(unseen, m, kappa)
            bm, bu = _brier_probe(marg, unseen, hidden, m)
            bm_sum += bm
            bu_sum += bu
            cnt += 1
    return bm_sum / max(cnt, 1), bu_sum / max(cnt, 1)


def _score_belief_baseline(streams):
    """The current calibrated Belief on the same probes (its compressed
    opp-event view is derived from the full stream)."""
    bm_sum, bu_sum, cnt = 0.0, 0.0, 0
    for events, probes, opp_size, hero_hand0 in streams:
        b = Belief(hero_id=0)
        pile = []
        cursor = 0
        for upto, unseen, hidden in probes:
            while cursor < upto:
                kind, card = events[cursor]
                if kind == "opp_draw":
                    if pile:
                        b._evt_decline(pile[-1])
                elif kind == "opp_pickup":
                    pile.pop()
                    b._evt_opp_pickup(card)
                elif kind == "opp_discard":
                    pile.append(card)
                    b._evt_opp_discard(card)
                elif kind == "hero_pickup":
                    pile.pop()
                elif kind == "hero_discard":
                    pile.append(card)
                cursor += 1
            b._clamp()
            m = len(hidden)
            if not unseen or m == 0:
                continue
            marg = b.marginals(unseen, m)
            bm, bu = _brier_probe(marg, unseen, hidden, m)
            bm_sum += bm
            bu_sum += bu
            cnt += 1
    return bm_sum / max(cnt, 1), bu_sum / max(cnt, 1)


def _grid_job(job):
    cfg, streams_key = job
    return cfg, _score_pf(_STREAMS[streams_key], cfg)[0]


_STREAMS = {}                            # worker-side stream store


def _grid_worker_init(streams_train):
    _STREAMS["train"] = streams_train


def calibrate(num_hands=400, seed=61500, workers=12):
    t0 = time.time()
    print(f"logging {num_hands} teacher + {num_hands // 2} clone + "
          f"{num_hands // 2} random hands...", flush=True)
    streams_t = _log_streams_v6(num_hands, seed, "teacher")
    streams_c = _log_streams_v6(num_hands // 2, seed + 1, "clone")
    streams_r = _log_streams_v6(num_hands // 2, seed + 2, "random")
    half = lambda s: (s[:len(s) // 2], s[len(s) // 2:])
    tr_t, te_t = half(streams_t)
    tr_c, te_c = half(streams_c)
    tr_r, te_r = half(streams_r)
    train = tr_t + tr_c
    print(f"logged in {time.time() - t0:.0f}s", flush=True)

    grid = []
    for tau_d in (1.0, 2.0, 4.0):
        for p_knock in (0.0, 0.5, 0.9):
            for mu in (0.0, 0.3):
                # v7: tau_p enters the grid for the first time (it was
                # NEVER swept -- HANDOFF_V7 item 2). logistic/1.0 is the
                # promoted v6 pickup model; the rule rows sweep the
                # mismatch rate of the redesigned binary-rule model,
                # with the mh evidence-forgetting fix riding along.
                for pmode, tau_p in (("logistic", 1.0), ("rule", 0.05),
                                     ("rule", 0.1), ("rule", 0.2)):
                    cfg = dict(DEFAULT_PF)
                    cfg.update(tau_d=tau_d, p_knock=p_knock, mu=mu,
                               tau_p=tau_p, pickup_mode=pmode,
                               mh_all_takes=(pmode == "rule"),
                               n_particles=128)
                    grid.append(cfg)
    # train scores at N=128 for speed; held-out runs at the full default N
    print(f"grid: {len(grid)} configs on {len(train)} train hands "
          f"({workers} workers)", flush=True)
    t0 = time.time()
    with mp.Pool(workers, initializer=_grid_worker_init,
                 initargs=(train,)) as pool:
        results = pool.map(_grid_job, [(cfg, "train") for cfg in grid])
    best_cfg, best = None, 1e9
    for cfg, bm in results:
        tag = (f"tau_d={cfg['tau_d']} p_knock={cfg['p_knock']} "
               f"mu={cfg['mu']} {cfg['pickup_mode']}/tp={cfg['tau_p']}")
        print(f"  {tag:56s} train Brier {bm:.5f}", flush=True)
        if bm < best:
            best, best_cfg = bm, cfg
    print(f"grid done in {time.time() - t0:.0f}s", flush=True)

    best_cfg = dict(best_cfg)
    best_cfg["n_particles"] = DEFAULT_PF["n_particles"]
    lines = [f"v7 particle calibration ({num_hands} teacher hands, "
             f"grid of {len(grid)}; train Brier {best:.5f})",
             f"chosen: tau_d={best_cfg['tau_d']} mu={best_cfg['mu']} "
             f"tau_p={best_cfg['tau_p']} p_knock={best_cfg['p_knock']} "
             f"pickup_mode={best_cfg['pickup_mode']} "
             f"mh_all_takes={best_cfg['mh_all_takes']} "
             f"eps={best_cfg['eps']} N={best_cfg['n_particles']}"]
    for name, test in (("teacher", te_t), ("clone", te_c),
                       ("random", te_r)):
        pf_m, pf_u = _score_pf(test, best_cfg, seed=seed + 7)
        v6_m, _ = _score_pf(test, DEFAULT_PF, seed=seed + 7)
        bl_m, bl_u = _score_belief_baseline(test)
        lines.append(
            f"held-out vs {name:7s}: chosen {pf_m:.5f}  "
            f"v6-default {v6_m:.5f}  belief {bl_m:.5f}  "
            f"uniform {pf_u:.5f}  "
            f"chosen-vs-uniform {(pf_u - pf_m) / pf_u:+.2%}  "
            f"chosen-vs-v6 {(v6_m - pf_m) / v6_m:+.2%}")
    report = "\n".join(lines)
    print(report, flush=True)
    with open("results/v7_particle_calibration.txt", "w",
              encoding="utf-8") as fh:
        fh.write(report + "\n")
    return best_cfg


def worlds_gate(num_hands=150, seed=61700, samples=16, opponent="teacher",
                cfg=None):
    """THE integration gate: quality of sampled WORLDS -- the thing the
    searcher actually consumes -- for each sampler, on the three
    search-consumed statistics:
      overlap : mean |sample cap true hidden| / num_hidden
      dw-gap  : mean (deadwood(sample) - deadwood(true))  [realism;
                boost=2.5 was tuned to zero the EARLY-game gap but
                cannot reach late-game truth -- corr_sample.py]
      knockB  : Brier of P(sampled hand can knock) vs the truth
                indicator [feeds _world_weight + rollout termination]
    Stages = opponent turn count (pickups+declines proxy).
    """
    cfg = dict(DEFAULT_PF, **(cfg or {}))
    cfg["p_knock"] = 0.0                 # teacher streams: no knock model
    model = make_model(cfg)
    streams = _log_streams_v6(num_hands, seed, opponent)
    rng = np.random.default_rng(seed + 13)
    stages = ((0, 3), (4, 7), (8, 99))
    names = ["uniform", "bel b=0", "bel b=2.5", "bel auto", "particle"]
    acc = {n: {s: [0.0, 0.0, 0.0, 0] for s in stages} for n in names}

    def knock_capable(ids10):
        return min_deadwood_ids(list(ids10)) <= 10

    for events, probes, opp_size, hero0 in streams:
        core = ParticleCore(hero0, opp_size, rng, model=model,
                            n_particles=cfg["n_particles"],
                            kdraw=cfg["kdraw"],
                            mh_all_takes=cfg.get("mh_all_takes", False))
        bel = Belief(hero_id=0)
        pile = []
        cursor = 0
        for upto, unseen, hidden in probes:
            while cursor < upto:
                kind, card = events[cursor]
                if kind == "opp_draw" and pile:
                    bel._evt_decline(pile[-1])
                elif kind == "opp_pickup":
                    pile.pop()
                    bel._evt_opp_pickup(card)
                elif kind == "opp_discard":
                    pile.append(card)
                    bel._evt_opp_discard(card)
                elif kind == "hero_pickup":
                    pile.pop()
                elif kind == "hero_discard":
                    pile.append(card)
                core.apply_event(*events[cursor])
                cursor += 1
            bel._clamp()
            m = len(hidden)
            if not unseen or m == 0 or m != core.h:
                continue
            turns = bel.opp_pickups + bel.opp_declines
            stage = next(s for s in stages if s[0] <= turns <= s[1])
            known = sorted(core.opp_known)
            true_full = sorted(hidden) + known
            true_dw = min_deadwood_ids(true_full[:10]) \
                if len(true_full) == 10 else min_deadwood_ids(true_full)
            true_knock = 1.0 if true_dw <= 10 else 0.0
            hs = set(hidden)
            auto_boost = min(3.0, 0.5 * turns)
            for name in names:
                ov = dwg = pknock = 0.0
                for _ in range(samples):
                    if name == "uniform":
                        pick = rng.choice(len(unseen), size=m,
                                          replace=False)
                        samp = [unseen[i] for i in pick]
                    elif name == "bel b=0":
                        samp = bel.sample_free_cards(unseen, m, rng, 0.0)
                    elif name == "bel b=2.5":
                        samp = bel.sample_free_cards(unseen, m, rng, 2.5)
                    elif name == "bel auto":
                        samp = bel.sample_free_cards(unseen, m, rng,
                                                     auto_boost)
                    else:
                        samp = core.sample(rng)
                    full = sorted(samp) + known
                    dw = min_deadwood_ids(full)
                    ov += len(hs & set(samp)) / m
                    dwg += dw - true_dw
                    pknock += 1.0 if dw <= 10 else 0.0
                a = acc[name][stage]
                a[0] += ov / samples
                a[1] += dwg / samples
                a[2] += (pknock / samples - true_knock) ** 2
                a[3] += 1
    print(f"worlds gate: {num_hands} {opponent} hands, {samples} samples"
          f"/probe, PF cfg {cfg['tau_d']}/{cfg['mu']}/{cfg['lik_floor']}"
          f"/N{cfg['n_particles']}", flush=True)
    lines = []
    for name in names:
        row = f"  {name:10s}"
        for s in stages:
            a = acc[name][s]
            n = max(a[3], 1)
            row += (f" | t{s[0]}-{s[1]:<2d} ov {a[0]/n:.3f} "
                    f"dwg {a[1]/n:+5.1f} kB {a[2]/n:.3f}")
        lines.append(row)
        print(row, flush=True)
    return lines


def anchors():
    """HANDOFF_V7 sanity anchors for the pickup likelihood, on the
    hidden-5S incident's own cards: P(take 4S | holds 5S6S7S) must come
    out near 1, P(take | no use) near the floor, and a meld/extension
    take must NEVER score below a junk-dump take. Prints both modes;
    exits nonzero if the rule mode violates an anchor."""
    from sim import CARD_ID

    def h(*names):
        return [CARD_ID[n] for n in names]

    top4s = CARD_ID["4S"]
    # developed hand extending the 5S6S7S run (the incident's truth)
    run_hand = h("5S", "6S", "7S", "7H", "8H", "9H", "TH", "AH", "AD",
                 "2C")
    # junk hand dumping a king on the 4S (value swap, no meld)
    junk_hand = h("KS", "KH", "9D", "6C", "2H", "TD", "3C", "8S", "JD",
                  "QC")
    # hand with no use for the 4S at all (taking it gains nothing)
    nouse_hand = h("9S", "9H", "9D", "JS", "JH", "JD", "AC", "2C", "AH",
                   "2D")
    cases = [("meld/extension take (5S6S7S + 4S)", run_hand, True),
             ("junk-dump take (K out, 4S in)", junk_hand, True),
             ("no-use decline hand", nouse_hand, False)]
    results = {}
    for pmode, tau_p in (("logistic", 1.0),
                         ("rule", DEFAULT_PF["tau_p"])):
        model = make_model(dict(DEFAULT_PF, pickup_mode=pmode,
                                tau_p=tau_p))
        print(f"pickup_mode={pmode} tau_p={tau_p}:")
        for name, hand, _ in cases:
            p = model.pickup_prob(hand, top4s)
            results[(pmode, name)] = p
            print(f"  P(take 4S | {name:34s}) = {p:.3f}", flush=True)
    meld_p = results[("rule", cases[0][0])]
    junk_p = results[("rule", cases[1][0])]
    nouse_p = results[("rule", cases[2][0])]
    ok = meld_p > 0.9 and nouse_p < 0.15 and meld_p >= junk_p
    print(f"rule-mode anchors: meld {meld_p:.3f} (>0.9), no-use "
          f"{nouse_p:.3f} (<0.15), meld >= junk-dump "
          f"({meld_p:.3f} >= {junk_p:.3f}) -- "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    inv = results[("logistic", cases[0][0])] < \
        results[("logistic", cases[1][0])]
    print(f"v6 logistic inversion (meld < junk-dump): "
          f"{'CONFIRMED' if inv else 'not present'}", flush=True)
    return ok


def timing(num_hands=20, seed=61999):
    """Wall-clock per hand of the filter replay (the in-search cost
    driver -- flagged as the main cost risk in the plan)."""
    streams = _log_streams_v6(num_hands, seed, "teacher")
    cfg = dict(DEFAULT_PF)
    t0 = time.time()
    _score_pf(streams, cfg)
    dt = time.time() - t0
    print(f"particle replay: {dt / num_hands * 1000:.0f} ms/hand "
          f"(N={cfg['n_particles']}, {num_hands} hands)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("v6 particle filter calibration")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--worlds", action="store_true")
    ap.add_argument("--opponent", default="teacher")
    ap.add_argument("--time", dest="timing", action="store_true")
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=61500)
    args = ap.parse_args()
    if args.anchors:
        import sys
        sys.exit(0 if anchors() else 1)
    elif args.timing:
        timing()
    elif args.worlds:
        worlds_gate(min(args.hands, 200), args.seed + 200,
                    opponent=args.opponent)
    elif args.calibrate:
        calibrate(args.hands, args.seed, args.workers)
