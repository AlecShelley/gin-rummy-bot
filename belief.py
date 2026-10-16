"""
Opponent-hand belief tracking and the history-aware observation.

The rlcard observation shows WHERE cards are now (5 planes) but nothing about
HOW they got there. Yet the move history is informative about the opponent's
hidden cards in exactly the way poker betting is informative about a range:

  * a card the opponent DECLINED to pick up argues against them collecting
    cards that meld with it;
  * a card they discarded argues (more weakly -- it may have been forced)
    against its meld neighbors;
  * a card they picked up argues FOR its meld neighbors.

`Belief` turns those events into per-card weights over the unseen pool via
O(52) multiplicative updates, replayed incrementally from `round.move_sheet`.
The sheet is perfect-information (it records the shuffled deck and every stock
draw), so the replay reads ONLY information-set-legal facts: who moved, the
move type, and cards that were publicly visible at the time.

`build_obs` assembles the (10, 52) observation used by the v3 network. It is
the single source of truth: hist_env.HistoryGinRummyEnv and the parallel data
generator both call it, so training and deployment observations are bit-exact
by construction (gate GA1).

The weight parameters are calibrated with `--calibrate`: hands are logged once
as compressed event streams with ground truth, then a grid of (alpha, beta,
gamma) is scored on held-out Brier against the uniform-over-unseen baseline
(gate GA2).

    python belief.py --calibrate
"""

import argparse

import numpy as np

from rlcard.games.gin_rummy.utils.move import (
    DealHandMove, DiscardMove, DrawCardMove, PickupDiscardMove)

from sim import (CARD_ID, DECK, HomePassMove, cid, ids_of,
                 min_deadwood_ids)

# Calibrated by --calibrate (results/v3_belief_calibration.txt): grid-searched
# on train hands, then held out -- Brier 12.5% better than uniform-over-unseen
# against both teacher and v2-clone opponents. Discards carry MORE evidence
# than declined upcards here (beta > alpha), the reverse of the initial guess.
DEFAULT_PARAMS = dict(alpha=0.25, beta=0.35, gamma=0.5)

# Precomputed meld-adjacency lists: adj[c] = ids that could ever share a meld
# with c (same rank, or same suit within two ranks).
_ADJ = [[] for _ in range(52)]
for _a in range(52):
    for _b in range(52):
        if _a != _b:
            if _a % 13 == _b % 13 and _a // 13 != _b // 13:
                _ADJ[_a].append(_b)
            elif _a // 13 == _b // 13 and abs(_a % 13 - _b % 13) in (1, 2):
                _ADJ[_a].append(_b)

W_MIN, W_MAX = 0.05, 20.0


class Belief:
    """Per-card weights over the opponent's hidden cards, one hero perspective.

    Weights live on all 52 cards; only their values on the current unseen pool
    matter, and membership in that pool is always taken from the live game, so
    a card leaving the pool (e.g. it gets discarded) needs no special-casing.
    """

    def __init__(self, hero_id, alpha=None, beta=None, gamma=None):
        p = DEFAULT_PARAMS
        self.hero_id = hero_id
        self.alpha = p["alpha"] if alpha is None else alpha
        self.beta = p["beta"] if beta is None else beta
        self.gamma = p["gamma"] if gamma is None else gamma
        self.w = np.ones(52)
        self.opp_declined = np.zeros(52, dtype=np.float32)   # multi-hot planes
        self.opp_discarded = np.zeros(52, dtype=np.float32)
        self.hero_discarded = np.zeros(52, dtype=np.float32)
        self.opp_pickups = 0
        self.opp_declines = 0
        self._processed = 0              # move_sheet index we've consumed
        self._pile = []                  # replayed discard pile, as card ids

    # -- event handlers (id-space; also driven directly by the calibrator) --

    def _evt_decline(self, upcard_id):
        self.opp_declines += 1
        self.opp_declined[upcard_id] = 1.0
        self.w[_ADJ[upcard_id]] *= (1.0 - self.alpha)

    def _evt_opp_discard(self, card_id):
        self.opp_discarded[card_id] = 1.0
        self.w[_ADJ[card_id]] *= (1.0 - self.beta)

    def _evt_opp_pickup(self, card_id):
        self.opp_pickups += 1
        self.w[_ADJ[card_id]] *= (1.0 + self.gamma)

    def _clamp(self):
        np.clip(self.w, W_MIN, W_MAX, out=self.w)

    def update(self, game):
        """Consume any move_sheet entries added since the last call.

        Only information-set-legal facts are read: move types, mover identity,
        and the replayed public discard pile. DrawCardMove.card (the hidden
        drawn card) and the deal are never touched.
        """
        # Home-variant flip: the pile's founding card is nobody's discard, so
        # it never appears in move_sheet -- seed the replayed pile with it or
        # a pickup of the flip would desync (empty pop). No-op elsewhere.
        if self._processed == 0 and not self._pile:
            flip_id = getattr(game.round, "_home_flip_id", None)
            if flip_id is not None:
                self._pile.append(flip_id)
        sheet = game.round.move_sheet
        for i in range(self._processed, len(sheet)):
            move = sheet[i]
            if isinstance(move, DealHandMove):
                continue
            is_opp = move.player.player_id != self.hero_id
            if isinstance(move, DrawCardMove):
                if is_opp and self._pile:
                    self._evt_decline(self._pile[-1])
            elif isinstance(move, HomePassMove):
                # opening pass (home rule 2026-09-01): public decline of
                # the flip; no card moves
                if is_opp and self._pile:
                    self._evt_decline(self._pile[-1])
            elif isinstance(move, PickupDiscardMove):
                taken = self._pile.pop()
                if is_opp:
                    self._evt_opp_pickup(taken)
            elif isinstance(move, DiscardMove):
                card_id = cid(move.action.card)
                self._pile.append(card_id)
                if is_opp:
                    self._evt_opp_discard(card_id)
                else:
                    self.hero_discarded[card_id] = 1.0
            # Knock/Gin/DeclareDeadHand/Score moves end the hand; the knock or
            # gin discard is NOT public until showdown and never joins the pile.
        self._clamp()
        self._processed = len(sheet)

    # -- outputs ------------------------------------------------------------

    def marginals(self, unseen_ids, num_hidden, known_ids=()):
        """P(opponent holds c) as a 52-vector: 1 for their known cards, a
        first-order inclusion probability on the unseen pool, 0 elsewhere."""
        p = np.zeros(52, dtype=np.float32)
        if unseen_ids:
            w = self.w[unseen_ids]
            p[unseen_ids] = np.minimum(1.0, num_hidden * w / max(w.sum(), 1e-9))
        for i in known_ids:
            p[i] = 1.0
        return p

    def sample_free_cards(self, unseen_ids, num_hidden, rng, boost=0.0):
        """One weighted sample (without replacement) of the opponent's hidden
        cards. boost=0: independent Efraimidis-Spirakis keys, vectorized --
        the historical sampler. boost>0: sequential, each pick multiplying the
        remaining meld-neighbor weights by (1 + boost) -- correlated worlds.

        Why boost exists (measured, corr_sample.py, 2026-08-23): real hidden
        opponent hands average ~23 deadwood but independent samples average
        ~42 -- structurally meldless, so the search underestimates opponent
        progress. boost=2.5 matches the true mean deadwood (23.1 vs 23.3)."""
        if boost <= 0.0:
            w = self.w[unseen_ids]
            keys = rng.random(len(unseen_ids)) ** (1.0 / np.maximum(w, 1e-9))
            top = np.argpartition(-keys, num_hidden - 1)[:num_hidden] \
                if num_hidden < len(unseen_ids) else np.arange(len(unseen_ids))
            return [unseen_ids[i] for i in top]
        if num_hidden >= len(unseen_ids):
            return list(unseen_ids)
        pool = np.array(unseen_ids)
        w = self.w[pool].astype(np.float64).copy()
        picked = []
        for _ in range(num_hidden):
            k = int(rng.choice(len(pool), p=w / w.sum()))
            c = int(pool[k])
            picked.append(c)
            keep = np.arange(len(pool)) != k
            pool, w = pool[keep], w[keep]
            w[np.isin(pool, _ADJ[c])] *= (1.0 + boost)
        return picked


def base_obs(game):
    """rlcard's stock (5, 52) observation for the current player, built from
    the game directly. Lets base-obs agents (the v2 clone) play inside
    engine-driven loops that never touch an Env."""
    rnd = game.round
    pile = rnd.dealer.discard_pile
    cur = rnd.players[rnd.current_player_id]
    opp = rnd.players[1 - rnd.current_player_id]
    obs = np.zeros((5, 52))
    obs[0][ids_of(cur.hand)] = 1
    if pile:
        obs[1][cid(pile[-1])] = 1
        obs[2][ids_of(pile[:-1])] = 1
    obs[3][ids_of(opp.known_cards)] = 1
    unseen = ids_of(rnd.dealer.stock_pile) + [
        cid(c) for c in opp.hand if c not in opp.known_cards]
    obs[4][unseen] = 1
    return obs


# ---------------------------------------------------------------------------
# The (10, 52) observation.
# ---------------------------------------------------------------------------

NUM_PLANES = 10
# Plane 9 layout: position-coded scalars, columns:
#   0 stock/52   1 len(move_sheet)/200   2 opp pickups/10   3 opp declines/13
#   4 knock legal   5 phase (0 draw / 1 discard)   6 own deadwood/100
#   7 opp known count/10   (rest zero)
SCALAR_COLS = 8


def build_obs(game, player_id, belief, legal_ids=None):
    """The v3 observation for `player_id`, as float32 (10, 52).

    Planes 0-4 are bit-identical to rlcard's (hand, top discard, dead cards,
    opponent known, unknown). 5 belief marginals; 6 upcards the opponent
    declined; 7 cards the opponent discarded; 8 cards the hero discarded;
    9 position-coded scalars (see SCALAR_COLS comment).

    `legal_ids` (action ids at this node) feeds the knock-legal flag; pass it
    when the node is the hero's decision, omit for spectator views.
    """
    rnd = game.round
    hero = rnd.players[player_id]
    opp = rnd.players[1 - player_id]
    discard_pile = rnd.dealer.discard_pile

    belief.update(game)

    obs = np.zeros((NUM_PLANES, 52), dtype=np.float32)
    hand_ids = ids_of(hero.hand)
    obs[0][hand_ids] = 1.0
    if discard_pile:
        obs[1][cid(discard_pile[-1])] = 1.0
        for c in discard_pile[:-1]:
            obs[2][cid(c)] = 1.0
    known_ids = ids_of(opp.known_cards)
    obs[3][known_ids] = 1.0
    unseen_ids = ids_of(rnd.dealer.stock_pile)
    unseen_ids += [cid(c) for c in opp.hand if c not in opp.known_cards]
    obs[4][unseen_ids] = 1.0

    num_hidden = len(opp.hand) - len(opp.known_cards)
    obs[5] = belief.marginals(unseen_ids, num_hidden, known_ids)
    obs[6] = belief.opp_declined
    obs[7] = belief.opp_discarded
    obs[8] = belief.hero_discarded

    if len(hand_ids) == 11:
        dw = min(min_deadwood_ids(hand_ids[:k] + hand_ids[k + 1:])
                 for k in range(11))
        phase = 1.0
    else:
        dw = min_deadwood_ids(hand_ids)
        phase = 0.0
    obs[9][0] = len(rnd.dealer.stock_pile) / 52.0
    obs[9][1] = len(rnd.move_sheet) / 200.0
    obs[9][2] = belief.opp_pickups / 10.0
    obs[9][3] = belief.opp_declines / 13.0
    obs[9][4] = 1.0 if legal_ids and any(a >= 58 for a in legal_ids) else 0.0
    obs[9][5] = phase
    obs[9][6] = dw / 100.0
    obs[9][7] = len(known_ids) / 10.0
    return obs


# ---------------------------------------------------------------------------
# Calibration: log event streams with ground truth once, then grid-search the
# evidence weights on Brier score. Pure counting -- no engine in the loop.
# ---------------------------------------------------------------------------

def _log_streams(num_hands, seed, opponent="teacher"):
    """Play hands, recording per-hand: the hero-view event stream and, at each
    hero decision, (unseen pool, hidden count, truly-held hidden ids).

    Hero is seat 0's view of seat 1. `opponent` picks seat 1's policy:
    teacher (FastTeacher) or clone (the v2 network via its env-free probs).
    """
    import random as pyrandom
    import sim

    policy0 = sim.FastTeacher()
    if opponent == "teacher":
        policy1 = sim.FastTeacher()
    else:                                # v2 clone, driven without an Env
        import torch
        from rlcard.utils.utils import remove_illegal
        from gin_rummy_bot import GinRummyBot
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
        events, probes = [], []
        pile = []
        while game.round.going_out_action is None and not game.round.is_over:
            legal = sim.fast_legal_actions(game)
            rnd = game.round
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
            # Mirror Belief.update's event extraction, hero = player 0.
            from rlcard.games.gin_rummy.utils.action_event import (
                DiscardAction, DrawCardAction, PickUpDiscardAction)
            mover = rnd.current_player_id
            if isinstance(action, DrawCardAction):
                if mover == 1 and pile:
                    events.append(("decline", pile[-1]))
            elif isinstance(action, PickUpDiscardAction):
                taken = pile.pop()
                if mover == 1:
                    events.append(("pickup", taken))
            elif isinstance(action, DiscardAction):
                card_id = cid(action.card)
                pile.append(card_id)
                if mover == 1:
                    events.append(("discard", card_id))
            sim.step_fast(game, action)
        streams.append((events, probes))
    return streams


def _score_params(streams, alpha, beta, gamma):
    """Mean Brier of the belief marginal vs truth, and the uniform baseline."""
    brier_model, brier_uniform, n = 0.0, 0.0, 0
    for events, probes in streams:
        b = Belief(hero_id=0, alpha=alpha, beta=beta, gamma=gamma)
        cursor = 0
        for upto, unseen, hidden in probes:
            while cursor < upto:
                kind, card_id = events[cursor]
                if kind == "decline":
                    b._evt_decline(card_id)
                elif kind == "pickup":
                    b._evt_opp_pickup(card_id)
                else:
                    b._evt_opp_discard(card_id)
                cursor += 1
            b._clamp()
            m = len(hidden)
            if not unseen or m == 0:
                continue
            p = b.marginals(unseen, m)[unseen]
            truth = np.isin(np.array(unseen), np.array(hidden)).astype(float)
            brier_model += float(np.mean((p - truth) ** 2))
            brier_uniform += float(np.mean((m / len(unseen) - truth) ** 2))
            n += 1
        # NOTE: probes within one hand share a stream; cursor never rewinds.
    return brier_model / n, brier_uniform / n


def calibrate(num_hands=1500, seed=515):
    grid = [0.0, 0.15, 0.25, 0.35, 0.5]
    print(f"logging {num_hands} teacher hands + {num_hands // 3} clone hands...",
          flush=True)
    streams_t = _log_streams(num_hands, seed, "teacher")
    streams_c = _log_streams(num_hands // 3, seed + 1, "clone")
    half_t, half_c = len(streams_t) // 2, len(streams_c) // 2
    train = streams_t[:half_t] + streams_c[:half_c]
    test_t, test_c = streams_t[half_t:], streams_c[half_c:]

    best, best_cfg = 1e9, None
    for a in grid:
        for bta in grid:
            for g in grid:
                bm, _ = _score_params(train, a, bta, g)
                if bm < best:
                    best, best_cfg = bm, (a, bta, g)
    a, bta, g = best_cfg
    bt_m, bt_u = _score_params(test_t, a, bta, g)
    bc_m, bc_u = _score_params(test_c, a, bta, g)
    lines = [
        f"belief calibration over grid {grid} (train Brier {best:.5f})",
        f"chosen alpha={a} beta={bta} gamma={g}",
        f"held-out vs teacher: model {bt_m:.5f}  uniform {bt_u:.5f}  "
        f"improvement {(bt_u - bt_m) / bt_u:+.2%}",
        f"held-out vs clone  : model {bc_m:.5f}  uniform {bc_u:.5f}  "
        f"improvement {(bc_u - bc_m) / bc_u:+.2%}",
    ]
    report = "\n".join(lines)
    print(report)
    with open("results/v3_belief_calibration.txt", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    return best_cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser("belief calibration")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--hands", type=int, default=1500)
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.hands)
