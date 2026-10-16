"""
Potential-based reward shaping for gin rummy.

Why this exists
---------------
RLCard's gin rummy rewards are terminal-only: every transition in a ~60-step
hand carries reward 0 except the last. Measured over four controlled runs
(see results/exp_*.csv), a DQN trained that way does not merely fail to
improve, it gets *worse* than its own random initialisation -- knock rate falls
from ~7% to ~2% and dead hands rise from 91% to 97%. The terminal return is
dominated by which cards you were dealt rather than by how you played them, so
Q collapses toward a near-constant and the argmax becomes arbitrary.

The fix is potential-based shaping (Ng, Harada & Russell 1999):

    F(s, a, s') = gamma * Phi(s') - Phi(s)

Adding F to the reward leaves the optimal policy unchanged for *any* choice of
Phi, so this buys a dense learning signal without redefining the game. The
true objective -- the real points scorer -- is untouched.

Phi(s) = -(deadwood of the hand in s) / 100, matching the reward scale. For an
11-card hand (between drawing and discarding) it is the deadwood you would have
after your best discard, which keeps the potential comparable across the two
half-turns instead of sawtoothing by one card's value.
"""

import numpy as np

from rlcard.games.gin_rummy.utils import melding
from rlcard.games.gin_rummy.utils import utils as gr_utils

_CACHE = {}
_CACHE_LIMIT = 200_000


def _min_deadwood(hand):
    """Deadwood under the best melding. Works for any hand size."""
    best = sum(gr_utils.get_deadwood_value(c) for c in hand)
    for cluster in melding.get_meld_clusters(hand=hand):
        melded = [c for meld in cluster for c in meld]
        value = sum(gr_utils.get_deadwood_value(c)
                    for c in hand if c not in melded)
        if value < best:
            best = value
    return best


def hand_potential(hand_plane):
    """Phi for the 52-bit hand plane of an observation (obs[0])."""
    key = np.packbits(np.asarray(hand_plane, dtype=np.uint8)).tobytes()
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    hand = gr_utils.decode_cards(hand_plane)
    if not hand:                       # terminal states encode an empty hand
        value = 0.0
    elif len(hand) > 10:
        # You still owe a discard: score the hand you would keep.
        value = -min(_min_deadwood([c for c in hand if c != drop])
                     for drop in hand) / 100.0
    else:
        value = -_min_deadwood(hand) / 100.0

    if len(_CACHE) < _CACHE_LIMIT:
        _CACHE[key] = value
    return value


def state_potential(state):
    if state is None:
        return 0.0
    obs = state["obs"] if isinstance(state, dict) else state
    return hand_potential(obs[0])


def shape_trajectory(trajectory, gamma):
    """Add gamma*Phi(s') - Phi(s) to each transition's reward, in place.

    `trajectory` is the output of rlcard's `reorganize`: each entry is
    [state, action, reward, next_state, done].
    """
    for transition in trajectory:
        state, _, reward, next_state, done = transition
        # Phi(terminal) must be 0 for the guarantee to hold.
        phi_next = 0.0 if done else state_potential(next_state)
        transition[2] = reward + gamma * phi_next - state_potential(state)
    return trajectory
