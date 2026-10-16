"""
The learned value leaf: features, model, batched evaluator.

Roadmap #1. The expert truncates rollouts and scores the frontier with
`sim.leaf_value` -- a handcrafted effective-deadwood difference. This module
replaces that leaf with a net trained by REGRESSION on real playout outcomes
(regression averages noisy outcomes; policy classification on the same search
failed -- see the handoff).

What the net estimates: V(s) = E[final full-rules payoff of the CURRENT
player | perfect-information state s, both seats played by FastTeacher to the
end]. That is *exactly* the expectation the truncation leaf stands in for --
rollout states are determinized, so the leaf may legally see both hands, and
the continuation policy in `RolloutExpert.playout` is FastTeacher for both
seats. Data generation (value_gen.py) therefore rolls in with diverse
policies but always finishes hands teacher-vs-teacher, labeling only
post-switch states.

`leaf_features` is the single source of truth for the input encoding:
generation and deployment both call it, so training and inference features
are bit-exact by construction (the GA1 lesson from v3).

Input encoding (hero = current player to act):
  planes uint8 (6, 52): hero hand; opp hand; dead cards (pile minus top);
    top discard; hero cards known to opp; opp cards known to hero.
    (Known-cards planes matter because FastTeacher's discard safety tiebreak
    reads them -- the continuation policy sees them, so V must too.)
  scalars float32 (8,): stock/52, phase(11 cards=1), hero dw/100, opp dw/100,
    hero residual/100, opp residual/100, handcrafted leaf (hero view),
    len(move_sheet)/200.
  The residual/leaf scalars hand the net the current heuristic as a feature:
  it starts from the shipped leaf and learns corrections.

Label/output scale: payoff units (points / 100), hero perspective.
"""

import numpy as np

from sim import _best10, cid, ids_of, min_deadwood_ids

NUM_PLANES = 6
NUM_SCALARS = 8
INPUT_DIM = NUM_PLANES * 52 + NUM_SCALARS


def leaf_features(game):
    """Perfect-information features of `game` from the current player's
    perspective. Returns (planes uint8 (6,52), scalars float32 (8,))."""
    from fullrules import defender_residual
    rnd = game.round
    hero_id = rnd.current_player_id
    hero = rnd.players[hero_id]
    opp = rnd.players[1 - hero_id]
    pile = rnd.dealer.discard_pile

    planes = np.zeros((NUM_PLANES, 52), dtype=np.uint8)
    hero_ids = ids_of(hero.hand)
    opp_ids = ids_of(opp.hand)
    planes[0][hero_ids] = 1
    planes[1][opp_ids] = 1
    if pile:
        planes[3][cid(pile[-1])] = 1
        for c in pile[:-1]:
            planes[2][cid(c)] = 1
    planes[4][ids_of(hero.known_cards)] = 1
    planes[5][ids_of(opp.known_cards)] = 1

    h_hero, h_opp = _best10(hero_ids), _best10(opp_ids)
    dw_hero = min_deadwood_ids(h_hero)
    dw_opp = min_deadwood_ids(h_opp)
    eff_hero = defender_residual(h_opp, h_hero)   # my residual if they knock
    eff_opp = defender_residual(h_hero, h_opp)
    scalars = np.array([
        len(rnd.dealer.stock_pile) / 52.0,
        1.0 if len(hero_ids) == 11 else 0.0,
        dw_hero / 100.0,
        dw_opp / 100.0,
        eff_hero / 100.0,
        eff_opp / 100.0,
        (eff_opp - eff_hero) / 100.0,             # the handcrafted leaf, hero view
        len(rnd.move_sheet) / 200.0,
    ], dtype=np.float32)
    return planes, scalars


def flatten(planes, scalars):
    """(planes, scalars) -> float32 flat vector(s) for the net. Accepts both
    single samples ((6,52),(8,)) and batches ((N,6,52),(N,8))."""
    if planes.ndim == 2:
        return np.concatenate([planes.reshape(-1).astype(np.float32),
                               scalars.astype(np.float32)])
    return np.concatenate(
        [planes.reshape(planes.shape[0], -1).astype(np.float32),
         scalars.astype(np.float32)], axis=1)


# ---------------------------------------------------------------------------
# Model (torch imported lazily: data-gen workers never need it).
# ---------------------------------------------------------------------------

def build_model(hidden=(512, 512, 256)):
    import torch.nn as nn
    layers, prev = [], INPUT_DIM
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def save_model(model, hidden, path):
    import torch
    torch.save({"state_dict": model.state_dict(), "hidden": tuple(hidden),
                "input_dim": INPUT_DIM}, path)


def load_model(path, device="cpu"):
    import torch
    blob = torch.load(path, map_location=device, weights_only=False)
    assert blob["input_dim"] == INPUT_DIM, "feature encoding drifted"
    model = build_model(blob["hidden"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model.to(device)


class ValueLeaf:
    """Batched leaf evaluator for the search: collect leaf states, evaluate
    them in one forward pass. CPU by default (workers), CUDA if asked."""

    def __init__(self, path, device="cpu"):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.device = device
        self.model = load_model(path, device)

    def evaluate(self, feature_list):
        """feature_list: [(planes, scalars), ...] -> np.float32 (N,) hero-view
        payoffs."""
        if not feature_list:
            return np.zeros(0, dtype=np.float32)
        planes = np.stack([p for p, _ in feature_list])
        scalars = np.stack([s for _, s in feature_list])
        x = self.torch.from_numpy(flatten(planes, scalars)).to(self.device)
        with self.torch.no_grad():
            out = self.model(x).squeeze(1)
        return out.cpu().numpy().astype(np.float32)
