import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
"""
The v5 action-value net: model, blob save/load, batched evaluator.

Roadmap #5 (HANDOFF_V5_TRAINING.md): distill the search by REGRESSION on its
per-candidate EV tables (RolloutExpert.analyze), not by classification on its
argmax actions -- the search's edge is rare deviations that argmax labels
smooth away (the v3-clone failure), while dense EV targets carry the 1-3 pt
candidate gaps directly.

Input: belief.build_obs (10, 52) float -- the IMPERFECT-info hero view, the
single source of truth for generation and every deployment path (GA1). The
net never sees a sampled world.

Output: a 110-dim head over the rlcard action space (2 draw, 3 pickup, 5 gin,
6..57 discard, 58..109 knock). Only labeled candidates carry training signal,
so deployment must argmax over a candidate set (label support), never over
all legal ids.

Target parameterization: Q[a] estimates 100 * (EV[a] - EV[teacher action])
in POINTS at the state -- the paired diff the search itself measures, whose
SE the labels store, and whose per-state constant cancels in ranking.
"""

import numpy as np

OBS_SHAPE = (10, 52)
INPUT_DIM = OBS_SHAPE[0] * OBS_SHAPE[1]     # 520
NUM_ACTIONS = 110


def flatten_obs(obs):
    """(10,52) or (N,10,52) -> float32 flat vector(s)."""
    obs = np.asarray(obs)
    if obs.ndim == 2:
        return obs.reshape(-1).astype(np.float32)
    return obs.reshape(obs.shape[0], -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Model (torch imported lazily: data-gen workers never need it).
# ---------------------------------------------------------------------------

def build_model(hidden=(512, 512, 256)):
    import torch.nn as nn
    layers, prev = [], INPUT_DIM
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, NUM_ACTIONS))
    return nn.Sequential(*layers)


def save_model(model, hidden, path, target="diff"):
    import torch
    torch.save({"state_dict": model.state_dict(), "hidden": tuple(hidden),
                "input_dim": INPUT_DIM, "num_actions": NUM_ACTIONS,
                "kind": "v5q", "target": target}, path)


def load_model(path, device="cpu"):
    import torch
    blob = torch.load(path, map_location=device, weights_only=False)
    assert blob.get("kind") == "v5q", "not a v5 Q-net blob"
    assert blob["input_dim"] == INPUT_DIM, "obs encoding drifted"
    assert blob["num_actions"] == NUM_ACTIONS
    model = build_model(blob["hidden"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model.to(device)


class QNet:
    """Batched Q evaluator. CPU with one thread by default (match workers);
    CUDA if asked."""

    def __init__(self, path, device="cpu"):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.device = device
        self.model = load_model(path, device)

    def q_values(self, obs_batch):
        """obs float (N,10,52) or (10,52) -> np.float32 (N,110) or (110,)."""
        x = flatten_obs(obs_batch)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        t = self.torch.from_numpy(x).to(self.device)
        with self.torch.no_grad():
            out = self.model(t)
        q = out.cpu().numpy().astype(np.float32)
        return q[0] if single else q

    def pick(self, obs, candidate_ids):
        """argmax Q over a candidate action-id list for one obs. Returns the
        index INTO candidate_ids (not the action id)."""
        q = self.q_values(obs)
        return int(np.argmax(q[np.asarray(candidate_ids)]))
